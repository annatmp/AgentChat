"""
Run configuration — one YAML file describing everything that shapes a run.

Anything that affects a conversation's outcome (roster, strategy, prompts,
turn budget, temperature, seed, models) is declared here and recorded with the
run. Nothing that matters is hardcoded in `main.py`.

`resolve()` turns a config into a `ResolvedRun`: the actual agents, prompt text,
a hash of every input file, and the `run_id` derived from all of it. It also
fails fast on the mistakes that would otherwise surface mid-grid — a roster name
with no YAML, an unknown provider, or a temperature on a model that rejects one.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from agent_chat.agents import PROVIDERS, TEMPERATURE_REJECTED, Agent, load_roster
from agent_chat.records import SCHEMA_VERSION, compute_run_id, file_sha256


class ConfigError(ValueError):
    """A run config that cannot be executed as written."""


@dataclass
class StrategyConfig:
    name: str = "round_robin"
    params: dict = field(default_factory=dict)


@dataclass
class SummarizerConfig:
    """
    The neutral summarizer: fixed model, same across every condition, never a
    participant, and outside the turn budget.
    """
    model: str = "deepseek-chat"
    provider: str = "deepseek"
    temperature: float | None = 0.0
    max_tokens: int = 4096
    role_prompt: str = "prompts/summarizer_role.txt"


@dataclass
class RunConfig:
    name: str
    roster: list[str] = field(default_factory=list)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    summarizer: SummarizerConfig = field(default_factory=SummarizerConfig)
    task_prompt: str = "prompts/prompt_complex.txt"
    system_prompt: str = "prompts/system_prompt.txt"
    summarize_prompt: str = "prompts/summarize_prompt.txt"
    turn_budget: int = 24
    seed: int = 0
    temperature: float | None = 0.7   # overrides per-agent temperature when set
    role_knowledge: bool = True       # factor D in EXPERIMENT_DESIGN §1
    panel: dict | None = None         # {role: {model, provider, ...}} overrides
    agents_dir: str = "agents"
    output_dir: str = "runs"


@dataclass
class ResolvedRun:
    config: RunConfig
    agents: dict[str, Agent]
    summarizer: Agent
    knowledge: dict[str, str]
    system_prompt: str
    task_prompt: str
    file_hashes: dict[str, str]
    run_id: str
    rng: random.Random

    def config_dict(self) -> dict:
        """The fully resolved config, as stored in the run record."""
        return {
            **asdict(self.config),
            "resolved_agents": {
                name: {
                    "model": agent.model,
                    "provider": agent.provider,
                    "max_tokens": agent.max_tokens,
                    "temperature": self.config.temperature
                    if self.config.temperature is not None else agent.temperature,
                    "knowledge": agent.knowledge if self.config.role_knowledge else None,
                }
                for name, agent in self.agents.items()
            },
            "resolved_summarizer": {
                "model": self.summarizer.model,
                "provider": self.summarizer.provider,
                "temperature": self.summarizer.temperature,
            },
        }

    def output_path(self) -> Path:
        return Path(self.config.output_dir) / f"{self.run_id}.json"


# --- Loading ---

def _construct(cls, data: dict | None, where: str):
    data = data or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    unknown = set(data) - set(cls.__dataclass_fields__)
    if unknown:
        known = ", ".join(sorted(cls.__dataclass_fields__))
        raise ConfigError(
            f"{where}: unknown keys {', '.join(sorted(unknown))}. Known keys: {known}"
        )
    return cls(**data)


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no run config at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    strategy = _construct(StrategyConfig, data.pop("strategy", None), f"{path}: strategy")
    summarizer = _construct(SummarizerConfig, data.pop("summarizer", None), f"{path}: summarizer")
    data.setdefault("name", path.stem)
    config = _construct(RunConfig, data, str(path))
    config.strategy = strategy
    config.summarizer = summarizer
    return config


# --- Validation ---

def _check_provider(label: str, provider: str) -> None:
    if provider not in PROVIDERS:
        raise ConfigError(
            f"{label}: unknown provider {provider!r}. Known: {', '.join(PROVIDERS)}"
        )


def _check_temperature(label: str, model: str, temperature: float | None) -> None:
    if temperature is not None and model in TEMPERATURE_REJECTED:
        raise ConfigError(
            f"{label}: {model} rejects `temperature` with a 400 error, but this run "
            f"sets temperature={temperature}. Set `temperature: null` in the run "
            f"config, or use a model that accepts it (e.g. claude-sonnet-4-6)."
        )


def _apply_panel(agents: dict[str, Agent], panel: dict | None) -> None:
    """Override per-role model/provider, e.g. to run a homogeneous panel."""
    for role, override in (panel or {}).items():
        if role not in agents:
            raise ConfigError(
                f"panel names {role!r}, which is not in the roster "
                f"({', '.join(sorted(agents))})"
            )
        if not isinstance(override, dict):
            raise ConfigError(f"panel.{role}: expected a mapping of Agent fields")
        for key, value in override.items():
            if key not in ("model", "provider", "max_tokens", "temperature"):
                raise ConfigError(
                    f"panel.{role}: cannot override {key!r}; "
                    "allowed: model, provider, max_tokens, temperature"
                )
            setattr(agents[role], key, value)


# --- Resolution ---

def _read(path: str | Path, label: str) -> str:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"{label}: no such file: {path}")
    return path.read_text()


def _run_id_payload(config: RunConfig, agents: dict[str, Agent],
                    summarizer: Agent, file_hashes: dict[str, str]) -> dict:
    """
    Everything that could change the conversation's outcome.

    `name` and `output_dir` are deliberately excluded: relabelling a config or
    writing it elsewhere is the same experimental cell and must keep the same
    run_id, or P3's "skip runs whose output exists" resume check breaks.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "roster": list(config.roster),
        "strategy": asdict(config.strategy),
        "turn_budget": config.turn_budget,
        "seed": config.seed,
        "temperature": config.temperature,
        "role_knowledge": config.role_knowledge,
        "prompts": {
            "task": config.task_prompt,
            "system": config.system_prompt,
            "summarize": config.summarize_prompt,
        },
        "agents": {
            name: {
                "model": agent.model,
                "provider": agent.provider,
                "max_tokens": agent.max_tokens,
                "temperature": agent.temperature,
            }
            for name, agent in agents.items()
        },
        "summarizer": {
            "model": summarizer.model,
            "provider": summarizer.provider,
            "temperature": summarizer.temperature,
            "max_tokens": summarizer.max_tokens,
        },
        "file_hashes": file_hashes,
    }


def resolve(config: RunConfig) -> ResolvedRun:
    """Load everything the config names, validate it, and derive the run_id."""
    if not config.roster:
        raise ConfigError("roster is empty — a run needs an explicit list of agents")
    duplicates = {n for n in config.roster if config.roster.count(n) > 1}
    if duplicates:
        raise ConfigError(f"roster lists duplicates: {', '.join(sorted(duplicates))}")

    agents = load_roster(config.agents_dir, config.roster)
    _apply_panel(agents, config.panel)

    # Private per-role context, off entirely when role_knowledge is false.
    knowledge: dict[str, str] = {}
    if config.role_knowledge:
        for name, agent in agents.items():
            if agent.knowledge:
                knowledge[name] = _read(agent.knowledge, f"{name}.knowledge").strip()

    system_prompt = _read(config.system_prompt, "system_prompt").strip()
    task_prompt = _read(config.task_prompt, "task_prompt")
    summarize_template = _read(config.summarize_prompt, "summarize_prompt")
    if "{transcript}" not in summarize_template:
        raise ConfigError(f"{config.summarize_prompt}: must contain a {{transcript}} placeholder")

    summarizer = Agent(
        name="summarizer",
        role=_read(config.summarizer.role_prompt, "summarizer.role_prompt").strip(),
        model=config.summarizer.model,
        provider=config.summarizer.provider,
        max_tokens=config.summarizer.max_tokens,
        temperature=config.summarizer.temperature,
    )
    if summarizer.name in agents:
        raise ConfigError("'summarizer' cannot also be a roster agent — it must be neutral")

    for name, agent in agents.items():
        _check_provider(name, agent.provider)
        effective = config.temperature if config.temperature is not None else agent.temperature
        _check_temperature(name, agent.model, effective)
    _check_provider("summarizer", summarizer.provider)
    _check_temperature("summarizer", summarizer.model, summarizer.temperature)

    # Hash every input file, so results can be told apart after a prompt tweak.
    hashed = [
        config.system_prompt, config.task_prompt, config.summarize_prompt,
        config.summarizer.role_prompt,
        *(str(Path(config.agents_dir) / f"{n}.yaml") for n in config.roster),
        *(a.knowledge for a in agents.values() if config.role_knowledge and a.knowledge),
    ]
    file_hashes = {Path(p).as_posix(): file_sha256(p) for p in hashed}

    run_id = compute_run_id(_run_id_payload(config, agents, summarizer, file_hashes))
    return ResolvedRun(
        config=config,
        agents=agents,
        summarizer=summarizer,
        knowledge=knowledge,
        system_prompt=system_prompt,
        task_prompt=task_prompt,
        file_hashes=file_hashes,
        run_id=run_id,
        rng=random.Random(config.seed),
    )


def load(path: str | Path) -> ResolvedRun:
    return resolve(load_run_config(path))
