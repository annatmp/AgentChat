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
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import yaml

from agent_chat.agents import PROVIDERS, TEMPERATURE_REJECTED, Agent, load_agent, load_roster
from agent_chat.records import SCHEMA_VERSION, compute_run_id, file_sha256
from agent_chat.strategies import REGISTRY as STRATEGY_REGISTRY


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
    # Injected into both system_prompt and summarizer.role_prompt wherever each
    # contains a {user_story_template} placeholder — one shared source for the
    # story format both the conversation and the summarizer must follow, so the
    # two can't drift out of sync with each other.
    user_story_template: str = "prompts/user_story_template.txt"
    consensus_stop: bool = True        # any agent can end the meeting early once every agent agrees
    consensus_prompt: str = "prompts/consensus_prompt.txt"
    turn_budget: int = 24
    seed: int = 0
    temperature: float | None = 0.7   # overrides per-agent temperature when set
    role_knowledge: bool = True       # factor D in EXPERIMENT_DESIGN §1
    # Solo baseline: union every role's own knowledge onto the roster, discovered
    # from agents_dir rather than a separately maintained list, so it can never
    # drift from what the team actually knows.
    all_role_knowledge: bool = False
    # Solo baseline: inject "REVIEW ROUND N" framing before each turn. Round 1 gets
    # its own template — there is nothing yet to "continue refining."
    review_rounds: bool = False
    review_round_template: str = "REVIEW ROUND {n}: continue refining the plan from where you left off."
    review_round_first_template: str = "REVIEW ROUND 1: come up with an initial plan for the request above — there is nothing to refine yet."
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
    consensus_prompt: str  # "" when config.consensus_stop is False
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


def apply_overrides(
    config: RunConfig, *, strategy: str | None = None, seed: int | None = None,
    experiment_id: str | None = None,
) -> RunConfig:
    """
    Return a new config with a grid cell's overrides applied, params otherwise kept.

    The one place that knows how a grid cell overrides a base panel config —
    `main.py`'s `--strategy`/`--seed`/`--experiment-id` flags and the harness's
    own pre-resolution pass both call this, so they can't drift apart.

    `strategy`/`seed` feed `_run_id_payload`, so overriding them is automatically
    reflected in `run_id` and the written record's `config` with no extra
    plumbing. `experiment_id` only reshapes `output_dir` (already excluded from
    `run_id` on purpose — see `_run_id_payload`), so it never perturbs identity.
    """
    if strategy is None and seed is None and experiment_id is None:
        return config
    updates: dict = {}
    if strategy is not None:
        updates["strategy"] = replace(config.strategy, name=strategy)
    if seed is not None:
        updates["seed"] = seed
    if experiment_id is not None:
        updates["output_dir"] = str(Path(config.output_dir) / experiment_id)
    return replace(config, **updates)


# --- Validation ---

def _check_provider(label: str, provider: str) -> None:
    if provider not in PROVIDERS:
        raise ConfigError(
            f"{label}: unknown provider {provider!r}. Known: {', '.join(PROVIDERS)}"
        )


def _check_strategy(name: str) -> None:
    if name not in STRATEGY_REGISTRY:
        raise ConfigError(
            f"strategy: unknown strategy {name!r}. "
            f"Available: {', '.join(sorted(STRATEGY_REGISTRY))}"
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
        "all_role_knowledge": config.all_role_knowledge,
        "consensus_stop": config.consensus_stop,
        "review_rounds": config.review_rounds,
        "review_round_template": config.review_round_template if config.review_rounds else None,
        "review_round_first_template": config.review_round_first_template if config.review_rounds else None,
        "prompts": {
            "task": config.task_prompt,
            "system": config.system_prompt,
            "summarize": config.summarize_prompt,
            "consensus": config.consensus_prompt if config.consensus_stop else None,
            "user_story_template": config.user_story_template,
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
    _check_strategy(config.strategy.name)

    agents = load_roster(config.agents_dir, config.roster)
    _apply_panel(agents, config.panel)

    # Private per-role context, off entirely when role_knowledge is false.
    knowledge: dict[str, str] = {}
    if config.role_knowledge:
        for name, agent in agents.items():
            if agent.knowledge:
                knowledge[name] = _read(agent.knowledge, f"{name}.knowledge").strip()

    # Solo baseline: union every role's knowledge onto the roster, discovered from
    # agents_dir itself rather than a hand-typed list — always the same source the
    # team reads from, never a separately maintained copy that could drift.
    _lens_word_overrides = {"qa": "QA"}
    def _lens_label(stem: str) -> str:
        return " ".join(_lens_word_overrides.get(w, w.title()) for w in stem.split("_"))

    all_role_knowledge_paths: list[str] = []
    if config.all_role_knowledge:
        discovered = [load_agent(p) for p in sorted(Path(config.agents_dir).glob("*.yaml"))]
        all_role_knowledge_paths = sorted({a.knowledge for a in discovered if a.knowledge})
        lenses = [
            f"## Lens: {_lens_label(Path(kpath).stem)}\n"
            f"{_read(kpath, f'all_role_knowledge:{kpath}').strip()}"
            for kpath in all_role_knowledge_paths
        ]
        union_text = "\n\n".join(lenses)
        for name in agents:
            knowledge[name] = "\n\n".join(filter(None, [knowledge.get(name), union_text]))

    user_story_template = _read(config.user_story_template, "user_story_template").strip()

    system_prompt_raw = _read(config.system_prompt, "system_prompt").strip()
    if "{user_story_template}" not in system_prompt_raw:
        raise ConfigError(f"{config.system_prompt}: must contain a {{user_story_template}} placeholder")
    system_prompt = system_prompt_raw.format(user_story_template=user_story_template)

    task_prompt = _read(config.task_prompt, "task_prompt")
    consensus_prompt = _read(config.consensus_prompt, "consensus_prompt") if config.consensus_stop else ""
    summarize_template = _read(config.summarize_prompt, "summarize_prompt")
    if "{transcript}" not in summarize_template:
        raise ConfigError(f"{config.summarize_prompt}: must contain a {{transcript}} placeholder")

    summarizer_role_raw = _read(config.summarizer.role_prompt, "summarizer.role_prompt").strip()
    if "{user_story_template}" not in summarizer_role_raw:
        raise ConfigError(
            f"{config.summarizer.role_prompt}: must contain a {{user_story_template}} placeholder"
        )
    summarizer = Agent(
        name="summarizer",
        role=summarizer_role_raw.format(user_story_template=user_story_template),
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
        config.summarizer.role_prompt, config.user_story_template,
        *(str(Path(config.agents_dir) / f"{n}.yaml") for n in config.roster),
        *(a.knowledge for a in agents.values() if config.role_knowledge and a.knowledge),
        *([config.consensus_prompt] if config.consensus_stop else []),
        *all_role_knowledge_paths,
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
        consensus_prompt=consensus_prompt,
        file_hashes=file_hashes,
        run_id=run_id,
        rng=random.Random(config.seed),
    )


def load(path: str | Path) -> ResolvedRun:
    return resolve(load_run_config(path))
