from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Providers `call_agent` knows how to reach. Validated at config load so a typo
# fails before any API call rather than partway through a run.
PROVIDERS = ("anthropic", "azure_openai", "azure_ai")

# Models that reject `temperature` (and `top_p`/`top_k`) with a 400. Pairing one
# with a temperature is a config error, caught at load time instead of surfacing
# as a mid-grid failure. `claude-sonnet-4-6` and `claude-haiku-4-5` accept it.
TEMPERATURE_REJECTED = frozenset({
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-mythos-5",
})


@dataclass
class Agent:
    name: str
    role: str
    model: str          # Anthropic model ID, or Azure deployment name
    provider: str       # one of PROVIDERS
    max_tokens: int = 4096
    temperature: float | None = None   # None = omit from the request
    knowledge: str | None = None       # path to this role's private context
    tools: list = field(default_factory=list)   # hook for later


def load_agent(path: str | Path) -> Agent:
    data = yaml.safe_load(Path(path).read_text())
    known = Agent.__dataclass_fields__
    return Agent(**{k: v for k, v in data.items() if k in known})


def load_agents(directory: str | Path) -> dict[str, Agent]:
    """
    Load every YAML in `directory`.

    Convenient for the interactive demo, but note that adding a role file
    changes what this returns — experiments must use `load_roster` so the
    roster is explicit and recorded.
    """
    return {a.name: a for p in Path(directory).glob("*.yaml") if (a := load_agent(p))}


def load_roster(directory: str | Path, names: list[str]) -> dict[str, Agent]:
    """
    Load exactly the named agents, in the order given.

    Raises if a name has no YAML — a run must never silently proceed with a
    roster smaller than its config asked for.
    """
    directory = Path(directory)
    roster: dict[str, Agent] = {}
    missing: list[str] = []
    for name in names:
        path = directory / f"{name}.yaml"
        if not path.exists():
            missing.append(name)
            continue
        agent = load_agent(path)
        if agent.name != name:
            raise ValueError(
                f"{path}: 'name: {agent.name}' does not match its filename ({name}.yaml)"
            )
        roster[name] = agent
    if missing:
        available = sorted(p.stem for p in directory.glob("*.yaml"))
        raise FileNotFoundError(
            f"roster names with no YAML in {directory}/: {', '.join(missing)}. "
            f"Available: {', '.join(available)}"
        )
    return roster
