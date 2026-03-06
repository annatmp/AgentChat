from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Agent:
    name: str
    role: str
    model: str          # Azure deployment name
    provider: str       # reserved for future multi-provider support
    max_tokens: int = 4096
    tools: list = field(default_factory=list)   # hook for later


def load_agent(path: str | Path) -> Agent:
    data = yaml.safe_load(Path(path).read_text())
    known = Agent.__dataclass_fields__
    return Agent(**{k: v for k, v in data.items() if k in known})


def load_agents(directory: str | Path) -> dict[str, Agent]:
    return {a.name: a for p in Path(directory).glob("*.yaml") if (a := load_agent(p))}
