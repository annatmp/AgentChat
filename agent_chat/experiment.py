"""
Experiment definitions: a grid of (panel config x strategy x seed) cells.

An experiment crosses a handful of base panel configs (configs/*.yaml, one
per model family per tier) against the turn-taking strategies and replicate
seeds TODO.md's P4 grid plan asks for. `run_experiment.py` is the thin CLI
entry point that walks the grid this module expands; this module stays pure
and testable, mirroring config.py's relationship to main.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent_chat.config import ConfigError


@dataclass
class TierConfig:
    configs: list[str] = field(default_factory=list)
    # None -> use the experiment-level default strategies; [] -> no crossing
    # at all (e.g. solo, which doesn't need a turn-taking strategy).
    strategies: list[str] | None = None


@dataclass
class ExperimentConfig:
    name: str
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    strategies: list[str] = field(
        default_factory=lambda: ["round_robin", "bidding", "obligation_first", "facilitator"]
    )
    tiers: dict[str, TierConfig] = field(default_factory=dict)


@dataclass
class Cell:
    tier: str
    panel_config: str
    strategy: str | None   # None -> no --strategy override, panel runs its own default
    seed: int


def _construct_tier(data: dict, where: str) -> TierConfig:
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    unknown = set(data) - set(TierConfig.__dataclass_fields__)
    if unknown:
        known = ", ".join(sorted(TierConfig.__dataclass_fields__))
        raise ConfigError(f"{where}: unknown keys {', '.join(sorted(unknown))}. Known keys: {known}")
    if not data.get("configs"):
        raise ConfigError(f"{where}: needs a non-empty 'configs' list")
    return TierConfig(**data)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"no experiment config at {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")

    tiers_data = data.pop("tiers", None) or {}
    if not isinstance(tiers_data, dict):
        raise ConfigError(f"{path}: tiers: expected a mapping of tier name -> config")
    tiers = {
        tier_name: _construct_tier(tier_data, f"{path}: tiers.{tier_name}")
        for tier_name, tier_data in tiers_data.items()
    }
    if not tiers:
        raise ConfigError(f"{path}: needs at least one tier under 'tiers'")

    data.setdefault("name", path.stem)
    unknown = set(data) - set(ExperimentConfig.__dataclass_fields__)
    if unknown:
        known = ", ".join(sorted(ExperimentConfig.__dataclass_fields__))
        raise ConfigError(f"{path}: unknown keys {', '.join(sorted(unknown))}. Known keys: {known}")
    config = ExperimentConfig(**data)
    config.tiers = tiers
    return config


def expand_grid(experiment: ExperimentConfig) -> list[Cell]:
    """
    Loop order is panel -> seed -> strategy, strategy innermost — TODO.md's
    "strategy loop innermost so paired runs share a seed/panel block": every
    strategy for a given (panel, seed) runs back to back, sharing that seed.

    A tier with `strategies: []` (solo) yields one cell per (panel, seed)
    with `strategy=None` — no crossing, the panel runs its own default.
    """
    cells: list[Cell] = []
    for tier_name, tier in experiment.tiers.items():
        strategies = tier.strategies if tier.strategies is not None else experiment.strategies
        for panel in tier.configs:
            for seed in experiment.seeds:
                if strategies:
                    for strategy in strategies:
                        cells.append(Cell(tier_name, panel, strategy, seed))
                else:
                    cells.append(Cell(tier_name, panel, None, seed))
    return cells
