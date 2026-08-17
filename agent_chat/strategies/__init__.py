"""
Turn-taking strategies, addressable by name from a run config.

A strategy is just a `TurnSelector` — `(history, agents) -> agent name`. That
stays the central design decision; nothing here changes that signature. A
strategy that wants to record *why* it picked someone writes to the
`SelectorLog` it closes over, and a strategy that makes its own LLM calls does
so via `call_agent_recorded(..., kind=KIND_SELECTOR)` and adds the record to the
same log, so its cost shows up as selector overhead rather than conversation
spend.

Each strategy lives in its own module (`round_robin.py`, ...); mechanics
shared across strategies (e.g. bidding) live in their own module too, so an
auction-style fallback isn't stuck reimplementing them. `build(name, params,
...)` is the seam the config format depends on: adding a strategy means
writing its module's `build_*` factory and adding one row to REGISTRY.
"""

from __future__ import annotations

import random
from typing import Callable

from agent_chat.agents import Agent
from agent_chat.conversation import TurnSelector
from agent_chat.records import SelectorLog

from agent_chat.strategies.round_robin import build_round_robin, round_robin  # noqa: F401 (re-exported for policies.py)
from agent_chat.strategies.bidding import build_urgency_auctioning, urgency_auctioning  # noqa: F401 (re-exported for policies.py)

StrategyBuilder = Callable[..., TurnSelector]

REGISTRY: dict[str, StrategyBuilder] = {
    "round_robin": build_round_robin,
    "bidding": build_urgency_auctioning,
}


def build(
    name: str,
    params: dict | None = None,
    *,
    roster: dict[str, Agent],
    rng: random.Random | None = None,
    log: SelectorLog | None = None,
    knowledge: dict[str, str] | None = None,
    system_prompt: str = "",
    turn_budget: int = 0,
) -> TurnSelector:
    """Construct the named strategy. `rng` is seeded from the run config."""
    try:
        builder = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        ) from None
    return builder(
        params or {}, roster=roster, rng=rng or random.Random(0), log=log,
        knowledge=knowledge or {}, system_prompt=system_prompt, turn_budget=turn_budget,
    )
