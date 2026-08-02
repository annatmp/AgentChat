"""
Turn-taking strategies, addressable by name from a run config.

A strategy is just a `TurnSelector` — `(history, agents) -> agent name`. That
stays the central design decision; nothing here changes that signature. A
strategy that wants to record *why* it picked someone writes to the
`SelectorLog` it closes over, and a strategy that makes its own LLM calls does
so via `call_agent_recorded(..., kind=KIND_SELECTOR)` and adds the record to the
same log, so its cost shows up as selector overhead rather than conversation
spend.

`build(name, params, ...)` is the seam the config format depends on: adding a
strategy means writing the factory and adding one row to REGISTRY.
"""

from __future__ import annotations

import random
from typing import Callable

from agent_chat.agents import Agent
from agent_chat.conversation import Message, TurnSelector
from agent_chat.records import SelectorLog


def round_robin(*names: str, log: SelectorLog | None = None) -> TurnSelector:
    """Cycle through agents in the given order."""
    order = list(names)

    def select(history: list[Message], agents: dict[str, Agent]) -> str:
        agent_turns = sum(1 for m in history if m.speaker != "user")
        position = agent_turns % len(order)
        if log:
            log.note(strategy="round_robin", position=position, order=order)
        return order[position]

    return select


# --- Registry ---

def _build_round_robin(
    params: dict, *, roster: dict[str, Agent], rng: random.Random, log: SelectorLog | None,
) -> TurnSelector:
    order = list(params.get("order") or roster)
    unknown = [name for name in order if name not in roster]
    if unknown:
        raise ValueError(
            f"round_robin order names agents outside the roster: {', '.join(unknown)}"
        )
    if not order:
        raise ValueError("round_robin needs at least one agent")
    return round_robin(*order, log=log)


StrategyBuilder = Callable[..., TurnSelector]

REGISTRY: dict[str, StrategyBuilder] = {
    "round_robin": _build_round_robin,
}


def build(
    name: str,
    params: dict | None = None,
    *,
    roster: dict[str, Agent],
    rng: random.Random | None = None,
    log: SelectorLog | None = None,
) -> TurnSelector:
    """Construct the named strategy. `rng` is seeded from the run config."""
    try:
        builder = REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"unknown strategy {name!r}. Available: {', '.join(sorted(REGISTRY))}"
        ) from None
    return builder(params or {}, roster=roster, rng=rng or random.Random(0), log=log)
