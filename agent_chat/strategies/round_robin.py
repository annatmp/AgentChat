"""Fixed-order turn taking: cycle through the roster in a set order."""

from __future__ import annotations

import random

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


def build_round_robin(
    params: dict, *, roster: dict[str, Agent], rng: random.Random, log: SelectorLog | None,
    knowledge: dict[str, str] | None = None, system_prompt: str = "",
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
