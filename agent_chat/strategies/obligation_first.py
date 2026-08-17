"""
Obligation-first: give the floor to whoever was just directly addressed,
falling back to the bidding auction when nobody was.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, replace

from agent_chat.agents import Agent
from agent_chat.conversation import Message, TurnSelector, call_agent_recorded, system_for
from agent_chat.records import KIND_SELECTOR, SelectorLog
from agent_chat.strategies.bidding import build_urgency_auctioning

_ADDRESSED_LINE_RE = re.compile(r"ADDRESSED:\s*(.+)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)


@dataclass
class Addressed:
    agent: str | None
    reason: str
    parsed: bool = True


def parse_addressed(text: str, roster_names: set[str], speaker: str) -> Addressed:
    """
    Parse who (if anyone) `speaker`'s last turn directly addressed.

    Fails closed to `agent=None` — which sends the turn to the bidding
    fallback — whenever the response is missing, names nobody in the roster,
    names only the speaker themselves, or names more than one person:
    obligation-first only ever hands the floor to exactly one agent.
    """
    line_match = _ADDRESSED_LINE_RE.search(text)
    reason_match = _REASON_RE.search(text)
    reason = reason_match.group(1) if reason_match else "No reason provided."

    if not line_match:
        return Addressed(agent=None, reason="Missing ADDRESSED in response.", parsed=False)

    line = line_match.group(1).strip()
    if line.upper().startswith("NONE"):
        return Addressed(agent=None, reason=reason, parsed=True)

    mentioned = {word.strip(",.") for word in line.split()} & roster_names
    mentioned.discard(speaker)

    if not mentioned:
        return Addressed(agent=None, reason=f"No roster member recognized in {line!r}.", parsed=False)
    if len(mentioned) > 1:
        return Addressed(
            agent=None,
            reason=f"Named more than one agent: {', '.join(sorted(mentioned))}.",
            parsed=False,
        )
    return Addressed(agent=next(iter(mentioned)), reason=reason, parsed=True)


def obligation_first(
    roster: dict[str, Agent], *, log: SelectorLog | None = None, knowledge: dict[str, str],
    system_prompt: str = "", params: dict = {}, rng: random.Random, turn_budget: int = 0,
) -> TurnSelector:
    obligation_prompt_path = params.get("obligation_prompt", "prompts/obligation_prompt.txt")
    with open(obligation_prompt_path) as f:
        obligation_prompt_text = f.read()
    obligation_max_tokens = int(params.get("obligation_max_tokens", 32))
    roster_names = set(roster)

    # Auction-style fallback, reused rather than reimplemented — same params
    # dict, so bid_prompt/bid_max_tokens/starting_agent all apply to it too.
    fallback = build_urgency_auctioning(
        params, roster=roster, rng=rng, log=log,
        knowledge=knowledge, system_prompt=system_prompt, turn_budget=turn_budget,
    )

    def select(history: list[Message], agents: dict[str, Agent]) -> str:
        last_turn = next((m for m in reversed(history) if m.speaker != "user"), None)
        if last_turn is None:
            return fallback(history, agents)  # turn 1 — nobody has spoken yet

        speaker = roster[last_turn.speaker]
        check_agent = replace(speaker, max_tokens=obligation_max_tokens)
        check_system = system_for(
            speaker, private_knowledge=knowledge.get(speaker.name, ""), system_prompt=system_prompt,
        )
        check_history = history + [Message(speaker="user", content=obligation_prompt_text)]

        result = call_agent_recorded(check_agent, check_history, check_system, kind=KIND_SELECTOR)
        if log:
            log.add_call(result.record)
        addressed = parse_addressed(result.text, roster_names, speaker.name)

        if log:
            log.note(obligation_check={
                "speaker": speaker.name, "addressed": addressed.agent,
                "reason": addressed.reason, "parsed": addressed.parsed,
            })

        if addressed.agent is not None:
            return addressed.agent

        return fallback(history, agents)

    return select


def build_obligation_first(
    params: dict, *, roster: dict[str, Agent], rng: random.Random, log: SelectorLog | None,
    knowledge: dict[str, str] | None = None, system_prompt: str = "", turn_budget: int = 0,
) -> TurnSelector:
    return obligation_first(
        roster, log=log, knowledge=knowledge or {}, system_prompt=system_prompt,
        params=params, rng=rng, turn_budget=turn_budget,
    )
