"""
Facilitator: a designated chair decides who speaks next each turn, including
themselves — a single centralized judgment, unlike bidding's distributed
scoring or obligation_first's reactive self-report.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass, replace

from agent_chat.agents import Agent
from agent_chat.conversation import Message, TurnSelector, call_agent_recorded, system_for
from agent_chat.records import KIND_SELECTOR, SelectorLog

_NEXT_RE = re.compile(r"NEXT:\s*(\S+)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)


@dataclass
class FacilitatorDecision:
    next_speaker: str | None
    reason: str
    parsed: bool = True


def parse_facilitator_decision(text: str, roster_names: set[str], chair_name: str) -> FacilitatorDecision:
    """
    Parse the chair's pick for who speaks next.

    `SELF` resolves to `chair_name`. Anything that isn't `SELF` or an exact
    roster name fails closed to `next_speaker=None` — the caller retries
    rather than guessing which agent was meant.
    """
    next_match = _NEXT_RE.search(text)
    reason_match = _REASON_RE.search(text)
    reason = reason_match.group(1) if reason_match else "No reason provided."

    if not next_match:
        return FacilitatorDecision(next_speaker=None, reason="Missing NEXT in response.", parsed=False)

    token = next_match.group(1).strip(",.")
    if token.upper() == "SELF":
        return FacilitatorDecision(next_speaker=chair_name, reason=reason, parsed=True)
    if token in roster_names:
        return FacilitatorDecision(next_speaker=token, reason=reason, parsed=True)
    return FacilitatorDecision(
        next_speaker=None, reason=f"{token!r} is not SELF or a roster name.", parsed=False,
    )


def _participation_summary(history: list[Message], roster_names: set[str]) -> str:
    counts = Counter(m.speaker for m in history if m.speaker in roster_names)
    return ", ".join(f"{name}: {counts.get(name, 0)}" for name in sorted(roster_names))


def facilitator(
    roster: dict[str, Agent], *, log: SelectorLog | None = None, knowledge: dict[str, str],
    system_prompt: str = "", params: dict = {}, rng: random.Random, turn_budget: int = 0,
) -> TurnSelector:
    chair_name = params.get("chair", "scrum_master")
    if chair_name not in roster:
        raise ValueError(
            f"facilitator: chair {chair_name!r} is not in the roster ({', '.join(sorted(roster))})"
        )
    facilitator_prompt_path = params.get("facilitator_prompt", "prompts/facilitator_prompt.txt")
    with open(facilitator_prompt_path) as f:
        base_prompt_template = f.read()
    decision_max_tokens = int(params.get("facilitator_max_tokens", 64))
    max_attempts = int(params.get("facilitator_max_attempts", 3))
    roster_names = set(roster)
    chair = roster[chair_name]

    def select(history: list[Message], agents: dict[str, Agent]) -> str:
        agent_turns_so_far = sum(1 for m in history if m.speaker != "user")
        if agent_turns_so_far == 0 and "starting_agent" in params:
            return params["starting_agent"]

        chair_agent = replace(chair, max_tokens=decision_max_tokens)
        chair_system = system_for(
            chair, private_knowledge=knowledge.get(chair_name, ""), system_prompt=system_prompt,
        )
        base_prompt = base_prompt_template.format(
            participation=_participation_summary(history, roster_names),
        )

        prompt_text = base_prompt
        decision: FacilitatorDecision | None = None
        attempt = 0
        for attempt in range(max_attempts):
            decision_history = history + [Message(speaker="user", content=prompt_text)]
            result = call_agent_recorded(chair_agent, decision_history, chair_system, kind=KIND_SELECTOR)
            if log:
                log.add_call(result.record)
            decision = parse_facilitator_decision(result.text, roster_names, chair_name)
            if decision.parsed:
                break
            prompt_text = (
                f"{base_prompt}\n\nYour previous response could not be used ({decision.reason}) "
                f"Respond in exactly the required format, naming exactly one of: "
                f"{', '.join(sorted(roster_names))}, or SELF."
            )

        if decision is None or not decision.parsed:
            raise RuntimeError(
                f"facilitator: chair {chair_name!r} failed to produce a valid decision "
                f"after {max_attempts} attempts"
            )

        if log:
            log.note(facilitator_decision={
                "chair": chair_name, "next": decision.next_speaker,
                "reason": decision.reason, "attempts": attempt + 1,
            })

        return decision.next_speaker

    return select


def build_facilitator(
    params: dict, *, roster: dict[str, Agent], rng: random.Random, log: SelectorLog | None,
    knowledge: dict[str, str] | None = None, system_prompt: str = "", turn_budget: int = 0,
) -> TurnSelector:
    return facilitator(
        roster, log=log, knowledge=knowledge or {}, system_prompt=system_prompt,
        params=params, rng=rng, turn_budget=turn_budget,
    )
