from __future__ import annotations

import re
from dataclasses import dataclass, replace

from .agents import Agent
from .conversation import (
    AgentCallError,
    Message,
    PostProcessor,
    StopCondition,
    TurnSelector,
    call_agent_recorded,
    system_for,
)
from .records import KIND_SELECTOR, KIND_SUMMARY, CallRecord, SelectorLog

# Turn selectors live in agent_chat/strategies/ so a run config can name one.
# Re-exported here because the README and existing notebooks import it from
# policies.
from .strategies import round_robin  # noqa: F401


# --- Stop conditions ---

def max_turns(n: int) -> StopCondition:
    """Stop after n agent turns (user messages don't count)."""
    def check(history: list[Message]) -> bool:
        return sum(1 for m in history if m.speaker != "user") >= n
    return check


def stop_on_keyword(*keywords: str) -> StopCondition:
    """Stop when any keyword appears in the last agent message."""
    def check(history: list[Message]) -> bool:
        agent_msgs = [m for m in history if m.speaker != "user"]
        if not agent_msgs:
            return False
        last = agent_msgs[-1].content.lower()
        return any(kw.lower() in last for kw in keywords)
    return check


def stop_when_any(*conditions: StopCondition) -> StopCondition:
    """Combine stop conditions: stop as soon as any one of them says stop."""
    def check(history: list[Message]) -> bool:
        return any(condition(history) for condition in conditions)
    return check


# --- Consensus early-stop ---

_VOTE_RE = re.compile(r"VOTE:\s*(STOP|CONTINUE)", re.IGNORECASE)
_VOTE_REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)


@dataclass
class ConsensusVote:
    stop: bool
    reason: str
    parsed: bool = True


def parse_consensus_vote(text: str) -> ConsensusVote:
    """
    Parse one agent's continue/stop vote.

    Fails closed: a missing or unparseable VOTE counts as CONTINUE, not STOP —
    ending a run early is harder to undo than running one extra turn.
    """
    vote_match = _VOTE_RE.search(text)
    reason_match = _VOTE_REASON_RE.search(text)
    reason = reason_match.group(1) if reason_match else "No reason provided."

    if not vote_match:
        return ConsensusVote(stop=False, reason="Missing VOTE in response.", parsed=False)

    return ConsensusVote(stop=vote_match.group(1).upper() == "STOP", reason=reason, parsed=True)


@dataclass
class ConsensusOutcome:
    """Where `consensus_stop` puts its last vote round, since a StopCondition returns only bool."""
    stopped: bool = False
    votes: list[dict] | None = None


def consensus_stop(
    agents: dict[str, Agent],
    *,
    vote_prompt_file: str = "prompts/consensus_prompt.txt",
    knowledge: dict[str, str] | None = None,
    system_prompt: str = "",
    log: SelectorLog | None = None,
    outcome: ConsensusOutcome | None = None,
    vote_max_tokens: int = 64,
) -> StopCondition:
    """
    Stop once every agent votes to end the meeting, checked before each turn.

    An explicit private vote per agent, the same shape as urgency_auction's
    bidding: each call is `kind=KIND_SELECTOR`, so its cost lands as selector
    overhead rather than conversation spend, and it doesn't consume a turn
    from the budget. Degrades cleanly to self-termination for a one-agent
    roster — unanimity of one is just that agent's own vote.
    """
    with open(vote_prompt_file) as f:
        vote_prompt_text = f.read()
    knowledge = knowledge or {}

    def check(history: list[Message]) -> bool:
        agent_turns_so_far = sum(1 for m in history if m.speaker != "user")
        if agent_turns_so_far == 0:
            return False  # nobody has spoken yet — nothing to vote on

        votes: dict[str, ConsensusVote] = {}
        for agent in agents.values():
            vote_agent = replace(agent, max_tokens=vote_max_tokens)
            vote_system = system_for(
                agent, private_knowledge=knowledge.get(agent.name, ""), system_prompt=system_prompt,
            )
            vote_history = history + [Message(speaker="user", content=vote_prompt_text)]
            result = call_agent_recorded(vote_agent, vote_history, vote_system, kind=KIND_SELECTOR)
            if log:
                log.add_call(result.record)
            votes[agent.name] = parse_consensus_vote(result.text)

        stopped = all(vote.stop for vote in votes.values())
        vote_payload = [
            {"agent": name, "stop": vote.stop, "reason": vote.reason, "parsed": vote.parsed}
            for name, vote in votes.items()
        ]
        if outcome is not None:
            outcome.stopped = stopped
            outcome.votes = vote_payload
        if log:
            log.note(consensus_stop=stopped, consensus_votes=vote_payload)

        return stopped

    return check


# --- Post processors ---

@dataclass
class SummaryOutcome:
    """Where `summarize` puts its result, since a PostProcessor returns None."""
    text: str | None = None
    call: CallRecord | None = None
    error: str | None = None


def summarize(
    agent: Agent,
    outcome: SummaryOutcome | None = None,
    stream: bool = False,
    prompt_file: str = "prompts/summarize_prompt.txt",
) -> PostProcessor:
    """
    Summarize the full conversation into the final backlog artifact.

    `agent` must be the neutral summarizer from the run config — a fixed model,
    identical across every condition, and never a participant. Using a
    participant here means the agent that wrote the plan also writes the
    artifact being judged.
    """
    with open(prompt_file) as f:
        prompt_template = f.read()

    def run(history: list[Message]) -> None:
        transcript = "\n\n".join(f"[{m.speaker.upper()}]\n{m.content}" for m in history)
        synthetic_history = [Message(
            speaker="user",
            content=prompt_template.format(transcript=transcript),
        )]
        print("\n--- SUMMARY ---\n", flush=True)
        on_token = (lambda t: print(t, end="", flush=True)) if stream else None
        try:
            result = call_agent_recorded(
                agent, synthetic_history, system=agent.role,
                kind=KIND_SUMMARY, on_token=on_token,
                temperature=agent.temperature,
            )
        except AgentCallError as exc:
            # The transcript is already collected; losing the summary should not
            # discard the run, so it is recorded as an error and the run goes on.
            if outcome is not None:
                outcome.error = exc.record.error
                outcome.call = exc.record
            print(f"[summary failed: {exc.record.error}]", flush=True)
            return
        if not stream:
            print(result.text)
        print()
        if outcome is not None:
            outcome.text = result.text
            outcome.call = result.record

    return run
