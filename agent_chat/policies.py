from __future__ import annotations

from dataclasses import dataclass

from .agents import Agent
from .conversation import (
    AgentCallError,
    Message,
    PostProcessor,
    StopCondition,
    TurnSelector,
    call_agent_recorded,
)
from .records import KIND_SUMMARY, CallRecord

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
