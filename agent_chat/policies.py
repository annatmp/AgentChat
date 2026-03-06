from __future__ import annotations

from .agents import Agent
from .conversation import Message, StopCondition, TurnSelector, PostProcessor, call_agent


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

def summarize(agent: Agent, stream: bool = False, prompt_file: str = "prompts/summarize_prompt.txt") -> PostProcessor:
    """Summarize the full conversation using the given agent."""
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
        result = call_agent(agent, synthetic_history, system=agent.role, on_token=on_token)
        if not stream:
            print(result)
        print()
    return run


# --- Turn selectors ---

def round_robin(*names: str) -> TurnSelector:
    """Cycle through agents in the given order."""
    order = list(names)
    def select(history: list[Message], agents: dict[str, Agent]) -> str:
        agent_turns = sum(1 for m in history if m.speaker != "user")
        return order[agent_turns % len(order)]
    return select
