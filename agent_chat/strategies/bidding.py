from dataclasses import dataclass, replace
import random
import re

from agent_chat.agents import Agent
from agent_chat.conversation import Message, TurnSelector, system_for, call_agent_recorded
from agent_chat.records import KIND_SELECTOR, SelectorLog

_LEVEL_RE = re.compile(r"LEVEL:\s*(\d)", re.IGNORECASE)
_REASON_RE = re.compile(r"REASON:\s*(.+)", re.IGNORECASE)

@dataclass
class Bid:
    level: int
    reason: str
    parsed: bool = True

def load_bid_prompt(params) -> str:

    with open(params.get("bid_prompt", "prompts/bid_prompt.txt"), 'r') as f:
        return f.read()



def parse_bid(bid_str: str) -> Bid:
    """
    Parse a bid string into a Bid object.

    Args:
        bid_str (str): The bid string in the format "level:reason".

    Returns:
        Bid: A Bid object with the parsed level and reason.
    """


    level_str = re.search(_LEVEL_RE, bid_str)
    reason_str = re.search(_REASON_RE, bid_str)

    if not level_str:
        print("Missing LEVEL in bid string.")
        return Bid(level=0, reason="Missing LEVEL in bid string.", parsed=False)

    try:
        level_as_int = int(level_str.group(1))
    except ValueError as e:
        print(f"Invalid LEVEL value: {level_str.group(1)}. Error: {e}")
        return Bid(level=0, reason="Invalid LEVEL value.", parsed=False)

    if level_as_int > 4:
        print(f"LEVEL out of range: {level_as_int}. Must be between 0 and 4.")
        return Bid(level=4, reason="LEVEL out of range. " + (reason_str.group(1) if reason_str else "No reason provided."), parsed=False)

    return Bid(level=level_as_int, reason=reason_str.group(1) if reason_str else "No reason provided.")


def urgency_auctioning(
    *names: str, log: SelectorLog | None = None, knowledge: dict[str, str],
    system_prompt: str = "", params: dict = {}, rng: random.Random, turn_budget: int = 0,
) -> TurnSelector:

    def select(history: list[Message], agents: dict[str, Agent]):
        agent_turns_so_far = sum(1 for m in history if m.speaker != "user")

        if agent_turns_so_far == 0 and 'starting_agent' in params:
            return params['starting_agent']

        turns_remaining = turn_budget - agent_turns_so_far
        bid_prompt = load_bid_prompt(params).format(turns_remaining=turns_remaining)
        bid_max_tokens = int(params.get("bid_max_tokens", 64))
        bids: dict[str, Bid] = {}

        for agent in agents.values():
            agent_knowledge = knowledge.get(agent.name, "")
            agent_system_prompt = system_for(agent, private_knowledge=agent_knowledge, system_prompt=system_prompt)
            bid_agent = replace(agent, max_tokens=bid_max_tokens)
            bid_history = history + [Message(speaker="user", content=bid_prompt)]

            result = call_agent_recorded(bid_agent, bid_history, agent_system_prompt, kind=KIND_SELECTOR)
            if log:
                log.add_call(result.record)
            bids[agent.name] = parse_bid(result.text)

        print('---')
        print("Bids received:")
        for name, bid in bids.items():
            print(f"Agent: {name}, Level: {bid.level}, Reason: {bid.reason}, Parsed: {bid.parsed}")
        print('---')

        sorted_bids = sorted(bids.items(), key=lambda x: x[1].level, reverse=True)
        max_level = sorted_bids[0][1].level
        highest_bids = [name for name, bid in sorted_bids if bid.level == max_level]
        winner = rng.choice(highest_bids)

        if log:
            log.note(
                strategy="urgency_auctioning",
                bids=[
                    {"agent": n, "level": b.level, "reason": b.reason, "parsed": b.parsed}
                    for n, b in bids.items()
                ],
                turns_remaining=turns_remaining,
                winner=winner,
            )

        return winner

    return select


def build_urgency_auctioning(
    params: dict, *, roster: dict[str, Agent], rng: random.Random, log: SelectorLog | None,
    knowledge: dict[str, str] | None = None, system_prompt: str = "", turn_budget: int = 0,
) -> TurnSelector:

    return urgency_auctioning(
        *roster, log=log, knowledge=knowledge or {}, system_prompt=system_prompt,
        params=params, rng=rng, turn_budget=turn_budget,
    )
