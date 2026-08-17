import random
from unittest.mock import patch

from agent_chat.agents import Agent
from agent_chat.conversation import CallResult, Message
from agent_chat.records import CallRecord, SelectorLog
from agent_chat.strategies.bidding import Bid, build_urgency_auctioning, parse_bid


def test_parses_a_well_formed_bid():
    text = "LEVEL: 3\nREASON: the API contract is still undecided"
    assert parse_bid(text) == Bid(level=3, reason="the API contract is still undecided", parsed=True)


def test_accepts_the_boundary_levels():
    assert parse_bid("LEVEL: 0\nREASON: nothing to add").level == 0
    assert parse_bid("LEVEL: 4\nREASON: I was addressed directly").level == 4


def test_is_case_insensitive():
    text = "level: 2\nreason: some general thoughts"
    assert parse_bid(text) == Bid(level=2, reason="some general thoughts", parsed=True)


def test_tolerates_preamble_before_the_level_line():
    # The prompt asks for exactly two lines, but the model doesn't always comply.
    text = "Sure, here is my bid.\nLEVEL: 1\nREASON: nothing urgent"
    assert parse_bid(text) == Bid(level=1, reason="nothing urgent", parsed=True)


def test_missing_level_defaults_to_zero_and_is_flagged_unparsed():
    bid = parse_bid("I don't think I have anything to say right now.")
    assert bid.level == 0
    assert bid.parsed is False


def test_out_of_range_level_is_clamped_and_flagged_unparsed():
    # A malformed level must stay visible as unparsed, not look like a real level-4 bid.
    bid = parse_bid("LEVEL: 7\nREASON: very urgent")
    assert bid.level == 4
    assert bid.parsed is False


def test_missing_reason_defaults_to_a_placeholder():
    bid = parse_bid("LEVEL: 3")
    assert bid.level == 3
    assert bid.reason == "No reason provided."
    assert bid.parsed is True


def test_reason_capture_stops_at_end_of_line():
    text = "LEVEL: 2\nREASON: short reason\nsomething unrelated on the next line"
    assert parse_bid(text).reason == "short reason"


# --- urgency_auctioning (select), mocking call_agent_recorded so nothing hits a real API ---

def _agent(name: str) -> Agent:
    return Agent(name=name, role=f"You are {name}.", model="test-model", provider="anthropic")


ROSTER = {"alice": _agent("alice"), "bob": _agent("bob")}
KICKOFF = [Message(speaker="user", content="kickoff")]


def _fake_call(bid_texts: dict[str, str], seen_messages: dict[str, str] | None = None):
    """Stand-in for call_agent_recorded: returns each agent's canned bid text."""
    def fake(agent, history, system, *, kind, **kwargs):
        if seen_messages is not None:
            seen_messages[agent.name] = history[-1].content
        return CallResult(
            text=bid_texts[agent.name],
            record=CallRecord(agent=agent.name, kind=kind, provider=agent.provider, model_requested=agent.model),
        )
    return fake


def _select(bid_texts, *, params=None, rng=None, log=None, turn_budget=0):
    selector = build_urgency_auctioning(
        params or {}, roster=ROSTER, rng=rng or random.Random(0), log=log,
        knowledge={}, system_prompt="", turn_budget=turn_budget,
    )
    with patch("agent_chat.strategies.bidding.call_agent_recorded", _fake_call(bid_texts)):
        return selector(KICKOFF, ROSTER)


def test_highest_bid_wins():
    winner = _select({
        "alice": "LEVEL: 1\nREASON: nothing much",
        "bob": "LEVEL: 3\nREASON: I have concerns",
    })
    assert winner == "bob"


def test_ties_are_broken_deterministically_by_the_seeded_rng():
    bid_texts = {
        "alice": "LEVEL: 2\nREASON: a thought",
        "bob": "LEVEL: 2\nREASON: another thought",
    }
    winner_1 = _select(bid_texts, rng=random.Random(42))
    winner_2 = _select(bid_texts, rng=random.Random(42))
    assert winner_1 == winner_2  # same seed, same outcome
    assert winner_1 == random.Random(42).choice(["alice", "bob"])


def test_starting_agent_skips_the_auction_on_the_first_turn():
    with patch("agent_chat.strategies.bidding.call_agent_recorded") as mock_call:
        selector = build_urgency_auctioning(
            {"starting_agent": "bob"}, roster=ROSTER, rng=random.Random(0), log=None,
            knowledge={}, system_prompt="", turn_budget=5,
        )
        winner = selector(KICKOFF, ROSTER)
    assert winner == "bob"
    mock_call.assert_not_called()


def test_records_bid_rationale_and_winner_in_the_selector_log():
    log = SelectorLog()
    _select({
        "alice": "LEVEL: 0\nREASON: quiet",
        "bob": "LEVEL: 4\nREASON: addressed directly",
    }, log=log, turn_budget=10)
    note = log.drain()
    assert note["winner"] == "bob"
    bids_by_agent = {b["agent"]: b for b in note["bids"]}
    assert bids_by_agent["bob"] == {"agent": "bob", "level": 4, "reason": "addressed directly", "parsed": True}
    assert note["turns_remaining"] == 10  # no agent has spoken yet in KICKOFF


def test_bid_prompt_is_told_how_many_turns_remain(tmp_path):
    prompt_file = tmp_path / "bid_prompt.txt"
    prompt_file.write_text("Turns left: {turns_remaining}\nLEVEL: <n>\nREASON: <text>")
    seen_messages: dict[str, str] = {}

    selector = build_urgency_auctioning(
        {"bid_prompt": str(prompt_file)}, roster=ROSTER, rng=random.Random(0), log=None,
        knowledge={}, system_prompt="", turn_budget=7,
    )
    bid_texts = {"alice": "LEVEL: 1\nREASON: ok", "bob": "LEVEL: 1\nREASON: ok"}
    with patch("agent_chat.strategies.bidding.call_agent_recorded", _fake_call(bid_texts, seen_messages)):
        selector(KICKOFF, ROSTER)

    assert "Turns left: 7" in seen_messages["alice"]
