import random
from unittest.mock import patch

from agent_chat.agents import Agent
from agent_chat.conversation import CallResult, Message
from agent_chat.records import CallRecord, SelectorLog
from agent_chat.strategies.obligation_first import Addressed, build_obligation_first, parse_addressed

ROSTER_NAMES = {"alice", "bob", "carol"}


# --- parse_addressed ---

def test_parses_a_single_addressed_agent():
    text = "ADDRESSED: bob\nREASON: I asked bob directly about the API"
    assert parse_addressed(text, ROSTER_NAMES, speaker="alice") == Addressed(
        agent="bob", reason="I asked bob directly about the API", parsed=True,
    )


def test_none_is_a_clean_no_obligation_result():
    result = parse_addressed("ADDRESSED: NONE\nREASON: general comment", ROSTER_NAMES, speaker="alice")
    assert result.agent is None
    assert result.parsed is True


def test_missing_addressed_line_fails_closed():
    result = parse_addressed("just some text with no structure", ROSTER_NAMES, speaker="alice")
    assert result.agent is None
    assert result.parsed is False


def test_naming_more_than_one_agent_fails_closed():
    # The exact case we're guarding against: the auction decides instead of guessing.
    result = parse_addressed("ADDRESSED: bob and carol\nREASON: both should weigh in", ROSTER_NAMES, speaker="alice")
    assert result.agent is None
    assert result.parsed is False


def test_hallucinated_name_fails_closed():
    result = parse_addressed("ADDRESSED: dave\nREASON: ??", ROSTER_NAMES, speaker="alice")
    assert result.agent is None
    assert result.parsed is False


def test_speaker_naming_themselves_is_discarded():
    result = parse_addressed("ADDRESSED: alice\nREASON: thinking out loud", ROSTER_NAMES, speaker="alice")
    assert result.agent is None


# --- obligation_first selector, mocking call_agent_recorded in every module it's called from ---

def _agent(name: str) -> Agent:
    return Agent(name=name, role=f"You are {name}.", model="test-model", provider="anthropic")


ROSTER = {"alice": _agent("alice"), "bob": _agent("bob")}
HISTORY = [
    Message(speaker="user", content="kickoff"),
    Message(speaker="alice", content="Bob, what do you think about the schema?"),
]


def _fake_call(responses: dict[str, str]):
    def fake(agent, history, system, *, kind, **kwargs):
        return CallResult(
            text=responses[agent.name],
            record=CallRecord(agent=agent.name, kind=kind, provider=agent.provider, model_requested=agent.model),
        )
    return fake


def test_gives_the_floor_to_the_addressed_agent_without_running_the_auction():
    selector = build_obligation_first({}, roster=ROSTER, rng=random.Random(0), log=None, knowledge={}, system_prompt="")
    with patch("agent_chat.strategies.obligation_first.call_agent_recorded",
               _fake_call({"alice": "ADDRESSED: bob\nREASON: directly asked"})), \
         patch("agent_chat.strategies.bidding.call_agent_recorded") as mock_bidding_call:
        winner = selector(HISTORY, ROSTER)
    assert winner == "bob"
    mock_bidding_call.assert_not_called()


def test_falls_back_to_bidding_when_nobody_was_addressed():
    selector = build_obligation_first({}, roster=ROSTER, rng=random.Random(0), log=None, knowledge={}, system_prompt="")
    with patch("agent_chat.strategies.obligation_first.call_agent_recorded",
               _fake_call({"alice": "ADDRESSED: NONE\nREASON: general remark"})), \
         patch("agent_chat.strategies.bidding.call_agent_recorded", _fake_call({
             "alice": "LEVEL: 1\nREASON: ok",
             "bob": "LEVEL: 3\nREASON: I have concerns",
         })):
        winner = selector(HISTORY, ROSTER)
    assert winner == "bob"


def test_falls_back_when_more_than_one_agent_was_named():
    selector = build_obligation_first({}, roster=ROSTER, rng=random.Random(0), log=None, knowledge={}, system_prompt="")
    with patch("agent_chat.strategies.obligation_first.call_agent_recorded",
               _fake_call({"alice": "ADDRESSED: bob and alice\nREASON: unclear"})), \
         patch("agent_chat.strategies.bidding.call_agent_recorded", _fake_call({
             "alice": "LEVEL: 0\nREASON: ok",
             "bob": "LEVEL: 2\nREASON: sure",
         })):
        winner = selector(HISTORY, ROSTER)
    assert winner == "bob"


def test_first_turn_skips_straight_to_the_auction_without_a_self_report_call():
    selector = build_obligation_first(
        {"starting_agent": "alice"}, roster=ROSTER, rng=random.Random(0), log=None, knowledge={}, system_prompt="",
    )
    kickoff_only = [Message(speaker="user", content="kickoff")]
    with patch("agent_chat.strategies.obligation_first.call_agent_recorded") as mock_check:
        winner = selector(kickoff_only, ROSTER)
    assert winner == "alice"
    mock_check.assert_not_called()


def test_records_the_obligation_check_in_the_selector_log():
    log = SelectorLog()
    selector = build_obligation_first({}, roster=ROSTER, rng=random.Random(0), log=log, knowledge={}, system_prompt="")
    with patch("agent_chat.strategies.obligation_first.call_agent_recorded",
               _fake_call({"alice": "ADDRESSED: bob\nREASON: directly asked"})):
        selector(HISTORY, ROSTER)
    note = log.drain()
    assert note["obligation_check"] == {
        "speaker": "alice", "addressed": "bob", "reason": "directly asked", "parsed": True,
    }
