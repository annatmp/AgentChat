import random
from unittest.mock import patch

import pytest

from agent_chat.agents import Agent
from agent_chat.conversation import CallResult, Message
from agent_chat.records import CallRecord, SelectorLog
from agent_chat.strategies.facilitator import (
    FacilitatorDecision,
    build_facilitator,
    parse_facilitator_decision,
)

ROSTER_NAMES = {"alice", "bob", "scrum_master"}


# --- parse_facilitator_decision ---

def test_parses_a_named_agent():
    text = "NEXT: bob\nREASON: bob raised the open question"
    assert parse_facilitator_decision(text, ROSTER_NAMES, chair_name="scrum_master") == FacilitatorDecision(
        next_speaker="bob", reason="bob raised the open question", parsed=True,
    )


def test_self_resolves_to_the_chair():
    text = "NEXT: SELF\nREASON: time to redirect the discussion"
    result = parse_facilitator_decision(text, ROSTER_NAMES, chair_name="scrum_master")
    assert result.next_speaker == "scrum_master"
    assert result.parsed is True


def test_is_case_insensitive():
    text = "next: self\nreason: wrapping up"
    assert parse_facilitator_decision(text, ROSTER_NAMES, chair_name="scrum_master").next_speaker == "scrum_master"


def test_missing_next_fails_closed():
    result = parse_facilitator_decision("no structure here", ROSTER_NAMES, chair_name="scrum_master")
    assert result.next_speaker is None
    assert result.parsed is False


def test_a_name_outside_the_roster_fails_closed():
    result = parse_facilitator_decision("NEXT: dave\nREASON: ??", ROSTER_NAMES, chair_name="scrum_master")
    assert result.next_speaker is None
    assert result.parsed is False


# --- facilitator selector, mocking call_agent_recorded so nothing hits a real API ---

def _agent(name: str) -> Agent:
    return Agent(name=name, role=f"You are {name}.", model="test-model", provider="anthropic")


ROSTER = {"alice": _agent("alice"), "bob": _agent("bob"), "scrum_master": _agent("scrum_master")}
HISTORY = [
    Message(speaker="user", content="kickoff"),
    Message(speaker="alice", content="I think we're close to done here."),
]


def _fake_call(*responses: str):
    """Return each response in order across successive calls, regardless of agent."""
    it = iter(responses)
    def fake(agent, history, system, *, kind, **kwargs):
        return CallResult(
            text=next(it),
            record=CallRecord(agent=agent.name, kind=kind, provider=agent.provider, model_requested=agent.model),
        )
    return fake


def test_chair_can_pick_another_agent():
    selector = build_facilitator({}, roster=ROSTER, rng=random.Random(0), log=None, knowledge={}, system_prompt="")
    with patch("agent_chat.strategies.facilitator.call_agent_recorded",
               _fake_call("NEXT: bob\nREASON: bob should confirm the plan")):
        winner = selector(HISTORY, ROSTER)
    assert winner == "bob"


def test_chair_can_pick_itself():
    selector = build_facilitator({}, roster=ROSTER, rng=random.Random(0), log=None, knowledge={}, system_prompt="")
    with patch("agent_chat.strategies.facilitator.call_agent_recorded",
               _fake_call("NEXT: SELF\nREASON: time to wrap up")):
        winner = selector(HISTORY, ROSTER)
    assert winner == "scrum_master"


def test_retries_with_corrective_feedback_and_succeeds():
    selector = build_facilitator(
        {"facilitator_max_attempts": 3}, roster=ROSTER, rng=random.Random(0), log=None,
        knowledge={}, system_prompt="",
    )
    with patch("agent_chat.strategies.facilitator.call_agent_recorded",
               side_effect=_fake_call("garbled nonsense", "NEXT: alice\nREASON: retried successfully")) as mock_call:
        winner = selector(HISTORY, ROSTER)
    assert winner == "alice"
    assert mock_call.call_count == 2
    # The retry prompt should explain what went wrong, not just repeat the original.
    second_call_history = mock_call.call_args_list[1].args[1]
    assert "could not be used" in second_call_history[-1].content


def test_raises_after_exhausting_all_attempts():
    selector = build_facilitator(
        {"facilitator_max_attempts": 2}, roster=ROSTER, rng=random.Random(0), log=None,
        knowledge={}, system_prompt="",
    )
    with patch("agent_chat.strategies.facilitator.call_agent_recorded",
               side_effect=_fake_call("garbled", "still garbled")) as mock_call:
        with pytest.raises(RuntimeError, match="failed to produce a valid decision"):
            selector(HISTORY, ROSTER)
    assert mock_call.call_count == 2


def test_unknown_chair_is_rejected_at_build_time():
    with pytest.raises(ValueError, match="not in the roster"):
        build_facilitator({"chair": "nobody"}, roster=ROSTER, rng=random.Random(0), log=None,
                           knowledge={}, system_prompt="")


def test_starting_agent_skips_the_decision_call_on_the_first_turn():
    selector = build_facilitator(
        {"starting_agent": "alice"}, roster=ROSTER, rng=random.Random(0), log=None,
        knowledge={}, system_prompt="",
    )
    kickoff_only = [Message(speaker="user", content="kickoff")]
    with patch("agent_chat.strategies.facilitator.call_agent_recorded") as mock_call:
        winner = selector(kickoff_only, ROSTER)
    assert winner == "alice"
    mock_call.assert_not_called()


def test_records_the_decision_in_the_selector_log():
    log = SelectorLog()
    selector = build_facilitator({}, roster=ROSTER, rng=random.Random(0), log=log, knowledge={}, system_prompt="")
    with patch("agent_chat.strategies.facilitator.call_agent_recorded",
               _fake_call("NEXT: bob\nREASON: bob should confirm the plan")):
        selector(HISTORY, ROSTER)
    note = log.drain()
    assert note["facilitator_decision"] == {
        "chair": "scrum_master", "next": "bob", "reason": "bob should confirm the plan", "attempts": 1,
    }
