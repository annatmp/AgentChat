from unittest.mock import patch

from agent_chat.agents import Agent
from agent_chat.conversation import CallResult, Message
from agent_chat.policies import (
    ConsensusOutcome,
    ConsensusVote,
    consensus_stop,
    parse_consensus_vote,
    review_round_context,
    stop_when_any,
)
from agent_chat.records import CallRecord, SelectorLog


# --- parse_consensus_vote ---

def test_parses_a_stop_vote():
    text = "VOTE: STOP\nREASON: the plan is complete"
    assert parse_consensus_vote(text) == ConsensusVote(stop=True, reason="the plan is complete", parsed=True)


def test_parses_a_continue_vote():
    text = "VOTE: CONTINUE\nREASON: QA hasn't weighed in yet"
    assert parse_consensus_vote(text) == ConsensusVote(stop=False, reason="QA hasn't weighed in yet", parsed=True)


def test_is_case_insensitive():
    text = "vote: stop\nreason: agreed"
    assert parse_consensus_vote(text) == ConsensusVote(stop=True, reason="agreed", parsed=True)


def test_missing_vote_fails_closed_to_continue():
    # Ending a run early is harder to undo than running one extra turn.
    vote = parse_consensus_vote("I don't really have an opinion here.")
    assert vote.stop is False
    assert vote.parsed is False


def test_missing_reason_defaults_to_a_placeholder():
    vote = parse_consensus_vote("VOTE: STOP")
    assert vote.stop is True
    assert vote.reason == "No reason provided."


# --- consensus_stop, mocking call_agent_recorded so nothing hits a real API ---

def _agent(name: str) -> Agent:
    return Agent(name=name, role=f"You are {name}.", model="test-model", provider="anthropic")


ROSTER = {"alice": _agent("alice"), "bob": _agent("bob")}
HISTORY = [Message(speaker="user", content="kickoff"), Message(speaker="alice", content="hi")]


def _fake_call(vote_texts: dict[str, str]):
    def fake(agent, history, system, *, kind, **kwargs):
        return CallResult(
            text=vote_texts[agent.name],
            record=CallRecord(agent=agent.name, kind=kind, provider=agent.provider, model_requested=agent.model),
        )
    return fake


def test_stops_only_when_every_agent_votes_stop():
    check = consensus_stop(ROSTER)
    with patch("agent_chat.policies.call_agent_recorded", _fake_call({
        "alice": "VOTE: STOP\nREASON: done",
        "bob": "VOTE: STOP\nREASON: agreed",
    })):
        assert check(HISTORY) is True


def test_a_single_continue_vote_keeps_the_meeting_going():
    check = consensus_stop(ROSTER)
    with patch("agent_chat.policies.call_agent_recorded", _fake_call({
        "alice": "VOTE: STOP\nREASON: done",
        "bob": "VOTE: CONTINUE\nREASON: one more thing",
    })):
        assert check(HISTORY) is False


def test_an_unparseable_vote_fails_closed_and_keeps_the_meeting_going():
    check = consensus_stop(ROSTER)
    with patch("agent_chat.policies.call_agent_recorded", _fake_call({
        "alice": "VOTE: STOP\nREASON: done",
        "bob": "not sure what to say",
    })):
        assert check(HISTORY) is False


def test_skips_voting_before_any_agent_has_spoken():
    check = consensus_stop(ROSTER)
    kickoff_only = [Message(speaker="user", content="kickoff")]
    with patch("agent_chat.policies.call_agent_recorded") as mock_call:
        assert check(kickoff_only) is False
    mock_call.assert_not_called()


def test_records_the_vote_outcome():
    outcome = ConsensusOutcome()
    log = SelectorLog()
    check = consensus_stop(ROSTER, outcome=outcome, log=log)
    with patch("agent_chat.policies.call_agent_recorded", _fake_call({
        "alice": "VOTE: STOP\nREASON: done",
        "bob": "VOTE: STOP\nREASON: agreed",
    })):
        check(HISTORY)

    assert outcome.stopped is True
    votes_by_agent = {v["agent"]: v for v in outcome.votes}
    assert votes_by_agent["bob"] == {"agent": "bob", "stop": True, "reason": "agreed", "parsed": True}

    note = log.drain()
    assert note["consensus_stop"] is True
    assert note["consensus_votes"] == outcome.votes


# --- stop_when_any ---

def test_stops_when_any_condition_is_true():
    always_stop = lambda history: True
    never_stop = lambda history: False
    assert stop_when_any(never_stop, always_stop)([]) is True
    assert stop_when_any(never_stop, never_stop)([]) is False


# --- review_round_context ---

def test_round_one_uses_the_first_template_not_the_continuation_one():
    context = review_round_context(
        template="REVIEW ROUND {n}: continue refining the plan from where you left off.",
        first_template="REVIEW ROUND 1: come up with an initial plan — there is nothing to refine yet.",
    )
    assert context(0) == "REVIEW ROUND 1: come up with an initial plan — there is nothing to refine yet."


def test_later_rounds_use_the_continuation_template():
    context = review_round_context(
        template="REVIEW ROUND {n}: continue refining the plan from where you left off.",
        first_template="REVIEW ROUND 1: come up with an initial plan — there is nothing to refine yet.",
    )
    assert context(1) == "REVIEW ROUND 2: continue refining the plan from where you left off."
    assert context(2) == "REVIEW ROUND 3: continue refining the plan from where you left off."
