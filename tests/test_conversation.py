from types import SimpleNamespace
from unittest.mock import patch

from agent_chat.agents import Agent
from agent_chat.conversation import CallResult, Conversation, Message, call_agent_recorded
from agent_chat.records import CallRecord


def _agent(name: str) -> Agent:
    return Agent(name=name, role=f"You are {name}.", model="test-model", provider="anthropic")


def _fake_call(record_content):
    def fake(agent, history, system, *, kind, **kwargs):
        record_content.append(history[-1].content)
        return CallResult(
            text="ok",
            record=CallRecord(agent=agent.name, kind=kind, provider=agent.provider, model_requested=agent.model),
        )
    return fake


def test_extra_context_is_visible_to_the_call_but_not_persisted_to_history():
    conv = Conversation(agents={"solo": _agent("solo")}, system_prompt="")
    conv.user("kickoff")
    seen: list[str] = []

    with patch("agent_chat.conversation.call_agent_recorded", _fake_call(seen)):
        conv.step("solo", extra_context="REVIEW ROUND 1")

    assert seen == ["REVIEW ROUND 1"]  # the call saw it...
    assert all(m.content != "REVIEW ROUND 1" for m in conv.history)  # ...but history never kept it


def test_turn_context_is_threaded_through_run_per_turn():
    conv = Conversation(agents={"solo": _agent("solo")}, system_prompt="")
    conv.user("kickoff")
    seen: list[str] = []

    with patch("agent_chat.conversation.call_agent_recorded", _fake_call(seen)):
        conv.run(
            turn_selector=lambda history, agents: "solo",
            stop_condition=lambda history: sum(1 for m in history if m.speaker != "user") >= 2,
            turn_context=lambda turn_count: f"REVIEW ROUND {turn_count + 1}",
        )

    assert seen == ["REVIEW ROUND 1", "REVIEW ROUND 2"]


def test_run_without_turn_context_never_injects_anything():
    conv = Conversation(agents={"solo": _agent("solo")}, system_prompt="")
    conv.user("kickoff")
    seen: list[str] = []

    with patch("agent_chat.conversation.call_agent_recorded", _fake_call(seen)):
        conv.run(
            turn_selector=lambda history, agents: "solo",
            stop_condition=lambda history: sum(1 for m in history if m.speaker != "user") >= 1,
        )

    assert seen == ["kickoff"]  # the call saw ordinary history, nothing injected


def _google_agent(name: str) -> Agent:
    return Agent(name=name, role=f"You are {name}.", model="gemini-3.6-flash", provider="google")


class _FakeStream:
    """Minimal stand-in for the chat.completions.create(stream=True) iterator."""

    def __init__(self, captured_kwargs: list[dict]):
        self._captured_kwargs = captured_kwargs

    def __call__(self, **kwargs):
        self._captured_kwargs.append(kwargs)
        yield SimpleNamespace(
            model="gemini-3.6-flash",
            usage=None,
            choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content="ok"))],
        )
        yield SimpleNamespace(
            model="gemini-3.6-flash",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, prompt_tokens_details=None),
            choices=[],
        )


def test_google_calls_never_send_seed():
    # Gemini's OpenAI-compatible endpoint 400s on "seed" every time; sending it
    # would mean paying for a guaranteed-failing first attempt on every turn.
    captured: list[dict] = []
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeStream(captured))))

    with patch.dict("agent_chat.conversation._OPENAI_COMPATIBLE_CLIENTS", {"google": lambda: fake_client}):
        call_agent_recorded(_google_agent("solo"), [Message(speaker="user", content="hi")], "system", seed=7)

    assert "seed" not in captured[0]


def test_google_calls_append_continuation_when_history_ends_on_own_turn():
    # Same agent speaking twice in a row (no one else's turn in between)
    # leaves the payload ending on a model turn, which Gemini rejects outright.
    captured: list[dict] = []
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeStream(captured))))
    history = [Message(speaker="solo", content="earlier turn")]

    with patch.dict("agent_chat.conversation._OPENAI_COMPATIBLE_CLIENTS", {"google": lambda: fake_client}):
        call_agent_recorded(_google_agent("solo"), history, "system")

    assert captured[0]["messages"][-1] == {"role": "user", "content": "Continue."}


def test_google_calls_add_nothing_when_history_already_ends_on_a_user_turn():
    captured: list[dict] = []
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeStream(captured))))
    history = [Message(speaker="other", content="over to you")]

    with patch.dict("agent_chat.conversation._OPENAI_COMPATIBLE_CLIENTS", {"google": lambda: fake_client}):
        call_agent_recorded(_google_agent("solo"), history, "system")

    assert captured[0]["messages"][-1] == {"role": "user", "content": "[other]: over to you"}
