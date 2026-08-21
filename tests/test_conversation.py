from types import SimpleNamespace
from unittest.mock import patch

import httpx
import openai

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


def _mistral_agent(name: str) -> Agent:
    return Agent(name=name, role=f"You are {name}.", model="mistral-medium-2505", provider="mistral")


def test_mistral_calls_never_send_seed():
    # Mistral's endpoint 422s on "seed" every time, same as Gemini's 400 —
    # skipped proactively rather than paying for a guaranteed-failing attempt.
    captured: list[dict] = []
    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=_FakeStream(captured))))

    with patch.dict("agent_chat.conversation._OPENAI_COMPATIBLE_CLIENTS", {"mistral": lambda: fake_client}):
        call_agent_recorded(_mistral_agent("solo"), [Message(speaker="user", content="hi")], "system", seed=7)

    assert "seed" not in captured[0]


def _unprocessable_entity_error(message: str) -> openai.UnprocessableEntityError:
    response = httpx.Response(422, request=httpx.Request("POST", "https://api.mistral.ai/v1/chat/completions"))
    return openai.UnprocessableEntityError(message, response=response, body=None)


def test_a_422_rejecting_an_optional_kwarg_falls_back_instead_of_aborting():
    # Regression test: Mistral rejects an unrecognized field with a 422
    # UnprocessableEntityError, a sibling of BadRequestError (400) under
    # APIStatusError, not a subclass of it. Before _rejects_optional_kwargs
    # checked APIStatusError instead of just BadRequestError, this aborted
    # the whole conversation instead of retrying without the optional kwargs.
    captured_with_optional: list[bool] = []

    def flaky_create(**kwargs):
        had_seed = "seed" in kwargs
        captured_with_optional.append(had_seed)
        if had_seed:
            raise _unprocessable_entity_error(
                "Extra inputs are not permitted: seed"
            )
        yield SimpleNamespace(
            model="mistral-medium-2505", usage=None,
            choices=[SimpleNamespace(finish_reason=None, delta=SimpleNamespace(content="ok"))],
        )
        yield SimpleNamespace(
            model="mistral-medium-2505",
            usage=SimpleNamespace(prompt_tokens=5, completion_tokens=1, prompt_tokens_details=None),
            choices=[],
        )

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=flaky_create)))

    # mistral is already in _NO_SEED_PROVIDERS, which would itself prevent
    # this from ever being hit — use a provider not on that list so the
    # request really does include seed on the first attempt, exercising
    # _rejects_optional_kwargs's fallback rather than the proactive skip.
    agent = Agent(name="solo", role="You are solo.", model="gpt-5.6-terra", provider="azure_ai")
    with patch.dict("agent_chat.conversation._OPENAI_COMPATIBLE_CLIENTS", {"azure_ai": lambda: fake_client}):
        result = call_agent_recorded(agent, [Message(speaker="user", content="hi")], "system", seed=7)

    assert result.text == "ok"
    assert captured_with_optional == [True, False]  # first attempt had seed, fallback didn't
