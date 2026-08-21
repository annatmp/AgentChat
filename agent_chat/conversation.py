from __future__ import annotations

import os

import time
from dataclasses import dataclass, field
from functools import cache
from typing import Callable, cast

import anthropic
import openai
from anthropic.types import MessageParam
from openai import AzureOpenAI, OpenAI

from agent_chat.agents import Agent
from agent_chat.pricing import cost_usd
from agent_chat.records import KIND_TURN, CallRecord, SelectorLog, TurnRecord, Usage
from agent_chat.retry import call_with_retry
from agent_chat.sanitize import strip_speaker_echo

# Prefix for a role's private context, so the agent treats it as knowledge only
# it holds rather than as shared meeting material.
KNOWLEDGE_HEADER = "Background knowledge only you have. Bring it into the discussion when relevant:"


@dataclass
class Message:
    speaker: str    # agent name, or "user" for human input
    content: str


class AgentCallError(RuntimeError):
    """
    An LLM call that could not be completed, carrying its partial record.

    Raised rather than swallowed: a run that lost a turn is a failed run, and
    EXPERIMENT_DESIGN §5 wants those recorded, not silently retried to success.
    """

    def __init__(self, record: CallRecord):
        super().__init__(record.error or "agent call failed")
        self.record = record


# Clients are built with max_retries=0 so no retry happens invisibly inside the
# SDK — retrying is done in retry.py, where the attempt count reaches the record.

@cache
def _azure_openai_client() -> AzureOpenAI:
    """AzureOpenAI client for classic Azure OpenAI GPT deployments."""
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        max_retries=0,
    )


@cache
def _azure_ai_client() -> OpenAI:
    """Plain OpenAI-compatible client for Azure AI Foundry serverless models (Mistral, Llama, etc.)."""
    return OpenAI(
        base_url=os.environ["AZURE_AI_ENDPOINT"],
        api_key=os.environ["AZURE_AI_API_KEY"],
        max_retries=0,
    )


@cache
def _anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], max_retries=0)


# Fixed, not per-resource like the Azure endpoints, since Gemini's
# OpenAI-compatibility layer lives at one public URL for every account.
GOOGLE_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@cache
def _google_client() -> OpenAI:
    """OpenAI-compatible client for Google's Gemini API."""
    return OpenAI(
        base_url=GOOGLE_BASE_URL,
        api_key=os.environ["GOOGLE_API_KEY"],
        max_retries=0,
    )


# Fixed, like GOOGLE_BASE_URL — Mistral's La Plateforme API, not a per-resource
# Azure AI Foundry deployment.
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"


@cache
def _mistral_client() -> OpenAI:
    """OpenAI-compatible client for Mistral's own API (La Plateforme)."""
    return OpenAI(
        base_url=MISTRAL_BASE_URL,
        api_key=os.environ["MISTRAL_API_KEY"],
        max_retries=0,
    )


# Fixed, like the other direct-vendor URLs above. DeepSeek's own docs use this
# exact base_url (no /v1 suffix — their REST paths don't have one).
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


@cache
def _deepseek_client() -> OpenAI:
    """OpenAI-compatible client for DeepSeek's own API."""
    return OpenAI(
        base_url=DEEPSEEK_BASE_URL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        max_retries=0,
    )


# Every non-Anthropic provider speaks the OpenAI-compatible chat.completions
# API; only the client (base URL + auth) differs between them.
_OPENAI_COMPATIBLE_CLIENTS: dict[str, Callable[[], OpenAI]] = {
    "azure_openai": _azure_openai_client,
    "azure_ai": _azure_ai_client,
    "google": _google_client,
    "mistral": _mistral_client,
    "deepseek": _deepseek_client,
}


def _build_history(agent: Agent, history: list[Message]) -> list[dict]:
    """
    Map shared conversation history to the messages list for one agent.

    The agent sees its own past turns as "assistant" and everyone else's
    as "user". Agent names are embedded as [name]: prefixes so the model
    understands who said what — the standard speaker-tag pattern used by
    most multi-agent frameworks (AutoGen, CrewAI, etc.).
    """
    messages = []
    for msg in history:
        if msg.speaker == agent.name:
            messages.append({"role": "assistant", "content": msg.content})
        else:
            prefix = "" if msg.speaker == "user" else f"[{msg.speaker}]: "
            messages.append({"role": "user", "content": f"{prefix}{msg.content}"})
    return messages


@dataclass
class CallResult:
    text: str
    record: CallRecord


# --- Provider calls ---

# Substrings that identify "this endpoint doesn't accept that optional kwarg",
# as opposed to a genuinely malformed request.
_OPTIONAL_KWARG_HINTS = (
    "stream_options", "include_usage", "seed",
    "unsupported", "unknown", "unrecognized", "extra inputs",
)

# Providers whose OpenAI-compatible endpoint rejects `seed` on every call
# (verified live, not just "some deployments don't support it" like Azure AI
# Foundry's stream_options quirk) — skipped proactively so every call to
# these doesn't pay for a guaranteed-failing first attempt.
_NO_SEED_PROVIDERS = frozenset({"google", "mistral"})


def _rejects_optional_kwargs(exc: BaseException) -> bool:
    # openai.APIStatusError, not just BadRequestError: Mistral rejects an
    # unrecognized field (like seed) with a 422 UnprocessableEntityError, a
    # sibling class of BadRequestError (400) under APIStatusError, not a
    # subclass of it. Narrowing to BadRequestError alone meant Mistral's
    # rejection was never caught here — it aborted the whole conversation
    # instead of retrying without the optional kwargs, unlike every other
    # OpenAI-compatible provider's 400-shaped version of the same rejection.
    return isinstance(exc, openai.APIStatusError) and any(
        hint in str(exc).lower() for hint in _OPTIONAL_KWARG_HINTS
    )


def usage_from_anthropic(usage_obj) -> Usage:
    """Map an Anthropic `Message.usage` to our Usage. Pure, so it's unit-tested."""
    return Usage(
        input_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
        output_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
        cache_read_input_tokens=getattr(usage_obj, "cache_read_input_tokens", 0) or 0,
        cache_creation_input_tokens=getattr(usage_obj, "cache_creation_input_tokens", 0) or 0,
    )


def usage_from_openai(usage_obj) -> Usage:
    """
    Map an OpenAI-compatible `CompletionUsage` to our Usage.

    `available=False` when the endpoint sent no usage object at all, so a run
    record never shows measured-looking zeros.

    Unlike Anthropic, `prompt_tokens` here already *includes* the cached
    subset (`prompt_tokens_details.cached_tokens`) rather than excluding it —
    so it's subtracted out here to give `input_tokens` the same "full-price,
    cache excluded" meaning Anthropic's field has. `cost_usd` bills the cache
    buckets additively (Anthropic's pricing model) for every provider; without
    this subtraction every cached token would be billed once at full price
    (still inside prompt_tokens) and again at the cache-read multiplier.
    """
    if usage_obj is None:
        return Usage(available=False)
    details = getattr(usage_obj, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
    return Usage(
        input_tokens=prompt_tokens - cached,
        output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
        cache_read_input_tokens=cached,
    )


def _call_anthropic(
    agent: Agent, messages: list[dict], system: str, temperature: float | None,
    emit: Callable[[str], None], emitted: list[str], record: CallRecord,
    max_attempts: int, on_retry, ) -> str:
    kwargs: dict = {
        "model": agent.model,
        "system": system,
        "messages": cast(list[MessageParam], messages),
        "max_tokens": agent.max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    # Anthropic has no seed parameter; record.seed carries it as a replicate label.

    def attempt():
        with _anthropic_client().messages.stream(**kwargs) as stream:
            for token in stream.text_stream:
                emit(token)
            return stream.get_final_message()

    message, retries = call_with_retry(
        attempt, max_attempts=max_attempts,
        retry_allowed=lambda: not emitted, on_retry=on_retry,
    )
    record.retries = retries
    record.model_resolved = message.model
    record.stop_reason = message.stop_reason
    record.usage = usage_from_anthropic(message.usage)
    return "".join(block.text for block in message.content if block.type == "text")


def _call_openai_compatible(
    agent: Agent, messages: list[dict], system: str, temperature: float | None,
    seed: int | None, emit: Callable[[str], None], emitted: list[str],
    record: CallRecord, max_attempts: int, on_retry,
) -> str:
    client = _OPENAI_COMPATIBLE_CLIENTS[agent.provider]()
    payload = [{"role": "system", "content": system}] + messages
    if agent.provider == "google" and payload[-1]["role"] == "assistant":
        # Gemini's own API (which this OpenAI-compatibility layer sits on top
        # of) 400s on "Requests ending with a model turn are not supported" —
        # e.g. the same agent speaking twice in a row with no one else's turn
        # in between. Every other provider tolerates this. The nudge is
        # call-local only, like extra_context; it never touches self.history.
        payload = payload + [{"role": "user", "content": "Continue."}]

    def attempt(with_optional: bool):
        kwargs: dict = {"model": agent.model, "messages": payload, "stream": True}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if with_optional:
            # Usage is only streamed back when asked for, and arrives in a final
            # chunk whose `choices` list is empty.
            kwargs["stream_options"] = {"include_usage": True}
            # Best-effort on OpenAI-compatible endpoints in general, but a
            # few reject it outright on every call (see _NO_SEED_PROVIDERS),
            # so it's skipped there rather than paying for a guaranteed-
            # failing first attempt each time.
            if seed is not None and agent.provider not in _NO_SEED_PROVIDERS:
                kwargs["seed"] = seed
        chunks: list[str] = []
        usage_obj = None
        for chunk in client.chat.completions.create(**kwargs):  # type: ignore[arg-type]
            if getattr(chunk, "model", None):
                record.model_resolved = chunk.model
            if getattr(chunk, "usage", None):
                usage_obj = chunk.usage
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                record.stop_reason = choice.finish_reason
            if choice.delta and choice.delta.content:
                emit(choice.delta.content)
                chunks.append(choice.delta.content)
        return "".join(chunks), usage_obj

    def run(with_optional: bool):
        return call_with_retry(
            lambda: attempt(with_optional), max_attempts=max_attempts,
            retry_allowed=lambda: not emitted, on_retry=on_retry,
        )

    try:
        (text, usage_obj), retries = run(True)
    except openai.APIStatusError as exc:
        # Not every Azure AI Foundry serverless deployment accepts stream_options
        # or seed, and Mistral rejects seed with a 422 (UnprocessableEntityError)
        # rather than a 400 (BadRequestError) — a sibling exception under
        # APIStatusError, not a subclass of BadRequestError, so this must catch
        # the shared base or Mistral's rejection aborts the call outright instead
        # of falling back. Fall back once without the optional kwargs and mark
        # usage unavailable, rather than reporting zeros as if they had been measured.
        if emitted or not _rejects_optional_kwargs(exc):
            raise
        (text, usage_obj), retries = run(False)

    record.retries = retries
    record.usage = usage_from_openai(usage_obj)
    return text


def call_agent_recorded(
    agent: Agent,
    history: list[Message],
    system: str,
    *,
    kind: str = KIND_TURN,
    on_token: Callable[[str], None] | None = None,
    temperature: float | None = None,
    seed: int | None = None,
    max_attempts: int = 5,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> CallResult:
    """
    Call one agent and record what it cost.

    `kind` separates conversation turns from selector-side calls (bids, think
    steps) and the summary, which `compute_totals` reports separately.
    """
    messages = _build_history(agent, history)
    resolved_temperature = agent.temperature if temperature is None else temperature
    record = CallRecord(
        agent=agent.name,
        kind=kind,
        provider=agent.provider,
        model_requested=agent.model,
        temperature=resolved_temperature,
        seed=seed,
    )
    emitted: list[str] = []  # non-empty once tokens have reached the caller

    def emit(token: str) -> None:
        emitted.append(token)
        if on_token:
            on_token(token)

    started = time.perf_counter()
    try:
        if agent.provider == "anthropic":
            text = _call_anthropic(
                agent, messages, system, resolved_temperature,
                emit, emitted, record, max_attempts, on_retry,
            )
        elif agent.provider in _OPENAI_COMPATIBLE_CLIENTS:
            text = _call_openai_compatible(
                agent, messages, system, resolved_temperature, seed,
                emit, emitted, record, max_attempts, on_retry,
            )
        else:
            raise ValueError(f"Unknown provider: {agent.provider}")
    except BaseException as exc:
        record.latency_s = time.perf_counter() - started
        record.error = f"{type(exc).__name__}: {exc}"
        record.cost_usd = cost_usd(agent.model, record.usage)
        raise AgentCallError(record) from exc

    record.latency_s = time.perf_counter() - started
    record.cost_usd = cost_usd(agent.model, record.usage)
    return CallResult(text=text, record=record)


def call_agent(
    agent: Agent,
    history: list[Message],
    system: str,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Text-only call. Thin wrapper for callers that don't need the run record."""
    return call_agent_recorded(agent, history, system, on_token=on_token).text


StopCondition = Callable[[list[Message]], bool]
TurnSelector = Callable[[list[Message], dict[str, Agent]], str]
PostProcessor = Callable[[list[Message]], None]



@dataclass
class Conversation:
    agents: dict[str, Agent]
    history: list[Message] = field(default_factory=list)
    system_prompt: str = ""             # prepended to every agent's role
    knowledge: dict[str, str] = field(default_factory=dict)  # agent name -> private context
    temperature: float | None = None    # overrides per-agent temperature
    seed: int | None = None

    # Populated as the conversation runs; read by main.py into the RunRecord.
    turns: list[TurnRecord] = field(default_factory=list)
    selector_calls: list[CallRecord] = field(default_factory=list)
    failed_calls: list[CallRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def user(self, content: str) -> None:
        """Inject a human message into the shared history."""
        self.history.append(Message(speaker="user", content=content))



    def _note_retry(self, attempt: int, delay: float, exc: BaseException) -> None:
        # Printed so the retry is visible in the terminal and the tee'd log;
        # the count itself lands in the CallRecord.
        print(f"\n[retry {attempt} in {delay:.1f}s after {type(exc).__name__}]", flush=True)

    def _check_speaker(self, name: str | None) -> None:
        if not name or name not in self.agents:
            roster = ", ".join(sorted(self.agents))
            message = f"turn selector returned {name!r}, which is not in the roster ({roster})"
            self.errors.append(message)
            raise ValueError(message)

    def step(
        self,
        agent_name: str,
        on_token: Callable[[str], None] | None = None,
        selector: dict | None = None,
        extra_context: str | None = None,
    ) -> Message:
        """Let one agent respond to the current history."""
        agent = self.agents[agent_name]
        # Ephemeral, call-local only — never appended to self.history, same
        # non-mutating pattern the strategies' own private calls already use.
        history_for_call = (
            self.history + [Message(speaker="user", content=extra_context)]
            if extra_context else self.history
        )
        try:
            result = call_agent_recorded(
                agent, history_for_call, system_for(agent, self.system_prompt, self.knowledge.get(agent.name, "")),
                kind=KIND_TURN, on_token=on_token,
                temperature=self.temperature, seed=self.seed,
                on_retry=self._note_retry,
            )
        except AgentCallError as exc:
            self.errors.append(f"turn {len(self.turns)} ({agent_name}): {exc.record.error}")
            self.failed_calls.append(exc.record)
            raise

        raw = result.text
        content = strip_speaker_echo(agent_name, raw)
        msg = Message(speaker=agent_name, content=content)
        self.history.append(msg)
        self.turns.append(TurnRecord(
            index=len(self.turns),
            speaker=agent_name,
            content=content,
            content_raw=raw if raw != content else None,
            call=result.record,
            selector=selector,
        ))
        return msg

    def run(
        self,
        turn_selector: TurnSelector,
        stop_condition: StopCondition,
        stream: bool = False,
        post_processors: list[PostProcessor] | None = None,
        selector_log: SelectorLog | None = None,
        turn_context: Callable[[int], str] | None = None,
    ) -> list[Message]:
        """
        Run until stop_condition is met, using turn_selector to pick each speaker.

        `turn_context`, if given, is called with the current turn count before
        each turn and its result passed as that turn's `extra_context` — e.g.
        the solo baseline's "REVIEW ROUND N" framing. Unused by every other
        condition.
        """
        produced = []
        try:
            while not stop_condition(self.history):
                name = turn_selector(self.history, self.agents)
                rationale = selector_log.drain() if selector_log else None
                if selector_log:
                    self.selector_calls.extend(selector_log.take_calls())
                self._check_speaker(name)
                context = turn_context(len(self.turns)) if turn_context else None
                if stream:
                    print(f"\n[{name.upper()}]\n", flush=True)
                    msg = self.step(
                        name,
                        on_token=lambda t: print(t, end="", flush=True),
                        selector=rationale,
                        extra_context=context,
                    )
                    print()
                else:
                    msg = self.step(name, selector=rationale, extra_context=context)
                produced.append(msg)
        finally:
            if selector_log:
                # Catches calls a stop_condition or turn_selector made but never
                # got to drain through the loop above — either the final call
                # that exited the loop normally, or one that raised (e.g. a
                # facilitator exhausting its retries). Either way those calls
                # cost money and must not vanish from the run's totals.
                self.selector_calls.extend(selector_log.take_calls())
        for processor in (post_processors or []):
            processor(self.history)
        return produced

def system_for(agent: Agent, system_prompt: str, private_knowledge: str) -> str:
    private_block = f"{KNOWLEDGE_HEADER}\n{private_knowledge}" if private_knowledge else None
    return "\n\n".join(filter(None, [system_prompt, agent.role, private_block]))