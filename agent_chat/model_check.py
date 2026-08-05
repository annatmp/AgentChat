"""
Verify every configured model actually exists, without spending a generation
token.

`models.list()` is a metadata call on all three provider SDKs — it costs no
input/output tokens, unlike a real turn — so it's a legitimate pre-flight
check. Not every Azure AI Foundry deployment exposes it (the same kind of
quirk as the `include_usage` flag noted in conversation.py); when a provider's
endpoint doesn't support listing, that provider's models are reported as
unverified rather than failed, so a missing feature doesn't masquerade as a
missing model.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_chat.agents import Agent
from agent_chat.conversation import (
    _anthropic_client,
    _azure_ai_client,
    _azure_openai_client,
    _deepseek_client,
    _google_client,
    _mistral_client,
)


@dataclass
class ModelCheck:
    provider: str
    model: str
    status: str   # "ok" | "not_found" | "unverified"
    detail: str = ""


def _list_ids(provider: str) -> set[str]:
    if provider == "anthropic":
        return {m.id for m in _anthropic_client().models.list()}
    if provider == "azure_openai":
        return {m.id for m in _azure_openai_client().models.list()}
    if provider == "azure_ai":
        return {m.id for m in _azure_ai_client().models.list()}
    if provider == "google":
        # The OpenAI-compat models.list() relays Gemini's native resource names
        # (`models/gemini-...`), but chat.completions.create wants the bare id
        # (matching the agent YAMLs) — strip the prefix or every model 404s here.
        return {m.id.removeprefix("models/") for m in _google_client().models.list()}
    if provider == "mistral":
        return {m.id for m in _mistral_client().models.list()}
    if provider == "deepseek":
        return {m.id for m in _deepseek_client().models.list()}
    raise ValueError(f"unknown provider: {provider}")


def check_models(agents: dict[str, Agent], summarizer: Agent) -> list[ModelCheck]:
    """One models.list() call per provider in use, then match every (provider, model) pair."""
    pairs = {(a.provider, a.model) for a in (*agents.values(), summarizer)}
    providers = {provider for provider, _ in pairs}

    catalog: dict[str, set[str] | Exception] = {}
    for provider in providers:
        try:
            catalog[provider] = _list_ids(provider)
        except Exception as exc:  # auth failure, unsupported endpoint, network error
            catalog[provider] = exc

    results = []
    for provider, model in sorted(pairs):
        listing = catalog[provider]
        if isinstance(listing, Exception):
            results.append(ModelCheck(
                provider, model, "unverified",
                f"{type(listing).__name__}: {listing}",
            ))
        elif model in listing:
            results.append(ModelCheck(provider, model, "ok"))
        else:
            results.append(ModelCheck(provider, model, "not_found"))
    return results
