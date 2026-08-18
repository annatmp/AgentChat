"""
Usage mapping for both provider shapes.

The two SDKs report usage under different names (`input_tokens` vs
`prompt_tokens`), and getting it wrong would silently zero out the cost and
token metrics for one whole panel.
"""

from types import SimpleNamespace

from agent_chat.conversation import usage_from_anthropic, usage_from_openai
from agent_chat.pricing import cost_usd


def test_anthropic_usage_includes_cache_fields():
    usage = usage_from_anthropic(SimpleNamespace(
        input_tokens=1200, output_tokens=340,
        cache_read_input_tokens=800, cache_creation_input_tokens=100,
    ))
    assert (usage.input_tokens, usage.output_tokens) == (1200, 340)
    assert usage.cache_read_input_tokens == 800
    assert usage.cache_creation_input_tokens == 100
    assert usage.available is True


def test_anthropic_usage_tolerates_missing_cache_fields():
    usage = usage_from_anthropic(SimpleNamespace(input_tokens=10, output_tokens=2))
    assert usage.cache_read_input_tokens == 0
    assert usage.cache_creation_input_tokens == 0


def test_openai_usage_maps_prompt_and_completion_tokens():
    usage = usage_from_openai(SimpleNamespace(
        prompt_tokens=900, completion_tokens=120,
        prompt_tokens_details=SimpleNamespace(cached_tokens=64),
    ))
    assert (usage.input_tokens, usage.output_tokens) == (900, 120)
    assert usage.cache_read_input_tokens == 64
    assert usage.available is True


def test_missing_openai_usage_is_marked_unavailable_not_zero():
    # Some Azure AI Foundry deployments reject stream_options; the record must
    # say "not measured" rather than reporting a free call.
    usage = usage_from_openai(None)
    assert usage.available is False
    assert usage.input_tokens == 0
    assert cost_usd("claude-sonnet-4-6", usage) is None


def test_cost_uses_cache_multipliers():
    usage = usage_from_anthropic(SimpleNamespace(
        input_tokens=1_000_000, output_tokens=0,
        cache_read_input_tokens=1_000_000, cache_creation_input_tokens=0,
    ))
    # 1M full-price input at $3 + 1M cache reads at 0.1x = $3.30
    assert cost_usd("claude-sonnet-4-6", usage) == 3.3


def test_unpriced_model_returns_none():
    usage = usage_from_openai(SimpleNamespace(prompt_tokens=10, completion_tokens=1,
                                             prompt_tokens_details=None))
    assert cost_usd("Llama-4-Scout-17B-16E-Instruct", usage) is None
