"""
Token pricing, so a run record carries cost alongside quality.

EXPERIMENT_DESIGN §5 wants cost reported as a first-class metric next to
quality, not folded into it. An unknown model returns `None` rather than 0 — a
missing price must not read as a free run.

Rates are USD per million tokens (input, output). Update when pricing changes;
`cost_complete` in the run record's totals tells you when this table has a gap.
"""

from __future__ import annotations

from agent_chat.records import Usage

# USD per 1M tokens: model -> (input, output)
PRICING: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    # docs/experiment_setup.md's rate card is date-tiered: $2/$10 through
    # 2026-08-31, $3/$15 from 2026-09-01. This is the pre-2026-09-01 tier —
    # not date-aware, just correct for now; update by hand after the switch.
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
    # gemini-3.6-flash and gpt-5.6-terra rates below are from
    # docs/experiment_setup.md's published table (gpt-5.6-terra: the "long
    # context, Global" row — its own cache figures, $0.50 cached-input /
    # $6.25 cache-write against a $5 base, work out to exactly the 0.1x/1.25x
    # multipliers below, so this repo's universal cache-multiplier model
    # isn't just an Anthropic assumption for this model either).
    "gemini-3.6-flash": (1.50, 7.50),      # google — flash tier
    "gpt-5.6-terra": (5.00, 22.50),        # azure_ai — long-context, Global pricing
    # deepseek-v4-flash has no published rate in experiment_setup.md's table —
    # still a rough estimate, not a vendor rate card. Good enough for relative
    # cost comparison between runs, not for a real invoice.
    "deepseek-v4-flash": (0.30, 1.20),     # deepseek — DeepSeek's flash tier has stayed well below frontier pricing
    # Azure OpenAI / Azure AI Foundry deployments are otherwise priced per
    # subscription and the `model` field there is a deployment name, so they
    # are deliberately absent. Add your own rates here to get cost totals for
    # those panels.
}

# Cache multipliers relative to the input rate (Anthropic pricing model).
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def cost_usd(model: str, usage: Usage) -> float | None:
    """Cost of one call, or None if we have no price for the model."""
    rates = PRICING.get(model)
    if rates is None or not usage.available:
        return None
    input_rate, output_rate = rates
    billable_input = (
        usage.input_tokens
        + usage.cache_read_input_tokens * CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * CACHE_WRITE_MULTIPLIER
    )
    return (billable_input * input_rate + usage.output_tokens * output_rate) / 1_000_000
