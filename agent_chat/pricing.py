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
    "claude-sonnet-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
    # Azure OpenAI / Azure AI Foundry deployments are priced per subscription
    # and the `model` field there is a deployment name, so they are deliberately
    # absent. Add your own rates here to get cost totals for those panels.
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
