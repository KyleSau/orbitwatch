"""Model prices, in ONE place, for estimating the cost of a traced run.

!!! CHECK THESE AGAINST ANTHROPIC'S CURRENT PRICING BEFORE TRUSTING A DOLLAR FIGURE !!!
https://www.anthropic.com/pricing  (or the pricing page of the Claude API docs)

Values are USD per million tokens, copied on 2026-10-13 from the model table in the
Claude API reference material (cached 2026-10-06). Cache writes use the documented
1.25x-of-input rate for the default 5-minute TTL. Cache-read rates for Opus 5.5 and
Sonnet 5.5 ($0.20) are documented; the Haiku 5.5 cache-read rate below is ASSUMED to be
the usual 0.1x of input and was not confirmed. Prices change: these are estimates for
dashboards and budgets, never for billing.

A model not listed here gets ``None`` (unknown cost), not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PRICES_CHECKED_ON = "2026-10-13"


@dataclass(frozen=True)
class ModelPrice:
    input: float  # USD per 1M uncached input tokens
    output: float  # USD per 1M output tokens (thinking tokens are billed as output)
    cache_read: float  # USD per 1M cache-read input tokens
    cache_write: float  # USD per 1M cache-write input tokens (5-minute TTL)


PRICES_PER_MTOK: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
}

USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def usage_dict(usage: Any) -> dict[str, int]:
    """The four billable token counts from an SDK ``usage`` object (missing -> 0)."""
    return {f: int(getattr(usage, f, 0) or 0) for f in USAGE_FIELDS}


def estimate_cost_usd(model: str | None, usage: dict[str, int]) -> float | None:
    """Estimated cost of one request, or None when the model's price is unknown.

    ``input_tokens`` is the *uncached* remainder; cache reads and writes are priced
    separately (total prompt = input + cache_read + cache_creation).
    """
    price = PRICES_PER_MTOK.get(model or "")
    if price is None:
        return None
    total = (
        usage.get("input_tokens", 0) * price.input
        + usage.get("output_tokens", 0) * price.output
        + usage.get("cache_read_input_tokens", 0) * price.cache_read
        + usage.get("cache_creation_input_tokens", 0) * price.cache_write
    )
    return round(total / 1_000_000, 6)
