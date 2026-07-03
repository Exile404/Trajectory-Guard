"""Token pricing for the abort policy's cost accounting.

Rates are USD per 1M tokens, input and output listed separately because every
hosted backend charges them differently (output runs 4-5x input). Sources:
Amazon Bedrock on-demand + Anthropic list price (same on Bedrock), 2026-07.
Local ollama is free (own GPU) -> zero rate, kept so one code path prices every
backend in the PROVIDER = ollama | nim | bedrock design.

Our trajectory logs store only a cumulative TOTAL token count per step, so to
turn tokens into dollars we blend the two rates with an assumed input fraction.
A self-healing repair loop is context-heavy -- every step resends the task,
current code, error history and test output, and emits one short function -- so
input dominates. INPUT_FRAC = 0.75 is deliberately conservative: a higher input
fraction lowers the blended rate, so if anything this UNDER-states the savings
rather than inflating them. Override per call if you measure the real split
(the Bedrock harvest can log it exactly).
"""

from __future__ import annotations

# USD per 1M tokens: (input, output)
PRICES = {
    "nova-lite": (0.06, 0.24),   # amazon.nova-lite   -- the model we validated on
    "nova-pro":  (0.80, 3.20),   # amazon.nova-pro
    "sonnet":    (3.00, 15.00),  # anthropic.claude-sonnet-5
    "opus":      (5.00, 25.00),  # anthropic.claude-opus-4-8 -- showcase model
    "local":     (0.0,  0.0),    # ollama on our own GPU
}

INPUT_FRAC = 0.75


def resolve(model_id: str) -> str:
    """Map a Bedrock/OLLAMA model id to a pricing profile key."""
    m = (model_id or "").lower()
    if "opus" in m:
        return "opus"
    if "sonnet" in m:
        return "sonnet"
    if "nova-pro" in m:
        return "nova-pro"
    if "nova" in m:
        return "nova-lite"
    return "local"


def rate_per_token(profile: str, input_frac: float = INPUT_FRAC) -> float:
    """Blended USD per single token for a profile."""
    inp, out = PRICES[profile]
    return (input_frac * inp + (1.0 - input_frac) * out) / 1_000_000


def cost(total_tokens: float, profile: str = "nova-lite",
         input_frac: float = INPUT_FRAC) -> float:
    """Dollars for a cumulative total-token count at a profile's blended rate."""
    return float(total_tokens) * rate_per_token(profile, input_frac)