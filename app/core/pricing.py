"""List prices (USD per 1M tokens) used for per-request cost accounting.

The defaults are the serverless prices of the hosted provider used for the benchmarks in the
README (verified 2026-10-12). The same open-weight model costs different amounts on different
providers, so set MODEL_PRICES for yours. Unknown models report cost as 0.
"""

from __future__ import annotations

from app.core.ports import Usage, model_key

# model_key(model) -> (input, cached_input, output)
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-oss-120b": (0.15, 0.015, 0.60),
    "gpt-oss-20b": (0.07, 0.007, 0.30),
    "deepseek-v4p1-flash": (0.30, 0.006, 1.20),
    "minimax-m3": (0.30, 0.06, 1.20),
    "glm-5p3": (1.40, 0.26, 4.40),
    "glm-5p3-flash": (0.15, 0.03, 0.50),
    "kimi-k3": (3.00, 0.30, 15.00),
    "nemotron-lightning-3p5-30b-a3b": (0.05, 0.01, 0.20),
    "qwen3-embedding-8b": (0.10, 0.10, 0.0),
}


def cost_usd(model: str, usage: Usage) -> float:
    price = PRICES.get(model_key(model))
    if price is None:
        return 0.0
    p_in, p_cached, p_out = price
    uncached = max(usage.prompt_tokens - usage.cached_tokens, 0)
    return (
        uncached * p_in + usage.cached_tokens * p_cached + usage.completion_tokens * p_out
    ) / 1e6
