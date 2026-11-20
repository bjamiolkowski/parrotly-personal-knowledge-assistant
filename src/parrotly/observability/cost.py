from src.config import MODEL_PRICING_USD_PER_1M_TOKENS


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate cost in USD."""
    pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model)

    if pricing is None:
        return 0.0

    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    cost = input_cost + output_cost

    return round(cost, 6)


def estimate_tokens(text: str) -> int:
    """Estimate token count."""
    if not text:
        return 0

    return max(1, len(text) // 4)