"""Shared context-window limits for autoregressive and speculative decoding."""

from typing import Any, Tuple


def limit_generation_to_context(
    model_or_config: Any,
    input_length: int,
    max_new_tokens: int,
    min_new_tokens: int = 0,
    *,
    label: str = "GEN",
) -> Tuple[int, int]:
    """Cap requested output lengths to positions available after the prompt."""
    input_length = int(input_length)
    max_new_tokens = int(max_new_tokens)
    min_new_tokens = int(min_new_tokens)
    if input_length < 0:
        raise ValueError("input_length must be nonnegative")
    if max_new_tokens < 0 or min_new_tokens < 0:
        raise ValueError("generation token counts must be nonnegative")

    config = getattr(model_or_config, "config", model_or_config)
    text_config = getattr(config, "text_config", config)
    context_window = getattr(text_config, "max_position_embeddings", None)
    if context_window is None:
        generation_limit = max_new_tokens
    else:
        try:
            context_window = int(context_window)
        except (TypeError, ValueError) as exc:
            raise ValueError("max_position_embeddings must be an integer") from exc
        if context_window < 1:
            raise ValueError("max_position_embeddings must be positive")

        available_tokens = max(0, context_window - input_length)
        if min_new_tokens > available_tokens:
            print(
                f"[{label}] WARNING: input length {input_length} leaves only "
                f"{available_tokens} generation positions, below the requested "
                f"minimum of {min_new_tokens}."
            )
        if max_new_tokens > available_tokens:
            print(
                f"[{label}] Capping max_new_tokens from {max_new_tokens} to "
                f"{available_tokens} for the {context_window}-token model context."
            )
        generation_limit = min(max_new_tokens, available_tokens)

    return generation_limit, min(min_new_tokens, generation_limit)
