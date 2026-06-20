"""
Pure diagnostic helper functions for LiteLLM completion.

These are deterministic, side-effect free functions extracted for testability.
They contain no imports that trigger settings loading or external dependencies.
"""

from typing import Any, Optional


def format_litellm_completion_error_context(
    *,
    model_key: str,
    model_name: str,
    api_base: Optional[str],
    streaming: bool,
    structured_output: bool,
    num_context_chunks: int,
    num_images: int,
    temperature: Any,
    max_tokens: Any,
    num_retries: Any,
) -> str:
    """Return a formatted string with relevant LiteLLM completion context for debugging."""
    return (
        f"model_key={model_key}, model_name={model_name}, "
        f"api_base={api_base or 'default'}, streaming={streaming}, "
        f"structured_output={structured_output}, num_context_chunks={num_context_chunks}, "
        f"num_images={num_images}, temperature={temperature}, "
        f"max_tokens={max_tokens}, num_retries={num_retries}"
    )
