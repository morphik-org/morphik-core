from typing import Any, Dict

import litellm


def add_output_limit(params: Dict[str, Any], max_output_tokens: int) -> None:
    """Set the output token limit under the name the provider accepts, unless the registered model sets one.

    OpenAI's reasoning models (gpt-5 and later) reject max_tokens, and LiteLLM only renames it for models it already
    knows. Every current OpenAI chat model accepts max_completion_tokens; older Azure API versions don't.
    """
    provider = None
    try:
        provider = litellm.get_llm_provider(params["model"])[1]
    except Exception:
        pass
    token_param = "max_completion_tokens" if provider == "openai" else "max_tokens"
    if "max_tokens" not in params and "max_completion_tokens" not in params:
        params[token_param] = max_output_tokens
    if "reasoning_effort" in params:
        # LiteLLM refuses reasoning_effort for OpenAI models newer than its model list unless allowed.
        params.setdefault("allowed_openai_params", ["reasoning_effort"])
