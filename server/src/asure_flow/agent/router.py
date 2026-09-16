"""Multi-provider LLM router — data-driven LiteLLM Router with ordered fallback."""

from __future__ import annotations

import logging

from litellm import Router

from asure_flow.config import Settings, settings

logger = logging.getLogger(__name__)


def build_router() -> Router | None:
    """
    Build a LiteLLM Router from the data-driven providers list.

    Iterates ``settings.providers`` in order so list position = fallback priority.
    Returns None if no providers are configured.
    """
    model_list: list[dict] = []

    for order, provider in enumerate(settings.providers, start=1):
        if not provider.enabled:
            continue
        # Hosted providers require a key even when they define a standard API
        # base (OpenRouter does). Keyless operation is only valid for custom
        # OpenAI-compatible endpoints such as Ollama/LM Studio.
        keyless_custom = provider.litellm_prefix == "openai" and bool(provider.api_base)
        if not provider.api_key and not keyless_custom:
            continue

        entry: dict = {
            "model_name": "assistant",
            "litellm_params": {
                "model": f"{provider.litellm_prefix}/{provider.model}",
                "api_key": provider.api_key or "not-needed",
                "order": order,
            },
        }
        if provider.api_base:
            entry["litellm_params"]["api_base"] = provider.api_base

        model_list.append(entry)

        # A distinct logical route lets live conversation turns use a faster
        # model while summaries and follow-ups retain the quality model. When
        # unset it cleanly falls back to the provider's normal model.
        fast_entry = {
            "model_name": "assistant_realtime",
            "litellm_params": {
                **entry["litellm_params"],
                "model": f"{provider.litellm_prefix}/{provider.realtime_model or provider.model}",
            },
        }
        model_list.append(fast_entry)

    if not model_list:
        logger.warning("No LLM providers configured — AI features will be unavailable")
        return None

    logger.info("LLM router configured with %d provider(s)", len(model_list) // 2)

    strategy = settings.routing_strategy
    kwargs: dict = {
        "model_list": model_list,
        "enable_pre_call_checks": True,
        "num_retries": 1,
        "retry_after": 1,
        "allowed_fails": 1,
        "cooldown_time": 15,
        "timeout": 20,
        "set_verbose": False,
    }
    if strategy and strategy != "simple-shuffle":
        kwargs["routing_strategy"] = strategy

    return Router(**kwargs)


llm_router: Router | None = None


def init_router() -> None:
    global llm_router
    llm_router = build_router()


def get_router() -> Router | None:
    return llm_router
