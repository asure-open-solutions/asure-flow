"""Multi-provider LLM router — LiteLLM Router with ordered fallback."""

from __future__ import annotations

import logging
from typing import Callable

from litellm import Router

from asure_flow.config import Settings, settings

logger = logging.getLogger(__name__)

# Provider registry: key → builder that returns a model entry or None.
# Each builder checks both the enabled flag and required credentials.
_PROVIDER_BUILDERS: dict[str, Callable[[Settings], dict | None]] = {
    "openrouter": lambda s: {
        "model_name": "assistant",
        "litellm_params": {
            "model": f"openrouter/{s.openrouter_model}",
            "api_key": s.openrouter_api_key,
            "api_base": "https://openrouter.ai/api/v1",
        },
    } if s.openrouter_api_key and s.openrouter_enabled else None,

    "openai": lambda s: {
        "model_name": "assistant",
        "litellm_params": {
            "model": f"openai/{s.openai_model}",
            "api_key": s.openai_api_key,
        },
    } if s.openai_api_key and s.openai_enabled else None,

    "gemini": lambda s: {
        "model_name": "assistant",
        "litellm_params": {
            "model": f"gemini/{s.gemini_model}",
            "api_key": s.gemini_api_key,
        },
    } if s.gemini_api_key and s.gemini_enabled else None,

    "huggingface": lambda s: {
        "model_name": "assistant",
        "litellm_params": {
            "model": f"huggingface/{s.hf_model}",
            "api_key": s.hf_api_key,
        },
    } if s.hf_api_key and s.hf_enabled else None,

    "github": lambda s: {
        "model_name": "assistant",
        "litellm_params": {
            "model": f"openai/{s.github_model}",
            "api_key": s.github_token,
            "api_base": "https://models.inference.ai.azure.com",
        },
    } if s.github_token and s.github_enabled else None,

    "custom": lambda s: {
        "model_name": "assistant",
        "litellm_params": {
            "model": f"openai/{s.custom_model}",
            "api_key": s.custom_api_key or "not-needed",
            "api_base": s.custom_api_base,
        },
    } if s.custom_api_base and s.custom_model and s.custom_enabled else None,
}


def build_router() -> Router | None:
    """
    Build a LiteLLM Router from configured providers.

    Iterates ``settings.provider_order`` so the user controls fallback priority.
    Returns None if no providers are configured.
    """
    model_list: list[dict] = []
    order = 0

    for provider_key in settings.provider_order:
        builder = _PROVIDER_BUILDERS.get(provider_key)
        if not builder:
            logger.warning("Unknown provider key in provider_order: %s", provider_key)
            continue
        entry = builder(settings)
        if entry:
            order += 1
            entry["litellm_params"]["order"] = order
            model_list.append(entry)

    if not model_list:
        logger.warning("No LLM providers configured — AI features will be unavailable")
        return None

    logger.info("LLM router configured with %d provider(s)", len(model_list))

    return Router(
        model_list=model_list,
        enable_pre_call_checks=True,
        num_retries=2,
        retry_after=1,
        allowed_fails=1,
        cooldown_time=60,
        timeout=30,
        fallbacks=[{"assistant": ["assistant"]}],
    )


llm_router: Router | None = None


def init_router() -> None:
    global llm_router
    llm_router = build_router()


def get_router() -> Router | None:
    return llm_router
