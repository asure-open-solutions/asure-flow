from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ── Provider Entry (data-driven) ──────────────────────────────────────────────


class ProviderEntry(BaseModel):
    """A single LLM provider configuration."""

    id: str                          # unique key, e.g. "openrouter", "my-ollama"
    name: str                        # display label
    litellm_prefix: str              # LiteLLM routing prefix: "openrouter", "openai", "anthropic", "gemini", "groq", …
    model: str                       # model ID after prefix, e.g. "anthropic/claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    api_base: Optional[str] = None   # custom API base URL (OpenRouter, Ollama, LM Studio, vLLM, …)
    enabled: bool = True


_DEFAULT_PROVIDERS: list[ProviderEntry] = [
    ProviderEntry(id="openrouter", name="OpenRouter", litellm_prefix="openrouter",
                  model="anthropic/claude-sonnet-4-20250514", api_base="https://openrouter.ai/api/v1"),
    ProviderEntry(id="openai", name="OpenAI", litellm_prefix="openai",
                  model="gpt-4.1"),
    ProviderEntry(id="anthropic", name="Anthropic", litellm_prefix="anthropic",
                  model="claude-sonnet-4-20250514"),
    ProviderEntry(id="gemini", name="Google Gemini", litellm_prefix="gemini",
                  model="gemini-2.5-flash"),
    ProviderEntry(id="groq", name="Groq", litellm_prefix="groq",
                  model="llama-3.3-70b-versatile"),
    ProviderEntry(id="custom", name="Custom (Ollama, LM Studio)", litellm_prefix="openai",
                  model=""),
]


# ── Old → new migration map ──────────────────────────────────────────────────

_OLD_PROVIDER_MIGRATION: dict[str, dict] = {
    "openrouter": {"key": "openrouter_api_key", "model": "openrouter_model", "enabled": "openrouter_enabled",
                   "prefix": "openrouter", "name": "OpenRouter",
                   "base_field": None, "default_model": "anthropic/claude-sonnet-4-20250514",
                   "default_base": "https://openrouter.ai/api/v1"},
    "openai":     {"key": "openai_api_key", "model": "openai_model", "enabled": "openai_enabled",
                   "prefix": "openai", "name": "OpenAI",
                   "base_field": None, "default_model": "gpt-4.1", "default_base": None},
    "gemini":     {"key": "gemini_api_key", "model": "gemini_model", "enabled": "gemini_enabled",
                   "prefix": "gemini", "name": "Google Gemini",
                   "base_field": None, "default_model": "gemini-2.5-flash", "default_base": None},
    "huggingface": {"key": "hf_api_key", "model": "hf_model", "enabled": "hf_enabled",
                    "prefix": "huggingface", "name": "HuggingFace",
                    "base_field": None, "default_model": "meta-llama/Llama-3.1-70B-Instruct",
                    "default_base": None},
    "github":     {"key": "github_token", "model": "github_model", "enabled": "github_enabled",
                   "prefix": "openai", "name": "GitHub Models",
                   "base_field": None, "default_model": "gpt-4o",
                   "default_base": "https://models.inference.ai.azure.com"},
    "custom":     {"key": "custom_api_key", "model": "custom_model", "enabled": "custom_enabled",
                   "prefix": "openai", "name": "Custom (Ollama, LM Studio)",
                   "base_field": "custom_api_base", "default_model": "", "default_base": None},
}

_OLD_DEFAULT_ORDER = ["openrouter", "openai", "gemini", "huggingface", "github", "custom"]


# ── Persistence ───────────────────────────────────────────────────────────────

# Fields persisted to ~/.asure-flow/config.json via the UI
# NOTE: user-preference fields (feature toggles, AI preset, privacy prefs, diarization)
# live in profile.py / profile.json — not here.
_PERSISTABLE_FIELDS = frozenset({
    "providers",
    "whisper_model", "whisper_device",
    # Audio capture
    "audio_capture_source", "mic_device_id", "system_device_id",
    # VAD flush
    "vad_silence_ms", "vad_min_buffer_sec", "vad_max_buffer_sec",
    # Diarization hardware
    "hf_diarization_token", "diarization_device", "diarization_buffer_sec",
    # LLM routing
    "routing_strategy",
    # Admin locks
    "locked_settings",
})


# ── Helpers ───────────────────────────────────────────────────────────────────


def _find_env_file() -> str:
    """Resolve .env from the project root (parent of server/), regardless of CWD."""
    # Walk up from this file: config.py -> asure_flow -> src -> server -> PROJECT_ROOT
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        return str(env_path)
    # Fallback: check CWD
    if Path(".env").exists():
        return ".env"
    return str(env_path)  # still pass it so pydantic doesn't error


def _default_session_dir() -> str:
    return str(Path.home() / ".asure-flow" / "sessions")


# ── Settings ──────────────────────────────────────────────────────────────────


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Server ──
    host: str = "0.0.0.0"
    port: int = 8000

    # ── Transcription ──
    whisper_model: str = "large-v3-turbo"
    whisper_device: Optional[str] = None  # auto-detect
    whisper_compute_type: Optional[str] = None  # auto-select
    whisper_language: Optional[str] = None  # auto-detect

    # ── LLM Providers (data-driven list, persisted in config.json) ──
    providers: list[ProviderEntry] = Field(default_factory=list)

    # ── Bootstrap env vars (seed provider API keys from .env) ──────────────
    # Parsed by pydantic-settings from .env / environment. NOT persisted.
    # Used to (a) seed providers on first run and (b) overlay API keys on every run.
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    custom_api_key: Optional[str] = None
    custom_api_base: Optional[str] = None
    # Legacy env vars (only used for migration from old .env / config.json)
    hf_api_key: Optional[str] = None
    github_token: Optional[str] = None
    openrouter_model: Optional[str] = None
    openai_model: Optional[str] = None
    gemini_model: Optional[str] = None
    hf_model: Optional[str] = None
    github_model: Optional[str] = None
    custom_model: Optional[str] = None

    # ── LLM Routing ──
    routing_strategy: str = "simple-shuffle"
    # Options: "simple-shuffle" (ordered fallback), "latency-based-routing", "usage-based-routing"

    # ── Audio Capture ──
    audio_capture_source: str = "client"  # "client" | "server"
    mic_device_id: Optional[str] = None
    system_device_id: Optional[str] = None

    # ── VAD Flush ──
    vad_silence_ms: int = 600
    vad_min_buffer_sec: float = 1.5
    vad_max_buffer_sec: float = 30.0

    # ── Diarization hardware (secrets + device — user preference is in profile.py) ──
    hf_diarization_token: Optional[str] = None
    diarization_device: Optional[str] = None  # "cpu" | "cuda" | None (auto)
    diarization_buffer_sec: float = 20.0

    # ── Admin ──
    locked_settings: list[str] = Field(default_factory=list)

    # ── Sessions ──
    session_dir: str = Field(default_factory=_default_session_dir)

    # ── Derived helpers ──

    def detect_device(self) -> str:
        if self.whisper_device:
            return self.whisper_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def detect_compute_type(self) -> str:
        if self.whisper_compute_type:
            return self.whisper_compute_type
        return "float16" if self.detect_device() == "cuda" else "int8"

    def to_client_config(self) -> dict:
        """Return config safe to expose to clients (no full API keys)."""

        def _mask(key: str | None) -> str:
            if not key:
                return ""
            if len(key) <= 8:
                return "****"
            return key[:4] + "****" + key[-4:]

        return {
            "server_platform": sys.platform,
            "hostname": platform.node(),
            "whisper_model": self.whisper_model,
            "whisper_device": self.detect_device(),
            "whisper_compute_type": self.detect_compute_type(),
            "whisper_language": self.whisper_language,
            "routing_strategy": self.routing_strategy,
            # Audio capture
            "audio_capture_source": self.audio_capture_source,
            "mic_device_id": self.mic_device_id,
            "system_device_id": self.system_device_id,
            # Diarization hardware
            "diarization_device": self.diarization_device,
            "hf_diarization_token_hint": _mask(self.hf_diarization_token),
            "locked_settings": self.locked_settings,
            # Providers (ordered array — position = priority)
            "llm_providers": [
                {
                    "id": p.id,
                    "name": p.name,
                    "litellm_prefix": p.litellm_prefix,
                    "model": p.model,
                    "api_base": p.api_base or "",
                    "api_key_hint": _mask(p.api_key),
                    "configured": bool(p.api_key or p.api_base),
                    "enabled": p.enabled,
                }
                for p in self.providers
            ],
        }


settings = Settings()


# ── Config path ───────────────────────────────────────────────────────────────


def _config_path() -> Path:
    """Path to the persisted config file."""
    return Path(settings.session_dir).parent / "config.json"


# ── Migration (old per-provider fields → new providers list) ──────────────────


def _migrate_old_providers(data: dict) -> list[dict] | None:
    """Convert old per-provider config.json fields to new providers list.

    Returns the migrated list, or None if no migration needed.
    """
    if "providers" in data:
        return None  # already migrated

    has_old = any(
        meta["key"] in data or meta["model"] in data or meta["enabled"] in data
        for meta in _OLD_PROVIDER_MIGRATION.values()
    )
    if not has_old:
        return None

    order = data.get("provider_order", _OLD_DEFAULT_ORDER)
    providers: list[dict] = []
    for pid in order:
        meta = _OLD_PROVIDER_MIGRATION.get(pid)
        if not meta:
            continue
        entry = {
            "id": pid,
            "name": meta["name"],
            "litellm_prefix": meta["prefix"],
            "model": data.get(meta["model"], meta["default_model"]),
            "api_key": data.get(meta["key"]),
            "api_base": (data.get(meta["base_field"]) if meta["base_field"] else None) or meta["default_base"],
            "enabled": data.get(meta["enabled"], True),
        }
        providers.append(entry)

    return providers if providers else None


# ── Load / save / seed ────────────────────────────────────────────────────────


def _load_persisted() -> None:
    """Load persisted overrides from config.json on top of current settings."""
    path = _config_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))

        # Migrate old provider format if needed
        migrated = _migrate_old_providers(data)
        if migrated is not None:
            data["providers"] = migrated
            # Clean up old fields so they don't linger
            for meta in _OLD_PROVIDER_MIGRATION.values():
                for f in [meta["key"], meta["model"], meta["enabled"]]:
                    data.pop(f, None)
                if meta["base_field"]:
                    data.pop(meta["base_field"], None)
            data.pop("provider_order", None)
            logger.info("Migrated old provider config to new format")

        for key, value in data.items():
            if key not in _PERSISTABLE_FIELDS or not hasattr(settings, key):
                continue
            if key == "providers":
                settings.providers = [ProviderEntry(**p) for p in value]
            else:
                setattr(settings, key, value)

        logger.info("Loaded persisted config from %s", path)
    except Exception:
        logger.warning("Failed to load persisted config from %s", path, exc_info=True)


def _seed_providers() -> None:
    """Seed providers from defaults and overlay env-var API keys."""
    if not settings.providers:
        settings.providers = [p.model_copy() for p in _DEFAULT_PROVIDERS]

    # Overlay env-var API keys onto matching providers
    env_keys: dict[str, str | None] = {
        "openrouter": settings.openrouter_api_key,
        "openai": settings.openai_api_key,
        "anthropic": settings.anthropic_api_key,
        "gemini": settings.gemini_api_key,
        "groq": settings.groq_api_key,
        "custom": settings.custom_api_key,
        # Legacy
        "huggingface": settings.hf_api_key,
        "github": settings.github_token,
    }
    env_bases: dict[str, str | None] = {
        "custom": settings.custom_api_base,
    }

    for provider in settings.providers:
        env_key = env_keys.get(provider.id)
        if env_key and not provider.api_key:
            provider.api_key = env_key
        env_base = env_bases.get(provider.id)
        if env_base and not provider.api_base:
            provider.api_base = env_base


def _save_persisted() -> None:
    """Save current persistable fields to config.json."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {}
    for key in _PERSISTABLE_FIELDS:
        value = getattr(settings, key, None)
        if value is not None:
            if key == "providers":
                data[key] = [p.model_dump() for p in value]
            else:
                data[key] = value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── Runtime mutation helpers ──────────────────────────────────────────────────


def update_settings(**kwargs: object) -> None:
    """Update individual settings fields at runtime and persist to disk."""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    _save_persisted()


def update_provider(provider_id: str, **kwargs: object) -> ProviderEntry | None:
    """Update a single provider by ID (merge-style). Returns the updated entry or None."""
    for p in settings.providers:
        if p.id == provider_id:
            for key, value in kwargs.items():
                if hasattr(p, key) and key != "id":
                    setattr(p, key, value)
            _save_persisted()
            return p
    return None


def add_provider(entry: ProviderEntry) -> None:
    """Add a new provider entry. Raises ValueError on duplicate ID."""
    if any(p.id == entry.id for p in settings.providers):
        raise ValueError(f"Provider with id '{entry.id}' already exists")
    settings.providers.append(entry)
    _save_persisted()


def remove_provider(provider_id: str) -> bool:
    """Remove a provider by ID. Returns True if found and removed."""
    for i, p in enumerate(settings.providers):
        if p.id == provider_id:
            settings.providers.pop(i)
            _save_persisted()
            return True
    return False


def reorder_providers(order: list[str]) -> None:
    """Reorder providers by ID list. IDs not in the list are appended at the end."""
    by_id = {p.id: p for p in settings.providers}
    new_list: list[ProviderEntry] = []
    for pid in order:
        if pid in by_id:
            new_list.append(by_id.pop(pid))
    new_list.extend(by_id.values())
    settings.providers = new_list
    _save_persisted()


def reset_settings() -> None:
    """Reset all persistable fields to their defaults and delete config.json."""
    defaults = Settings()
    for key in _PERSISTABLE_FIELDS:
        if key == "providers":
            settings.providers = [p.model_copy() for p in _DEFAULT_PROVIDERS]
        else:
            setattr(settings, key, getattr(defaults, key))
    path = _config_path()
    if path.exists():
        path.unlink()
    logger.info("Settings reset to defaults")


# ── Initialize on import ─────────────────────────────────────────────────────

_load_persisted()
_seed_providers()
