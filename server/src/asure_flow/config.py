from __future__ import annotations

import json
import logging
import platform
import sys
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER_ORDER = [
    "openrouter", "openai", "gemini", "huggingface", "github", "custom",
]

# Fields that are persisted to ~/.asure-flow/config.json via the UI
# NOTE: user-preference fields (feature toggles, AI preset, privacy prefs, diarization)
# live in profile.py / profile.json — not here.
_PERSISTABLE_FIELDS = frozenset({
    "openrouter_api_key", "openrouter_model",
    "openai_api_key", "openai_model",
    "gemini_api_key", "gemini_model",
    "hf_api_key", "hf_model",
    "github_token", "github_model",
    "custom_api_key", "custom_api_base", "custom_model",
    "whisper_model", "whisper_device",
    # LLM provider toggles + order
    "openrouter_enabled", "openai_enabled", "gemini_enabled",
    "hf_enabled", "github_enabled", "custom_enabled",
    "provider_order",
    # Audio capture (server-mode device IDs — only used when audio_capture_source="server")
    "audio_capture_source", "mic_device_id", "system_device_id",
    # VAD flush
    "vad_silence_ms", "vad_min_buffer_sec", "vad_max_buffer_sec",
    # Diarization hardware (secrets + hardware — preferences are in profile.json)
    "hf_diarization_token", "diarization_device", "diarization_buffer_sec",
    # Admin locks
    "locked_settings",
})


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

    # ── LLM Providers ──
    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-sonnet-4-20250514"

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4.1"

    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.5-flash"

    hf_api_key: Optional[str] = None
    hf_model: str = "meta-llama/Llama-3.1-70B-Instruct"

    github_token: Optional[str] = None
    github_model: str = "gpt-4o"

    custom_api_key: Optional[str] = None
    custom_api_base: Optional[str] = None
    custom_model: Optional[str] = None

    # ── LLM Provider Toggles ──
    openrouter_enabled: bool = True
    openai_enabled: bool = True
    gemini_enabled: bool = True
    hf_enabled: bool = True
    github_enabled: bool = True
    custom_enabled: bool = True

    # ── LLM Provider Priority ──
    provider_order: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_PROVIDER_ORDER),
    )

    # ── Audio Capture ──
    # audio_capture_source: deployment decision — "client" streams PCM to server,
    # "server" captures audio locally on the server machine.
    # mic_device_id / system_device_id: only relevant when audio_capture_source="server".
    audio_capture_source: str = "client"  # "client" | "server"
    mic_device_id: Optional[str] = None
    system_device_id: Optional[str] = None

    # ── VAD Flush ──
    vad_silence_ms: int = 600  # ms of trailing silence after speech to trigger flush
    vad_min_buffer_sec: float = 1.5  # don't flush until at least this much audio
    vad_max_buffer_sec: float = 30.0  # hard max — force flush even mid-speech

    # ── Diarization hardware (secrets + device — user preference is in profile.py) ──
    hf_diarization_token: Optional[str] = None
    diarization_device: Optional[str] = None  # "cpu" | "cuda" | None (auto)
    diarization_buffer_sec: float = 20.0  # rolling window size for speaker tracking

    # ── Admin ──
    locked_settings: list[str] = Field(default_factory=list)
    # Fields in this list cannot be updated via PUT /api/config from a client.
    # Example: ["whisper_model", "provider_order", "openrouter_enabled"]

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
            "provider_order": self.provider_order,
            # Audio capture mode + server-side device IDs (only relevant when audio_capture_source="server")
            "audio_capture_source": self.audio_capture_source,
            "mic_device_id": self.mic_device_id,
            "system_device_id": self.system_device_id,
            # Diarization hardware info (device preference is in profile, not here)
            "diarization_device": self.diarization_device,
            "hf_diarization_token_hint": _mask(self.hf_diarization_token),
            "locked_settings": self.locked_settings,
            "llm_providers": {
                "openrouter": {
                    "configured": bool(self.openrouter_api_key),
                    "enabled": self.openrouter_enabled,
                    "model": self.openrouter_model,
                    "api_key_hint": _mask(self.openrouter_api_key),
                },
                "openai": {
                    "configured": bool(self.openai_api_key),
                    "enabled": self.openai_enabled,
                    "model": self.openai_model,
                    "api_key_hint": _mask(self.openai_api_key),
                },
                "gemini": {
                    "configured": bool(self.gemini_api_key),
                    "enabled": self.gemini_enabled,
                    "model": self.gemini_model,
                    "api_key_hint": _mask(self.gemini_api_key),
                },
                "huggingface": {
                    "configured": bool(self.hf_api_key),
                    "enabled": self.hf_enabled,
                    "model": self.hf_model,
                    "api_key_hint": _mask(self.hf_api_key),
                },
                "github": {
                    "configured": bool(self.github_token),
                    "enabled": self.github_enabled,
                    "model": self.github_model,
                    "api_key_hint": _mask(self.github_token),
                },
                "custom": {
                    "configured": bool(self.custom_api_base and self.custom_model),
                    "enabled": self.custom_enabled,
                    "model": self.custom_model or "",
                    "api_base": self.custom_api_base or "",
                    "api_key_hint": _mask(self.custom_api_key),
                },
            },
        }


settings = Settings()


def _config_path() -> Path:
    """Path to the persisted config file."""
    return Path(settings.session_dir).parent / "config.json"


def _load_persisted() -> None:
    """Load persisted overrides from config.json on top of current settings."""
    path = _config_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if key in _PERSISTABLE_FIELDS and hasattr(settings, key):
                setattr(settings, key, value)
        logger.info("Loaded persisted config from %s", path)
    except Exception:
        logger.warning("Failed to load persisted config from %s", path, exc_info=True)


def _save_persisted() -> None:
    """Save current persistable fields to config.json."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {}
    for key in _PERSISTABLE_FIELDS:
        value = getattr(settings, key, None)
        if value is not None:
            data[key] = value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_settings(**kwargs: object) -> None:
    """Update individual settings fields at runtime and persist to disk."""
    for key, value in kwargs.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
    _save_persisted()


def reset_settings() -> None:
    """Reset all persistable fields to their defaults and delete config.json."""
    defaults = Settings()
    for key in _PERSISTABLE_FIELDS:
        setattr(settings, key, getattr(defaults, key))
    path = _config_path()
    if path.exists():
        path.unlink()
    logger.info("Settings reset to defaults")


# Apply persisted overrides on import
_load_persisted()
