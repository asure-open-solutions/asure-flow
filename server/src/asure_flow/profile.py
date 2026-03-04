"""User profile — portable preferences that follow the user across client machines.

Tier 2 of the 3-tier settings model:
  Tier 1: Server-admin  (config.py)  — hardware, secrets, locked_settings
  Tier 2: User profile  (profile.py) — feature toggles, AI preset, privacy prefs
  Tier 3: Device-local  (client localStorage) — serverUrl, audio devices, overlay layout
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_PROFILE_FIELDS = frozenset({
    "fact_checking",
    "suggestions",
    "notes",
    "search_transcript",
    "search_sessions",
    "web_search",
    "format_code",
    "deep_think",
    "agent_mode",
    "parallel_tools",
    "ai_preset",
    "custom_system_prompt",
    "diarization_enabled",
    "pii_redaction",
    "privacy_mode",
})


class UserProfile:
    """User-portable preferences — follow the user across client machines.

    Stored in ~/.asure-flow/profile.json.  All fields are user preferences,
    not hardware or admin configuration.  Clients read/write via GET/PUT /api/profile.
    """

    def __init__(self) -> None:
        # Feature toggles
        self.fact_checking: bool = True
        self.suggestions: bool = True
        self.notes: bool = True
        self.search_transcript: bool = True
        self.search_sessions: bool = False
        self.web_search: bool = True
        self.format_code: bool = False
        self.deep_think: str = "off"  # "off" | "auto" | "always"

        # Agent execution mode
        self.agent_mode: str = "unified"  # "unified" | "specialists"
        self.parallel_tools: bool = False  # parallel tool execution in unified mode

        # AI behaviour
        self.ai_preset: str = "general"
        self.custom_system_prompt: Optional[str] = None

        # Diarization preference (not hardware config — that stays in admin settings)
        self.diarization_enabled: bool = False

        # Safety
        self.pii_redaction: bool = False
        self.privacy_mode: bool = False  # disables web_search + enables pii_redaction

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in _PROFILE_FIELDS}


profile = UserProfile()


def _profile_path() -> Path:
    """Path to the persisted profile file."""
    return Path.home() / ".asure-flow" / "profile.json"


def _load_persisted_profile() -> None:
    """Load persisted profile overrides from profile.json."""
    path = _profile_path()
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if key in _PROFILE_FIELDS and hasattr(profile, key):
                setattr(profile, key, value)
        logger.info("Loaded persisted profile from %s", path)
    except Exception:
        logger.warning("Failed to load persisted profile from %s", path, exc_info=True)


def _save_persisted_profile() -> None:
    """Save current profile fields to profile.json."""
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, object] = {}
    for key in _PROFILE_FIELDS:
        value = getattr(profile, key, None)
        if value is not None:
            data[key] = value
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def update_profile(**kwargs: object) -> None:
    """Update individual profile fields at runtime and persist to disk."""
    for key, value in kwargs.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
    _save_persisted_profile()


def reset_profile() -> None:
    """Reset all profile fields to their defaults and delete profile.json."""
    defaults = UserProfile()
    for key in _PROFILE_FIELDS:
        setattr(profile, key, getattr(defaults, key))
    path = _profile_path()
    if path.exists():
        path.unlink()
    logger.info("Profile reset to defaults")


# Apply persisted overrides on import
_load_persisted_profile()
