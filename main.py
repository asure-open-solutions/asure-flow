import asyncio
import threading
import uvicorn
import webbrowser
import json
import os
import socket
import sys
import logging
import faulthandler
import uuid
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
import time
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse, FileResponse, Response, RedirectResponse
from contextlib import asynccontextmanager, suppress
from datetime import datetime

from backend.audio import AudioManager
from backend.transcription import TranscriptionEngine, TranscriptionChunk
from backend.diarization import SpeakerDiarizer
from backend.llm import LLMClient
from backend.speech_processing import SpeechPreprocessConfig
from starlette.websockets import WebSocketDisconnect

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Main")
important_logger = logging.getLogger("Main.IMPORTANT")

_IMPORTANT_LAST_BY_KEY: dict[str, float] = {}
_NOISY_LOGGERS = (
    "backend.transcription",
    "backend.audio",
    "faster_whisper",
    "uvicorn.access",
)


def _safe_log_value(value: Any, *, max_len: int = 96) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value).strip()
    if not s:
        return "-"
    s = " ".join(s.split())
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def log_important(
    event: str,
    *,
    level: int = logging.INFO,
    dedupe_key: str | None = None,
    dedupe_window_s: float = 0.0,
    **fields: Any,
) -> None:
    try:
        ev = _safe_log_value(event, max_len=64)
        if dedupe_key and dedupe_window_s > 0:
            token = f"{ev}|{dedupe_key}"
            now_ts = time.time()
            prev_ts = _IMPORTANT_LAST_BY_KEY.get(token, 0.0)
            if now_ts - prev_ts < float(dedupe_window_s):
                return
            _IMPORTANT_LAST_BY_KEY[token] = now_ts

        if fields:
            parts = [f"{k}={_safe_log_value(v)}" for k, v in sorted(fields.items())]
            important_logger.log(level, f"IMPORTANT {ev} | " + " ".join(parts))
        else:
            important_logger.log(level, f"IMPORTANT {ev}")
    except Exception:
        logger.exception("Failed to emit important log")

# Best-effort: dump tracebacks on native crashes (segfault/abort).
with suppress(Exception):
    faulthandler.enable(all_threads=True)

# Global State
audio_manager = AudioManager()
transcription_model = None  # Lazy load (shared for mic + loopback)
llm_client = None
third_party_diarizer: SpeakerDiarizer | None = None

# We'll use a queue to send transcriptions to the websocket loop
transcription_queue = asyncio.Queue()

_TRANSCRIPT_LAST_UPDATE_TS: dict[str, float] = {}
_TRANSCRIPT_CLEANUP_LAST_TS: float = 0.0
_TRANSCRIPT_ACTIVITY_SEQ: int = 0
_AI_TRIGGER_BASELINE_TEXT: dict[str, str] = {}
_PYTEXTRANK_NLP: Any = None
_PYTEXTRANK_INIT_ATTEMPTED: bool = False


# ============================================
# DATA MODELS
# ============================================

@dataclass
class NoteItem:
    id: str
    content: str
    timestamp: str
    source: str = "ai"  # "ai" or "manual"
    category: str = "general"  # "decision", "action", "risk", "fact", "general"
    pinned: bool = False
    completed: bool = False
    session_id: Optional[str] = None


@dataclass
class ResponseItem:
    id: str
    content: str
    timestamp: str
    responding_to_text: str = ""
    responding_to_source: str = ""
    responding_to_speaker: str = ""
    responding_to_timestamp: str = ""
    basis_facts: List[str] = field(default_factory=list)
    cautions: List[str] = field(default_factory=list)
    confidence: str = ""
    source: str = "ai"
    session_id: Optional[str] = None


@dataclass
class FactCheckItem:
    id: str
    claim: str
    verdict: str
    analysis: str
    timestamp: str
    evidence: List[str] = field(default_factory=list)
    confidence: str = ""
    session_id: Optional[str] = None


@dataclass
class TranscriptMessage:
    id: str
    text: str
    source: str  # "user" (You), "third_party"/"loopback" (Third-Party)
    timestamp: str
    clean_text: Optional[str] = None
    speaker_id: Optional[str] = None
    speaker_label: Optional[str] = None


@dataclass 
class Session:
    id: str
    started_at: str
    ended_at: Optional[str] = None
    title: str = "Untitled Session"
    context: str = ""
    transcript: List[TranscriptMessage] = field(default_factory=list)
    notes: List[NoteItem] = field(default_factory=list)
    responses: List[ResponseItem] = field(default_factory=list)
    fact_checks: List[FactCheckItem] = field(default_factory=list)
    speaker_names: Dict[str, str] = field(default_factory=dict)  # speaker_id -> friendly name
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "title": self.title,
            "context": self.context,
            "transcript": [asdict(m) for m in self.transcript],
            "notes": [asdict(n) for n in self.notes],
            "responses": [asdict(r) for r in self.responses],
            "fact_checks": [asdict(f) for f in self.fact_checks],
            "speaker_names": dict(self.speaker_names or {}),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        responses: list[ResponseItem] = []
        for r in data.get("responses", []) or []:
            if not isinstance(r, dict):
                continue
            try:
                responses.append(ResponseItem(**r))
            except Exception:
                continue

        fact_checks: list[FactCheckItem] = []
        for f in data.get("fact_checks", []) or []:
            if not isinstance(f, dict):
                continue
            try:
                fact_checks.append(FactCheckItem(**f))
            except Exception:
                continue

        return cls(
            id=data["id"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            title=data.get("title", "Untitled Session"),
            context=data.get("context", "") or "",
            transcript=[TranscriptMessage(**m) for m in data.get("transcript", [])],
            notes=[NoteItem(**n) for n in data.get("notes", [])],
            responses=responses,
            fact_checks=fact_checks,
            speaker_names=dict(data.get("speaker_names") or {}),
        )


# Current session (in-memory)
current_session: Optional[Session] = None


# ============================================
# CONFIGURATION
# ============================================

DEFAULT_CONFIG = {
    # API
    "api_provider": "openrouter",  # "openrouter", "openai", "canopywave", "custom"
    "api_key": "",
    "base_url": "https://openrouter.ai/api/v1",
    "model": "openai/gpt-4o-mini",
    "api_extra_headers": {},  # Optional extra headers passed to the OpenAI-compatible client.
    "api_fallback_enabled": True,
    # Ordered fallback routes (each entry mirrors primary API fields).
    # Example:
    # [{"provider":"openai","api_key":"...","base_url":"https://api.openai.com/v1","model":"gpt-4o-mini","api_extra_headers":{}}]
    "api_routes": [],
    
    # Devices
    "mic_device": None,
    "mic_enabled": False,
    "loopback_device": None,
    "loopback_enabled": False,
    
    # Assistant
    "system_prompt": "You are a helpful AI assistant. Summarize the user's input and reply concisely. Focus on being practical and actionable.",
    "ai_enabled": False,
    "ai_min_interval_seconds": 8.0,
    "auto_respond": False,
    "response_enabled": True,
    "response_context_messages": 14,
    "response_context_max_chars": 36000,
    "response_max_items": 20,
    "response_require_policy_gate": True,
    "response_policy_min_confidence": 0.58,
    # While live STT is still updating text, hold AI calls and coalesce to latest.
    "ai_transcript_settle_seconds": 1.8,
    "ai_transcript_settle_max_wait_seconds": 18.0,
    "response_prompt": (
        "Generate a strategic, fact-based response the user can say in an argument or legal-style dispute. "
        "You are always aligned with the user; never argue against, scold, or fact-check the user directly. "
        "If the latest message is from the user, strengthen or reframe it into a safer defense instead of countering it. "
        "Prefer responding to counterparty claims when available. "
        "Use only information grounded in the transcript/context. Be concise, calm, and defensible. "
        "Do not invent facts."
    ),
    "web_search_enabled": False,
    "web_search_mode": "auto",  # "auto" or "always" (only used when web_search_enabled is True)
    "web_search_max_results": 5,
    "web_search_timeout_seconds": 6.0,
    "web_search_cache_ttl_seconds": 180.0,
    "web_search_min_interval_seconds": 6.0,
    # Local/free context compression for long sessions.
    "context_local_summary_enabled": True,
    "context_local_summary_method": "pytextrank",  # "pytextrank", "heuristic", "auto"
    
    # Policy
    "policy_enabled": True,
    "policy_prompt": "Only respond when it is clearly helpful or strategically appropriate. If the user is in an argument, respond only at a suitable time with a concise, de-escalatory defense based on what was said. If it's not the right moment, do not respond.",
    "policy_show_withheld": True,
    "policy_min_interval_seconds": 4.0,
    
    # Notes - NEW SEPARATE PROCESS
    "notes_enabled": True,
    "notes_interval_seconds": 30,
    "notes_on_interaction_only": False,
    "notes_format": "bullets",  # "bullets", "structured", "summary", "custom"
    "notes_prompt": "Analyze the conversation and extract key information. Focus on: decisions made, action items, important facts, and potential risks or concerns.",
    "notes_context_messages": 10,
    "notes_max_chars": 24000,
    "notes_live_on_message": True,
    "notes_debounce_seconds": 2.5,
    "notes_trigger_min_interval_seconds": 10.0,
    # Smart notes maintenance: keeps the notes list concise over time by merging/pruning.
    "notes_smart_enabled": True,
    "notes_smart_target_ai_notes": 12,
    "notes_smart_max_ai_notes": 18,
    "notes_extract_decisions": True,
    "notes_extract_actions": True,
    "notes_extract_risks": True,
    "notes_extract_facts": True,

    # Fact Check
    "fact_check_enabled": True,
    "fact_check_interval_seconds": 15,
    "fact_check_on_interaction_only": False,
    "fact_check_live_on_message": True,
    "fact_check_debounce_seconds": 1.0,
    "fact_check_trigger_min_interval_seconds": 4.0,
    "fact_check_context_messages": 18,
    "fact_check_context_max_chars": 42000,
    "fact_check_max_items": 12,
    "fact_check_prompt": (
        "Identify key claims in the conversation and classify each as supported, contradicted, or uncertain "
        "based only on known context. Do not fabricate evidence."
    ),

    # Speech preprocessing (voice-only transcription).
    "speech_preprocess_enabled": True,
    "speech_vad_enabled": True,
    "speech_vad_threshold": 0.5,
    "speech_vad_neg_threshold": 0.35,
    "speech_vad_min_speech_ms": 200,
    "speech_vad_max_speech_s": 30.0,
    # Increase silence threshold to reduce mid-sentence segmentation.
    "speech_vad_min_silence_ms": 650,
    "speech_vad_speech_pad_ms": 120,
    # Keep a bit more natural pause between merged segments.
    "speech_vad_concat_silence_ms": 220,
    "speech_denoise_enabled": False,
    "speech_denoise_strength": 0.8,
    "speech_denoise_floor": 0.06,
    "whisper_vad_filter": True,
    "whisper_beam_size": 1,
    "transcription_chunk_duration_seconds": 3.2,
    "transcription_chunk_overlap_seconds": 0.16,
    # "auto" chooses profile based on whisper_device.
    # "manual" uses whisper_beam_size/chunk_duration/chunk_overlap as provided.
    # cpu_realtime|cpu_accuracy|gpu_realtime|gpu_accuracy apply tuned defaults.
    "transcription_profile": "auto",
    "whisper_model_size": "tiny",
    "whisper_device": "cpu",  # "cpu" or "cuda"
    "speaker_diarization_enabled": False,
    
    # Session management
    "autosave_enabled": True,
    "session_timeout_minutes": 30,
    "verbose_logging": False,

    # Transcript quality
    "transcript_merge_enabled": True,
    "transcript_merge_window_seconds": 4.0,
    # Allows merging strong same-speaker continuations across slower chunk arrivals.
    "transcript_merge_continuation_window_seconds": 18.0,
    # "off" = disabled, "cleanup" = punctuation/fluency only, "paraphrase" = rewrite into what speaker likely meant.
    "transcript_ai_mode": "off",
    "transcript_ai_cleanup_debounce_seconds": 1.25,
    "transcript_ai_cleanup_min_interval_seconds": 3.0,
    "transcript_display_mode": "raw",  # "raw" or "clean"
}


@dataclass
class ConnectionState:
    last_policy_eval_ts: float = 0.0
    last_response_ts: float = 0.0
    last_fact_check_ts: float = 0.0
    llm_not_configured_warned: bool = False


def _get_config_path() -> Path:
    configured = os.environ.get("AI_ASSISTANT_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser().resolve()

    base_dir = os.environ.get("APPDATA") or str(Path.home())
    config_dir = Path(base_dir) / "AI Assistant"
    return (config_dir / "settings.json").resolve()


def _get_data_dir() -> Path:
    base_dir = os.environ.get("APPDATA") or str(Path.home())
    data_dir = Path(base_dir) / "AI Assistant" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


_CONFIG_PATH = _get_config_path()
_DATA_DIR = _get_data_dir()


def _get_presets_path() -> Path:
    # Store presets alongside settings.json (under APPDATA/AI Assistant).
    return _CONFIG_PATH.with_name("presets.json")


def _load_presets() -> dict:
    path = _get_presets_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Failed to load presets file")
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save_presets(presets: dict) -> None:
    path = _get_presets_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(presets, indent=2), encoding="utf-8")
        tmp_path.replace(path)
    except Exception:
        logger.exception("Failed to save presets file")
        raise


def _sanitize_preset_name(name: str) -> str:
    n = (name or "").strip()
    n = " ".join(n.split())
    if not n:
        return ""
    if len(n) > 48:
        n = n[:48].strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_ ()[]")
    n2 = "".join(ch for ch in n if ch in allowed)
    n2 = " ".join(n2.split()).strip()
    return n2


_API_PROVIDER_PRESETS: dict[str, dict[str, object]] = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_env": "OPENAI_API_KEY",
    },
    # https://canopywave.com docs: https://inference.canopywave.io/v1 + CANOPYWAVE_API_KEY
    "canopywave": {
        "base_url": "https://inference.canopywave.io/v1",
        "model": "deepseek/deepseek-chat-v3.1",
        "api_key_env": "CANOPYWAVE_API_KEY",
    },
    # HF Inference Providers router (OpenAI-compatible): https://router.huggingface.co/v1
    # Model name typically includes provider suffix: "<repo_id>:<provider>".
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "HuggingFaceTB/SmolLM2-1.7B-Instruct:groq",
        "api_key_envs": ["HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"],
    },
    # Gemini via OpenAI-compatible endpoint.
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    # GitHub Models OpenAI-compatible endpoint.
    "github_models": {
        "base_url": "https://models.github.ai/inference",
        "model": "openai/gpt-4.1",
        "api_key_envs": ["GITHUB_TOKEN", "GH_TOKEN"],
        "api_extra_headers": {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    },
    "custom": {},
}


def _normalize_api_provider(provider: str | None) -> str:
    p = (provider or "").strip().casefold()
    if not p:
        return "custom"
    p = p.replace("-", "_").replace(" ", "_")
    if p in ("canopy", "canopy_wave"):
        p = "canopywave"
    if p in ("hf", "hugging_face"):
        p = "huggingface"
    if p in ("google", "google_ai", "google_gemini"):
        p = "gemini"
    if p in ("github", "github_model", "githubmodels", "gh_models", "gh_model"):
        p = "github_models"
    if p in ("open_router",):
        p = "openrouter"
    if p not in _API_PROVIDER_PRESETS:
        return "custom"
    return p


def _infer_provider_from_base_url(base_url: str | None) -> str:
    u = (base_url or "").strip().casefold()
    if not u:
        return "custom"
    if "openrouter.ai" in u:
        return "openrouter"
    if "api.openai.com" in u:
        return "openai"
    if "inference.canopywave.io" in u:
        return "canopywave"
    if "router.huggingface.co" in u or "api-inference.huggingface.co" in u:
        return "huggingface"
    if "generativelanguage.googleapis.com" in u:
        return "gemini"
    if "models.github.ai" in u:
        return "github_models"
    return "custom"


def _coerce_headers(value: object) -> dict[str, str]:
    if isinstance(value, dict):
        out: dict[str, str] = {}
        for k, v in value.items():
            ks = str(k).strip()
            vs = str(v).strip()
            if ks and vs:
                out[ks] = vs
        return out
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        try:
            data = json.loads(s)
        except Exception:
            return {}
        return _coerce_headers(data)
    return {}


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("1", "true", "yes", "on", "y"):
            return True
        if s in ("0", "false", "no", "off", "n", ""):
            return False
    return default


def _coerce_str(value: object, default: str, *, strip: bool = True, max_len: int | None = None) -> str:
    if value is None:
        out = default
    elif isinstance(value, str):
        out = value
    else:
        out = str(value)
    if strip:
        out = out.strip()
    if max_len is not None and max_len >= 0:
        out = out[:max_len]
    return out


def _coerce_int_in_range(value: object, default: int, *, min_v: int | None = None, max_v: int | None = None) -> int:
    try:
        out = int(float(value))
    except Exception:
        out = int(default)
    if min_v is not None and out < min_v:
        out = min_v
    if max_v is not None and out > max_v:
        out = max_v
    return out


def _coerce_float_in_range(
    value: object,
    default: float,
    *,
    min_v: float | None = None,
    max_v: float | None = None,
) -> float:
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if min_v is not None and out < min_v:
        out = min_v
    if max_v is not None and out > max_v:
        out = max_v
    return out


def _coerce_choice(value: object, choices: set[str], default: str) -> str:
    s = _coerce_str(value, default).lower()
    return s if s in choices else default


def _coerce_device(value: object) -> str | None:
    if value is None:
        return None
    s = _coerce_str(value, "", strip=True, max_len=256)
    return s or None


def _sanitize_api_route_entry(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None

    base_url_raw = _coerce_str(value.get("base_url"), "", max_len=2048)
    provider = _normalize_api_provider(_coerce_str(value.get("provider"), ""))
    inferred_provider = _infer_provider_from_base_url(base_url_raw)
    if provider == "custom" and inferred_provider != "custom":
        provider = inferred_provider
    preset = _API_PROVIDER_PRESETS.get(provider, {})

    base_url = base_url_raw or str(preset.get("base_url") or "")
    model = _coerce_str(
        value.get("model"),
        str(preset.get("model") or DEFAULT_CONFIG["model"]),
        max_len=512,
    ) or str(DEFAULT_CONFIG["model"])
    api_key = _coerce_str(value.get("api_key"), "", max_len=4096)
    preset_headers = _coerce_headers(preset.get("api_extra_headers"))
    extra_headers = {**preset_headers, **_coerce_headers(value.get("api_extra_headers"))}
    enabled = _coerce_bool(value.get("enabled"), True)

    if not base_url:
        return None

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "api_extra_headers": extra_headers,
        "enabled": enabled,
    }


def _coerce_api_routes_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    out: list[dict[str, object]] = []
    for raw in value:
        item = _sanitize_api_route_entry(raw)
        if not item:
            continue
        out.append(item)
        if len(out) >= 8:
            break
    return out


def _sanitize_config_values(raw: dict | None, *, base: dict | None = None) -> dict:
    raw_dict = raw if isinstance(raw, dict) else {}
    src: dict[str, object] = {}
    if isinstance(base, dict):
        src.update(base)
    if raw_dict:
        src.update(raw_dict)

    # Back-compat migration: older boolean flag maps to cleanup mode when explicit mode is missing.
    if "transcript_ai_mode" not in raw_dict and _coerce_bool(raw_dict.get("transcript_ai_cleanup_enabled"), False):
        src["transcript_ai_mode"] = "cleanup"

    # Back-compat migration: if older config only had a custom system prompt, re-use it for response prompt.
    response_prompt_raw = _coerce_str(src.get("response_prompt"), "", max_len=12000)
    system_prompt_raw = _coerce_str(src.get("system_prompt"), DEFAULT_CONFIG["system_prompt"], max_len=12000)
    default_legacy = _coerce_str(DEFAULT_CONFIG.get("system_prompt"), "", max_len=12000)
    if "response_prompt" not in raw_dict:
        legacy_prompt = _coerce_str(raw_dict.get("system_prompt"), "", max_len=12000)
        if legacy_prompt and legacy_prompt != default_legacy:
            response_prompt_raw = legacy_prompt
    if not response_prompt_raw:
        if system_prompt_raw and system_prompt_raw != default_legacy:
            response_prompt_raw = system_prompt_raw
        else:
            response_prompt_raw = _coerce_str(DEFAULT_CONFIG["response_prompt"], "", max_len=12000)

    provider = _normalize_api_provider(_coerce_str(src.get("api_provider"), str(DEFAULT_CONFIG["api_provider"])))
    base_url_raw = _coerce_str(src.get("base_url"), str(DEFAULT_CONFIG["base_url"]), max_len=2048)
    inferred_provider = _infer_provider_from_base_url(base_url_raw)
    if provider == "custom" and inferred_provider != "custom":
        provider = inferred_provider

    out: dict[str, object] = dict(DEFAULT_CONFIG)

    out["api_provider"] = provider
    out["api_key"] = _coerce_str(src.get("api_key"), "", max_len=4096)
    out["base_url"] = base_url_raw or str(DEFAULT_CONFIG["base_url"])
    out["model"] = _coerce_str(src.get("model"), str(DEFAULT_CONFIG["model"]), max_len=512) or str(DEFAULT_CONFIG["model"])
    out["api_extra_headers"] = _coerce_headers(src.get("api_extra_headers"))
    out["api_fallback_enabled"] = _coerce_bool(
        src.get("api_fallback_enabled"),
        bool(DEFAULT_CONFIG["api_fallback_enabled"]),
    )
    out["api_routes"] = _coerce_api_routes_list(src.get("api_routes"))

    out["mic_device"] = _coerce_device(src.get("mic_device"))
    out["mic_enabled"] = _coerce_bool(src.get("mic_enabled"), bool(DEFAULT_CONFIG["mic_enabled"]))
    out["loopback_device"] = _coerce_device(src.get("loopback_device"))
    out["loopback_enabled"] = _coerce_bool(src.get("loopback_enabled"), bool(DEFAULT_CONFIG["loopback_enabled"]))

    out["system_prompt"] = system_prompt_raw or str(DEFAULT_CONFIG["system_prompt"])
    out["ai_enabled"] = _coerce_bool(src.get("ai_enabled"), bool(DEFAULT_CONFIG["ai_enabled"]))
    out["ai_min_interval_seconds"] = _coerce_float_in_range(src.get("ai_min_interval_seconds"), 8.0, min_v=0.0, max_v=120.0)
    out["auto_respond"] = _coerce_bool(src.get("auto_respond"), bool(DEFAULT_CONFIG["auto_respond"]))
    out["response_enabled"] = _coerce_bool(src.get("response_enabled"), bool(DEFAULT_CONFIG["response_enabled"]))
    # 0 means "use all available messages".
    out["response_context_messages"] = _coerce_int_in_range(src.get("response_context_messages"), 14, min_v=0, max_v=5000)
    out["response_context_max_chars"] = _coerce_int_in_range(
        src.get("response_context_max_chars"),
        int(DEFAULT_CONFIG["response_context_max_chars"]),
        min_v=4000,
        max_v=250000,
    )
    out["response_max_items"] = _coerce_int_in_range(src.get("response_max_items"), 20, min_v=1, max_v=100)
    out["response_require_policy_gate"] = _coerce_bool(
        src.get("response_require_policy_gate"),
        bool(DEFAULT_CONFIG["response_require_policy_gate"]),
    )
    out["response_policy_min_confidence"] = _coerce_float_in_range(
        src.get("response_policy_min_confidence"),
        0.58,
        min_v=0.0,
        max_v=1.0,
    )
    out["ai_transcript_settle_seconds"] = _coerce_float_in_range(
        src.get("ai_transcript_settle_seconds"),
        float(DEFAULT_CONFIG["ai_transcript_settle_seconds"]),
        min_v=0.0,
        max_v=20.0,
    )
    out["ai_transcript_settle_max_wait_seconds"] = _coerce_float_in_range(
        src.get("ai_transcript_settle_max_wait_seconds"),
        float(DEFAULT_CONFIG["ai_transcript_settle_max_wait_seconds"]),
        min_v=0.2,
        max_v=120.0,
    )
    out["response_prompt"] = response_prompt_raw
    out["web_search_enabled"] = _coerce_bool(src.get("web_search_enabled"), bool(DEFAULT_CONFIG["web_search_enabled"]))
    out["web_search_mode"] = _coerce_choice(
        src.get("web_search_mode"),
        {"auto", "always"},
        str(DEFAULT_CONFIG["web_search_mode"]),
    )
    out["web_search_max_results"] = _coerce_int_in_range(src.get("web_search_max_results"), 5, min_v=1, max_v=20)
    out["web_search_timeout_seconds"] = _coerce_float_in_range(src.get("web_search_timeout_seconds"), 6.0, min_v=1.0, max_v=30.0)
    out["web_search_cache_ttl_seconds"] = _coerce_float_in_range(
        src.get("web_search_cache_ttl_seconds"),
        180.0,
        min_v=15.0,
        max_v=3600.0,
    )
    out["web_search_min_interval_seconds"] = _coerce_float_in_range(
        src.get("web_search_min_interval_seconds"),
        6.0,
        min_v=0.0,
        max_v=300.0,
    )
    out["context_local_summary_enabled"] = _coerce_bool(
        src.get("context_local_summary_enabled"),
        bool(DEFAULT_CONFIG["context_local_summary_enabled"]),
    )
    out["context_local_summary_method"] = _coerce_choice(
        src.get("context_local_summary_method"),
        {"pytextrank", "heuristic", "auto"},
        str(DEFAULT_CONFIG["context_local_summary_method"]),
    )

    out["policy_enabled"] = _coerce_bool(src.get("policy_enabled"), bool(DEFAULT_CONFIG["policy_enabled"]))
    out["policy_prompt"] = _coerce_str(src.get("policy_prompt"), str(DEFAULT_CONFIG["policy_prompt"]), max_len=12000)
    out["policy_show_withheld"] = _coerce_bool(src.get("policy_show_withheld"), bool(DEFAULT_CONFIG["policy_show_withheld"]))
    out["policy_min_interval_seconds"] = _coerce_float_in_range(src.get("policy_min_interval_seconds"), 4.0, min_v=0.0, max_v=60.0)

    out["notes_enabled"] = _coerce_bool(src.get("notes_enabled"), bool(DEFAULT_CONFIG["notes_enabled"]))
    out["notes_interval_seconds"] = _coerce_int_in_range(src.get("notes_interval_seconds"), 30, min_v=5, max_v=600)
    out["notes_on_interaction_only"] = _coerce_bool(
        src.get("notes_on_interaction_only"),
        bool(DEFAULT_CONFIG["notes_on_interaction_only"]),
    )
    out["notes_format"] = _coerce_choice(
        src.get("notes_format"),
        {"bullets", "structured", "summary", "custom"},
        str(DEFAULT_CONFIG["notes_format"]),
    )
    out["notes_prompt"] = _coerce_str(src.get("notes_prompt"), str(DEFAULT_CONFIG["notes_prompt"]), max_len=12000)
    # 0 means "use all available messages".
    out["notes_context_messages"] = _coerce_int_in_range(src.get("notes_context_messages"), 10, min_v=0, max_v=5000)
    out["notes_max_chars"] = _coerce_int_in_range(src.get("notes_max_chars"), 24000, min_v=2000, max_v=200000)
    out["notes_live_on_message"] = _coerce_bool(src.get("notes_live_on_message"), bool(DEFAULT_CONFIG["notes_live_on_message"]))
    out["notes_debounce_seconds"] = _coerce_float_in_range(src.get("notes_debounce_seconds"), 2.5, min_v=0.2, max_v=10.0)
    out["notes_trigger_min_interval_seconds"] = _coerce_float_in_range(
        src.get("notes_trigger_min_interval_seconds"),
        10.0,
        min_v=0.0,
        max_v=300.0,
    )
    out["notes_smart_enabled"] = _coerce_bool(src.get("notes_smart_enabled"), bool(DEFAULT_CONFIG["notes_smart_enabled"]))
    out["notes_smart_target_ai_notes"] = _coerce_int_in_range(
        src.get("notes_smart_target_ai_notes"),
        12,
        min_v=2,
        max_v=100,
    )
    out["notes_smart_max_ai_notes"] = _coerce_int_in_range(
        src.get("notes_smart_max_ai_notes"),
        18,
        min_v=3,
        max_v=120,
    )
    if int(out["notes_smart_max_ai_notes"]) < int(out["notes_smart_target_ai_notes"]):
        out["notes_smart_max_ai_notes"] = int(out["notes_smart_target_ai_notes"])
    out["notes_extract_decisions"] = _coerce_bool(src.get("notes_extract_decisions"), bool(DEFAULT_CONFIG["notes_extract_decisions"]))
    out["notes_extract_actions"] = _coerce_bool(src.get("notes_extract_actions"), bool(DEFAULT_CONFIG["notes_extract_actions"]))
    out["notes_extract_risks"] = _coerce_bool(src.get("notes_extract_risks"), bool(DEFAULT_CONFIG["notes_extract_risks"]))
    out["notes_extract_facts"] = _coerce_bool(src.get("notes_extract_facts"), bool(DEFAULT_CONFIG["notes_extract_facts"]))

    out["fact_check_enabled"] = _coerce_bool(src.get("fact_check_enabled"), bool(DEFAULT_CONFIG["fact_check_enabled"]))
    out["fact_check_interval_seconds"] = _coerce_int_in_range(src.get("fact_check_interval_seconds"), 15, min_v=5, max_v=600)
    out["fact_check_on_interaction_only"] = _coerce_bool(
        src.get("fact_check_on_interaction_only"),
        bool(DEFAULT_CONFIG["fact_check_on_interaction_only"]),
    )
    out["fact_check_live_on_message"] = _coerce_bool(
        src.get("fact_check_live_on_message"),
        bool(DEFAULT_CONFIG["fact_check_live_on_message"]),
    )
    out["fact_check_debounce_seconds"] = _coerce_float_in_range(
        src.get("fact_check_debounce_seconds"),
        1.0,
        min_v=0.2,
        max_v=10.0,
    )
    out["fact_check_trigger_min_interval_seconds"] = _coerce_float_in_range(
        src.get("fact_check_trigger_min_interval_seconds"),
        4.0,
        min_v=0.0,
        max_v=300.0,
    )
    # 0 means "use all available messages".
    out["fact_check_context_messages"] = _coerce_int_in_range(src.get("fact_check_context_messages"), 18, min_v=0, max_v=5000)
    out["fact_check_context_max_chars"] = _coerce_int_in_range(
        src.get("fact_check_context_max_chars"),
        int(DEFAULT_CONFIG["fact_check_context_max_chars"]),
        min_v=4000,
        max_v=250000,
    )
    out["fact_check_max_items"] = _coerce_int_in_range(src.get("fact_check_max_items"), 12, min_v=1, max_v=100)
    out["fact_check_prompt"] = _coerce_str(src.get("fact_check_prompt"), str(DEFAULT_CONFIG["fact_check_prompt"]), max_len=12000)

    out["speech_preprocess_enabled"] = _coerce_bool(src.get("speech_preprocess_enabled"), bool(DEFAULT_CONFIG["speech_preprocess_enabled"]))
    out["speech_vad_enabled"] = _coerce_bool(src.get("speech_vad_enabled"), bool(DEFAULT_CONFIG["speech_vad_enabled"]))
    out["speech_vad_threshold"] = _coerce_float_in_range(src.get("speech_vad_threshold"), 0.5, min_v=0.05, max_v=0.98)
    out["speech_vad_neg_threshold"] = _coerce_float_in_range(src.get("speech_vad_neg_threshold"), 0.35, min_v=0.0, max_v=0.95)
    if float(out["speech_vad_neg_threshold"]) >= float(out["speech_vad_threshold"]):
        out["speech_vad_neg_threshold"] = max(0.0, float(out["speech_vad_threshold"]) - 0.05)
    out["speech_vad_min_speech_ms"] = _coerce_int_in_range(src.get("speech_vad_min_speech_ms"), 200, min_v=50, max_v=4000)
    out["speech_vad_max_speech_s"] = _coerce_float_in_range(src.get("speech_vad_max_speech_s"), 30.0, min_v=1.0, max_v=120.0)
    out["speech_vad_min_silence_ms"] = _coerce_int_in_range(src.get("speech_vad_min_silence_ms"), 650, min_v=40, max_v=4000)
    out["speech_vad_speech_pad_ms"] = _coerce_int_in_range(src.get("speech_vad_speech_pad_ms"), 120, min_v=0, max_v=1200)
    out["speech_vad_concat_silence_ms"] = _coerce_int_in_range(src.get("speech_vad_concat_silence_ms"), 220, min_v=0, max_v=2500)
    out["speech_denoise_enabled"] = _coerce_bool(src.get("speech_denoise_enabled"), bool(DEFAULT_CONFIG["speech_denoise_enabled"]))
    out["speech_denoise_strength"] = _coerce_float_in_range(src.get("speech_denoise_strength"), 0.8, min_v=0.0, max_v=1.0)
    out["speech_denoise_floor"] = _coerce_float_in_range(src.get("speech_denoise_floor"), 0.06, min_v=0.0, max_v=1.0)
    out["whisper_vad_filter"] = _coerce_bool(src.get("whisper_vad_filter"), bool(DEFAULT_CONFIG["whisper_vad_filter"]))
    out["whisper_beam_size"] = _coerce_int_in_range(src.get("whisper_beam_size"), 1, min_v=1, max_v=8)
    out["transcription_profile"] = _coerce_choice(
        src.get("transcription_profile"),
        {"auto", "manual", "cpu_realtime", "cpu_accuracy", "gpu_realtime", "gpu_accuracy", "gpu_balanced", "gpu_quality"},
        str(DEFAULT_CONFIG["transcription_profile"]),
    )
    if out["transcription_profile"] in ("gpu_balanced", "gpu_quality"):
        out["transcription_profile"] = "gpu_accuracy"
    out["transcription_chunk_duration_seconds"] = _coerce_float_in_range(
        src.get("transcription_chunk_duration_seconds"),
        3.2,
        min_v=1.0,
        max_v=8.0,
    )
    out["transcription_chunk_overlap_seconds"] = _coerce_float_in_range(
        src.get("transcription_chunk_overlap_seconds"),
        0.16,
        min_v=0.0,
        max_v=1.2,
    )
    if float(out["transcription_chunk_overlap_seconds"]) >= float(out["transcription_chunk_duration_seconds"]):
        out["transcription_chunk_overlap_seconds"] = max(0.0, float(out["transcription_chunk_duration_seconds"]) * 0.15)
    # Back-compat performance migration:
    # if older defaults were persisted as explicit values, lift to newer lower-overhead defaults.
    try:
        raw_dur = float(raw_dict.get("transcription_chunk_duration_seconds")) if "transcription_chunk_duration_seconds" in raw_dict else None
        raw_ovr = float(raw_dict.get("transcription_chunk_overlap_seconds")) if "transcription_chunk_overlap_seconds" in raw_dict else None
        if raw_dur is not None and raw_ovr is not None and abs(raw_dur - 2.4) < 1e-6 and abs(raw_ovr - 0.24) < 1e-6:
            out["transcription_chunk_duration_seconds"] = 3.2
            out["transcription_chunk_overlap_seconds"] = 0.16
    except Exception:
        pass
    # Back-compat responsiveness migration:
    # if older fact-check defaults were persisted explicitly, lift to faster defaults.
    try:
        raw_fc_interval = int(raw_dict.get("fact_check_interval_seconds")) if "fact_check_interval_seconds" in raw_dict else None
        raw_fc_debounce = float(raw_dict.get("fact_check_debounce_seconds")) if "fact_check_debounce_seconds" in raw_dict else None
        raw_fc_min = float(raw_dict.get("fact_check_trigger_min_interval_seconds")) if "fact_check_trigger_min_interval_seconds" in raw_dict else None
        if raw_fc_interval is not None and raw_fc_interval == 25:
            out["fact_check_interval_seconds"] = 15
        if raw_fc_debounce is not None and abs(raw_fc_debounce - 2.5) < 1e-6:
            out["fact_check_debounce_seconds"] = 1.0
        if raw_fc_min is not None and abs(raw_fc_min - 10.0) < 1e-6:
            out["fact_check_trigger_min_interval_seconds"] = 4.0
    except Exception:
        pass
    out["whisper_model_size"] = _coerce_str(src.get("whisper_model_size"), str(DEFAULT_CONFIG["whisper_model_size"]), max_len=64)
    out["whisper_device"] = _coerce_choice(src.get("whisper_device"), {"cpu", "cuda", "gpu"}, str(DEFAULT_CONFIG["whisper_device"]))
    if out["whisper_device"] == "gpu":
        out["whisper_device"] = "cuda"
    out["speaker_diarization_enabled"] = _coerce_bool(
        src.get("speaker_diarization_enabled"),
        bool(DEFAULT_CONFIG["speaker_diarization_enabled"]),
    )

    out["autosave_enabled"] = _coerce_bool(src.get("autosave_enabled"), bool(DEFAULT_CONFIG["autosave_enabled"]))
    out["session_timeout_minutes"] = _coerce_int_in_range(src.get("session_timeout_minutes"), 30, min_v=1, max_v=24 * 60)
    out["verbose_logging"] = _coerce_bool(src.get("verbose_logging"), bool(DEFAULT_CONFIG["verbose_logging"]))

    out["transcript_merge_enabled"] = _coerce_bool(src.get("transcript_merge_enabled"), bool(DEFAULT_CONFIG["transcript_merge_enabled"]))
    out["transcript_merge_window_seconds"] = _coerce_float_in_range(
        src.get("transcript_merge_window_seconds"),
        4.0,
        min_v=0.0,
        max_v=30.0,
    )
    out["transcript_merge_continuation_window_seconds"] = _coerce_float_in_range(
        src.get("transcript_merge_continuation_window_seconds"),
        18.0,
        min_v=0.0,
        max_v=45.0,
    )
    if float(out["transcript_merge_continuation_window_seconds"]) < float(out["transcript_merge_window_seconds"]):
        out["transcript_merge_continuation_window_seconds"] = float(out["transcript_merge_window_seconds"])
    out["transcript_ai_mode"] = _coerce_choice(
        src.get("transcript_ai_mode"),
        {"off", "cleanup", "paraphrase"},
        str(DEFAULT_CONFIG["transcript_ai_mode"]),
    )
    out["transcript_ai_cleanup_debounce_seconds"] = _coerce_float_in_range(
        src.get("transcript_ai_cleanup_debounce_seconds"),
        1.25,
        min_v=0.2,
        max_v=15.0,
    )
    out["transcript_ai_cleanup_min_interval_seconds"] = _coerce_float_in_range(
        src.get("transcript_ai_cleanup_min_interval_seconds"),
        3.0,
        min_v=0.0,
        max_v=120.0,
    )
    out["transcript_display_mode"] = _coerce_choice(
        src.get("transcript_display_mode"),
        {"raw", "clean"},
        str(DEFAULT_CONFIG["transcript_display_mode"]),
    )
    if out["transcript_ai_mode"] == "off":
        out["transcript_display_mode"] = "raw"

    return out


def _resolve_api_key_for_provider(provider: str, explicit_key: object) -> str:
    api_key = _coerce_str(explicit_key, "", max_len=4096)
    if api_key:
        return api_key

    preset = _API_PROVIDER_PRESETS.get(_normalize_api_provider(provider), {})
    env_names = preset.get("api_key_envs")
    if isinstance(env_names, list):
        for env_name in env_names:
            if not isinstance(env_name, str):
                continue
            api_key = (os.environ.get(env_name) or "").strip()
            if api_key:
                return api_key

    env_name = preset.get("api_key_env")
    if isinstance(env_name, str) and env_name:
        api_key = (os.environ.get(env_name) or "").strip()
        if api_key:
            return api_key

    return ""


def _effective_api_route_from_values(values: dict[str, object]) -> dict[str, object]:
    base_url_raw = _coerce_str(values.get("base_url"), "", max_len=2048)
    provider = _normalize_api_provider(_coerce_str(values.get("provider"), ""))
    inferred = _infer_provider_from_base_url(base_url_raw)
    if provider == "custom" and inferred != "custom":
        provider = inferred
    preset = _API_PROVIDER_PRESETS.get(provider, {})

    base_url = (
        base_url_raw
        or (preset.get("base_url") if isinstance(preset.get("base_url"), str) else "")
        or str(DEFAULT_CONFIG["base_url"])
    )
    model = (
        _coerce_str(values.get("model"), "", max_len=512)
        or (preset.get("model") if isinstance(preset.get("model"), str) else "")
        or str(DEFAULT_CONFIG["model"])
    )
    preset_headers = _coerce_headers(preset.get("api_extra_headers"))
    extra_headers = {**preset_headers, **_coerce_headers(values.get("api_extra_headers"))}
    api_key = _resolve_api_key_for_provider(provider, values.get("api_key"))
    enabled = _coerce_bool(values.get("enabled"), True)

    return {
        "provider": provider,
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "api_extra_headers": extra_headers,
        "enabled": enabled,
    }


def _effective_api_routes(cfg: dict) -> list[dict[str, object]]:
    raw_primary = {
        "provider": cfg.get("api_provider"),
        "api_key": cfg.get("api_key"),
        "base_url": cfg.get("base_url"),
        "model": cfg.get("model"),
        "api_extra_headers": cfg.get("api_extra_headers"),
        "enabled": True,
    }
    candidates: list[dict[str, object]] = [_effective_api_route_from_values(raw_primary)]
    for route in _coerce_api_routes_list(cfg.get("api_routes")):
        candidates.append(_effective_api_route_from_values(route))

    out: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for item in candidates:
        if not _coerce_bool(item.get("enabled"), True):
            continue
        api_key = _coerce_str(item.get("api_key"), "", max_len=4096)
        if not api_key:
            continue
        base_url = _coerce_str(item.get("base_url"), "", max_len=2048)
        model = _coerce_str(item.get("model"), "", max_len=512)
        provider = _normalize_api_provider(_coerce_str(item.get("provider"), "custom"))
        headers = _coerce_headers(item.get("api_extra_headers"))
        try:
            hdr_sig = json.dumps(headers, sort_keys=True, separators=(",", ":"))
        except Exception:
            hdr_sig = "{}"
        sig = (provider, base_url, model, api_key, hdr_sig)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(
            {
                "provider": provider,
                "api_key": api_key,
                "base_url": base_url,
                "model": model,
                "api_extra_headers": headers,
            }
        )
        if len(out) >= 8:
            break

    return out


def _effective_api_settings(cfg: dict) -> tuple[str, str, str, dict[str, str]]:
    routes = _effective_api_routes(cfg)
    if routes:
        first = routes[0]
        return (
            str(first.get("api_key") or ""),
            str(first.get("base_url") or ""),
            str(first.get("model") or ""),
            _coerce_headers(first.get("api_extra_headers")),
        )
    return "", str(DEFAULT_CONFIG["base_url"]), str(DEFAULT_CONFIG["model"]), {}


def _normalize_model_id_token(value: object) -> str:
    s = _coerce_str(value, "", max_len=512).strip().casefold()
    if s.startswith("models/"):
        s = s[len("models/") :]
    return s


def _model_id_matches(requested: object, returned: object) -> bool:
    req = _normalize_model_id_token(requested)
    got = _normalize_model_id_token(returned)
    return bool(req and got and req == got)


def load_config() -> dict:
    loaded: dict = {}
    try:
        if _CONFIG_PATH.is_file():
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                loaded = data
    except Exception:
        logger.exception("Failed to load settings file")
    return _sanitize_config_values(loaded, base=DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    clean_cfg = _sanitize_config_values(cfg, base=DEFAULT_CONFIG)
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _CONFIG_PATH.with_suffix(_CONFIG_PATH.suffix + ".tmp")
        tmp_path.write_text(json.dumps(clean_cfg, indent=2), encoding="utf-8")
        tmp_path.replace(_CONFIG_PATH)
    except Exception:
        logger.exception("Failed to save settings file")
        raise


def _apply_runtime_log_levels(cfg: dict) -> None:
    verbose = bool((cfg or {}).get("verbose_logging", False))

    # Main logs stay at INFO; verbose mode enables DEBUG details on our app logger.
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    important_logger.setLevel(logging.INFO)

    # In default mode, keep noisy libraries to warnings/errors only.
    noisy_level = logging.INFO if verbose else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(noisy_level)

    log_important(
        "logging.mode",
        dedupe_key=f"verbose={verbose}",
        dedupe_window_s=0.5,
        verbose=verbose,
        noisy_level=("info" if verbose else "warning"),
    )


# Configuration
config = load_config()
_apply_runtime_log_levels(config)


# ============================================
# SESSION PERSISTENCE
# ============================================

def _sessions_dir() -> Path:
    d = _DATA_DIR / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_session(session: Session) -> None:
    if not config.get("autosave_enabled", True):
        return
    try:
        path = _sessions_dir() / f"{session.id}.json"
        path.write_text(json.dumps(session.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Failed to save session")


_SESSION_SAVE_MIN_INTERVAL_S = 0.75
_last_session_save_ts = 0.0


def save_session_throttled(session: Session) -> None:
    """Throttle session writes to reduce UI hitches under frequent transcription updates."""
    global _last_session_save_ts
    if not config.get("autosave_enabled", True):
        return

    now_ts = time.time()
    if now_ts - _last_session_save_ts < _SESSION_SAVE_MIN_INTERVAL_S:
        return

    _last_session_save_ts = now_ts
    save_session(session)


def load_session(session_id: str) -> Optional[Session]:
    try:
        path = _sessions_dir() / f"{session_id}.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return Session.from_dict(data)
    except Exception:
        logger.exception(f"Failed to load session {session_id}")
    return None


def list_sessions() -> List[dict]:
    """Return list of session summaries (id, title, started_at, message count)."""
    sessions = []
    try:
        for f in sorted(_sessions_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append({
                    "id": data.get("id"),
                    "title": data.get("title", "Untitled"),
                    "started_at": data.get("started_at"),
                    "ended_at": data.get("ended_at"),
                    "message_count": len(data.get("transcript", [])),
                    "note_count": len(data.get("notes", [])),
                    "response_count": len(data.get("responses", [])),
                    "fact_check_count": len(data.get("fact_checks", [])),
                })
            except Exception:
                pass
    except Exception:
        logger.exception("Failed to list sessions")
    return sessions


def delete_session(session_id: str) -> bool:
    try:
        path = _sessions_dir() / f"{session_id}.json"
        if path.is_file():
            path.unlink()
            return True
    except Exception:
        logger.exception(f"Failed to delete session {session_id}")
    return False


def _rebuild_llm_history_from_session(session: Session | None = None) -> None:
    """Rebuild LLM transcript history from a session after edits/deletes/loads."""
    if llm_client is None:
        return

    src = session if session is not None else current_session
    try:
        llm_client.clear_history()
        if src is None:
            return

        for m in (src.transcript or []):
            text = (getattr(m, "text", "") or "").strip()
            if not text:
                continue
            llm_client.add_transcript_message(
                getattr(m, "source", "user"),
                text,
                speaker_id=getattr(m, "speaker_id", None),
                speaker_label=getattr(m, "speaker_label", None),
            )
    except Exception:
        logger.exception("Failed to rebuild LLM history from session")


def _find_transcript_index_by_id(session: Session, message_id: str) -> int:
    target = (message_id or "").strip()
    if not target:
        return -1
    for i, tm in enumerate(session.transcript or []):
        if str(getattr(tm, "id", "") or "") == target:
            return i
    return -1


def _is_active_session(session_id: str | None) -> bool:
    sid = (session_id or "").strip()
    if not sid:
        return False
    try:
        return bool(current_session is not None and str(current_session.id or "") == sid)
    except Exception:
        return False


def create_new_session() -> Session:
    global current_session, third_party_diarizer, _TRANSCRIPT_CLEANUP_LAST_TS
    
    # End previous session
    if current_session:
        current_session.ended_at = datetime.now().isoformat()
        save_session(current_session)
    
    # Create new session
    session_id = str(uuid.uuid4())[:8]
    current_session = Session(
        id=session_id,
        started_at=datetime.now().isoformat(),
        title=f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )

    # Reset transient AI/transcript state so old-session jobs/context cannot bleed through.
    _TRANSCRIPT_LAST_UPDATE_TS.clear()
    _AI_TRIGGER_BASELINE_TEXT.clear()
    _TRANSCRIPT_CLEANUP_LAST_TS = 0.0
    _rebuild_llm_history_from_session(current_session)

    third_party_diarizer = None
    log_important("session.new", session_id=session_id, title=current_session.title)
    return current_session


def _looks_like_system_mix_device_name(name: str | None) -> bool:
    if not name:
        return False
    n = str(name).casefold()
    return any(
        pat in n
        for pat in (
            "stereo mix",
            "what u hear",
            "wave out mix",
            "monitor of",
            "mixage stÃ©rÃ©o",
        )
    )


def _looks_like_generic_input_device_name(name: str | None) -> bool:
    if not name:
        return False
    n = str(name).casefold()
    return any(
        pat in n
        for pat in (
            "microsoft sound mapper",
            "primary sound capture driver",
        )
    )


def _speech_preprocess_from_config(cfg: dict) -> SpeechPreprocessConfig:
    return SpeechPreprocessConfig(
        enabled=bool(cfg.get("speech_preprocess_enabled", True)),
        # App-level VAD is intentionally disabled; keep Whisper VAD as the only VAD path.
        vad_enabled=False,
        vad_threshold=float(cfg.get("speech_vad_threshold", 0.5)),
        vad_neg_threshold=float(cfg.get("speech_vad_neg_threshold", 0.35)),
        vad_min_speech_duration_ms=int(cfg.get("speech_vad_min_speech_ms", 200)),
        vad_max_speech_duration_s=float(cfg.get("speech_vad_max_speech_s", 30.0)),
        vad_min_silence_duration_ms=int(cfg.get("speech_vad_min_silence_ms", 250)),
        vad_speech_pad_ms=int(cfg.get("speech_vad_speech_pad_ms", 120)),
        vad_concat_silence_ms=int(cfg.get("speech_vad_concat_silence_ms", 60)),
        denoise_enabled=bool(cfg.get("speech_denoise_enabled", False)),
        denoise_strength=float(cfg.get("speech_denoise_strength", 0.8)),
        denoise_floor=float(cfg.get("speech_denoise_floor", 0.06)),
    )


def _whisper_device_from_config(cfg: dict) -> str:
    d = str(cfg.get("whisper_device", "cpu") or "cpu").strip().lower()
    if d in ("gpu", "cuda"):
        return "cuda"
    return "cpu"


def _resolve_transcription_runtime_config(cfg: dict) -> dict[str, object]:
    device = _whisper_device_from_config(cfg)
    model_size = str(cfg.get("whisper_model_size", "tiny") or "tiny").strip().lower()
    requested = str(cfg.get("transcription_profile", "auto") or "auto").strip().lower()
    valid = {"auto", "manual", "cpu_realtime", "cpu_accuracy", "gpu_realtime", "gpu_accuracy", "gpu_balanced", "gpu_quality"}
    if requested not in valid:
        requested = "auto"
    if requested in ("gpu_balanced", "gpu_quality"):
        requested = "gpu_accuracy"

    resolved = requested
    if resolved == "auto":
        resolved = "gpu_accuracy" if device == "cuda" else "cpu_realtime"
    if device != "cuda" and resolved.startswith("gpu_"):
        resolved = "cpu_realtime" if resolved == "gpu_realtime" else "cpu_accuracy"
    if device == "cuda" and resolved.startswith("cpu_"):
        resolved = "gpu_realtime" if resolved == "cpu_realtime" else "gpu_accuracy"

    profile_defaults: dict[str, dict[str, object]] = {
        "cpu_realtime": {
            "beam_size": 1,
            "chunk_duration_s": 3.8,
            "chunk_overlap_s": 0.14,
            "condition_on_previous_text": False,
        },
        "cpu_accuracy": {
            "beam_size": 2,
            "chunk_duration_s": 3.4,
            "chunk_overlap_s": 0.24,
            "condition_on_previous_text": True,
        },
        "gpu_realtime": {
            "beam_size": 1,
            "chunk_duration_s": 2.4,
            "chunk_overlap_s": 0.14,
            "condition_on_previous_text": False,
        },
        "gpu_accuracy": {
            # Prioritize transcript stability for long-form speech on CUDA.
            # Larger chunks give Whisper more sentence context, reducing
            # boundary artefacts (e.g. em-dash → period, mid-phrase splits).
            # RTF is typically 0.10-0.16 on modern GPUs, so 5.6 s is safe.
            "beam_size": 5,
            "chunk_duration_s": 5.6,
            "chunk_overlap_s": 0.50,
            "condition_on_previous_text": True,
        },
    }

    if resolved == "manual":
        beam = int(max(1, min(8, int(cfg.get("whisper_beam_size", 1) or 1))))
        duration = float(max(1.0, min(8.0, float(cfg.get("transcription_chunk_duration_seconds", 3.2) or 3.2))))
        overlap = float(max(0.0, min(1.2, float(cfg.get("transcription_chunk_overlap_seconds", 0.16) or 0.16))))
        cond_prev = bool(device == "cuda")
        # Manual baseline uplift:
        # when users keep older low-latency manual defaults on CUDA with bigger models,
        # prefer a slightly more stable decode profile without overriding custom values.
        legacy_manual_defaults = (
            beam == 1
            and abs(duration - 3.2) < 1e-6
            and abs(overlap - 0.16) < 1e-6
        )
        large_model = model_size.startswith("medium") or model_size.startswith("large")
        if device == "cuda" and large_model and legacy_manual_defaults:
            beam = 2
            duration = 3.8
            overlap = 0.24
            cond_prev = True
    else:
        params = profile_defaults.get(resolved, profile_defaults["cpu_realtime"])
        beam = int(params["beam_size"])
        duration = float(params["chunk_duration_s"])
        overlap = float(params["chunk_overlap_s"])
        cond_prev = bool(params["condition_on_previous_text"])

    if overlap >= duration:
        overlap = max(0.0, duration * 0.15)

    return {
        "requested_profile": requested,
        "resolved_profile": resolved,
        "device": device,
        "beam_size": beam,
        "chunk_duration_s": duration,
        "chunk_overlap_s": overlap,
        "condition_on_previous_text": cond_prev,
    }


def _default_speaker_label_from_id(speaker_id: str | None) -> str:
    sid = (speaker_id or "").strip()
    if not sid:
        return "Speaker"
    digits = "".join(ch for ch in sid if ch.isdigit())
    if digits:
        return f"Speaker {digits}"
    return "Speaker"


def _safe_openai_name_component(text: str, *, max_len: int = 40) -> str:
    t = (text or "").strip().replace(" ", "_")
    t = "".join(ch for ch in t if ch.isalnum() or ch in ("_", "-"))
    while "__" in t:
        t = t.replace("__", "_")
    t = t.strip("_-")
    if max_len:
        t = t[: int(max_len)]
    return t


def init_llm_client_from_config() -> None:
    global llm_client

    routes = _effective_api_routes(config)
    if not routes:
        llm_client = None
        log_important(
            "llm.unconfigured",
            level=logging.WARNING,
            dedupe_key="no-api-key",
            dedupe_window_s=30.0,
            provider=config.get("api_provider"),
        )
        return

    fallback_enabled = bool(config.get("api_fallback_enabled", True))
    if not fallback_enabled:
        routes = routes[:1]

    system_prompt = (config.get("system_prompt") or DEFAULT_CONFIG["system_prompt"])
    first = routes[0]
    api_key = str(first.get("api_key") or "")
    base_url = str(first.get("base_url") or "")
    model = str(first.get("model") or "")
    extra_headers = _coerce_headers(first.get("api_extra_headers"))
    fallback_routes = []
    for r in routes[1:]:
        fallback_routes.append(
            {
                "provider": str(r.get("provider") or "custom"),
                "api_key": str(r.get("api_key") or ""),
                "base_url": str(r.get("base_url") or ""),
                "model": str(r.get("model") or ""),
                "api_extra_headers": _coerce_headers(r.get("api_extra_headers")),
            }
        )
    signature = {
        "fallback_enabled": fallback_enabled,
        "routes": [
            {
                "provider": str(r.get("provider") or "custom"),
                "base_url": str(r.get("base_url") or ""),
                "model": str(r.get("model") or ""),
                "api_key": str(r.get("api_key") or ""),
                "api_extra_headers": _coerce_headers(r.get("api_extra_headers")),
            }
            for r in routes
        ],
    }

    if (
        llm_client is not None
        and hasattr(llm_client, "get_config_signature")
        and llm_client.get_config_signature() == signature
    ):
        llm_client.set_system_prompt(system_prompt)
        return

    previous_history = list(getattr(llm_client, "history", []) or [])
    llm_client = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=model,
        default_headers=extra_headers,
        fallback_routes=fallback_routes,
        failover_enabled=fallback_enabled,
    )
    if hasattr(llm_client, "set_config_signature"):
        llm_client.set_config_signature(signature)
    llm_client.history = previous_history
    llm_client.set_system_prompt(system_prompt)
    log_important(
        "llm.configured",
        provider=first.get("provider"),
        model=model,
        base_url=base_url,
        extra_headers=len(extra_headers or {}),
        fallback_enabled=fallback_enabled,
        routes=len(routes),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Server starting...")
    log_important("server.starting")
    init_llm_client_from_config()
    # Create initial session
    create_new_session()
    yield
    # Shutdown
    logger.info("Shutting down...")
    log_important("server.stopping")
    # Save current session
    if current_session:
        current_session.ended_at = datetime.now().isoformat()
        save_session(current_session)
    audio_manager.stop_recording()


app = FastAPI(lifespan=lifespan)

_PROJECT_ROOT = Path(__file__).resolve().parent


async def _ws_send_json(
    websocket: WebSocket,
    payload: dict,
    send_lock: asyncio.Lock | None = None,
) -> bool:
    try:
        if send_lock is None:
            await websocket.send_json(payload)
        else:
            async with send_lock:
                await websocket.send_json(payload)
        return True
    except Exception:
        return False


# ============================================
# HTTP ROUTES
# ============================================

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.get("/static/{path:path}")
def get_static(path: str):
    """Serve only the intended UI assets (avoid exposing the whole project directory)."""
    if not path:
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    normalized = Path(path).as_posix().lstrip("/")
    allowed = (
        normalized == "index.html"
        or normalized.startswith("css/")
        or normalized.startswith("js/")
        or normalized.startswith("branding/")
    )
    if not allowed or ".." in normalized:
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    file_path = (_PROJECT_ROOT / normalized).resolve()
    try:
        file_path.relative_to(_PROJECT_ROOT)
    except ValueError:
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    if not file_path.is_file():
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    # Disable caching for development - forces browser to always fetch fresh files
    headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return FileResponse(str(file_path), headers=headers)


@app.get("/favicon.ico")
def favicon():
    icon_path = (_PROJECT_ROOT / "branding" / "icon.svg").resolve()
    if icon_path.is_file():
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        return FileResponse(str(icon_path), media_type="image/svg+xml", headers=headers)
    return Response(status_code=404)


@app.get("/api/devices")
def get_devices():
    devices = audio_manager.list_microphones()

    microphones = []
    loopbacks = []
    for d in devices:
        if isinstance(d, dict):
            name = str(d.get("name", ""))
            device_id = str(d.get("id", name))
            is_loopback = bool(d.get("is_loopback", False)) or bool(d.get("is_system_mix", False)) or ("loopback" in name.lower())
        else:
            name = str(getattr(d, "name", ""))
            device_id = name
            is_loopback = bool(getattr(d, "isloopback", False)) or _looks_like_system_mix_device_name(name) or ("loopback" in name.lower())

        if not is_loopback and _looks_like_generic_input_device_name(name):
            continue

        item = {"id": device_id, "name": name}
        if is_loopback:
            loopbacks.append(item)
        else:
            microphones.append(item)

    return {"microphones": microphones, "loopbacks": loopbacks}


@app.get("/api/settings")
def get_settings():
    return {"status": "ok", "config": config}


@app.post("/api/settings")
async def update_settings(request: Request):
    global config
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"status": "error", "message": "JSON body must be an object"}, status_code=400)

    prev_config = dict(config)
    config = _sanitize_config_values(data, base=config)
    try:
        save_config(config)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to save settings: {e}"}, status_code=500)

    _apply_runtime_log_levels(config)
    init_llm_client_from_config()
    changed = [k for k in config.keys() if config.get(k) != prev_config.get(k)]
    changed_list = ",".join(changed[:12]) + (",..." if len(changed) > 12 else "")
    log_important(
        "settings.updated",
        changed_count=len(changed),
        changed_keys=(changed_list or "-"),
    )
    
    return {"status": "ok", "config": config}


@app.post("/api/settings/reset")
def api_reset_settings():
    """Reset settings to defaults and persist to disk."""
    global config, llm_client

    try:
        audio_manager.stop_recording()
    except Exception:
        pass

    config = _sanitize_config_values({}, base=DEFAULT_CONFIG)
    try:
        save_config(config)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to save settings: {e}"}, status_code=500)

    _apply_runtime_log_levels(config)
    init_llm_client_from_config()
    log_important("settings.reset")
    return {"status": "ok", "config": config}


@app.get("/api/presets")
def api_list_presets():
    presets = _load_presets()
    out = []
    for name, entry in presets.items():
        if not isinstance(entry, dict):
            continue
        saved_at = entry.get("saved_at")
        cfg = entry.get("config")
        has_api_key = False
        if isinstance(cfg, dict):
            has_api_key = bool((cfg.get("api_key") or "").strip())
            if not has_api_key:
                for route in (cfg.get("api_routes") or []):
                    if not isinstance(route, dict):
                        continue
                    if (str(route.get("api_key") or "").strip()):
                        has_api_key = True
                        break
        out.append(
            {
                "name": str(name),
                "saved_at": saved_at,
                "has_api_key": has_api_key,
            }
        )
    out.sort(key=lambda x: str(x.get("name") or "").casefold())
    return {"status": "ok", "presets": out}


@app.post("/api/presets")
async def api_save_preset(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"status": "error", "message": "JSON body must be an object"}, status_code=400)

    name = _sanitize_preset_name(str(data.get("name") or ""))
    if not name:
        return JSONResponse({"status": "error", "message": "Preset name required"}, status_code=400)

    include_api_key = bool(data.get("include_api_key", False))

    preset_cfg = _sanitize_config_values(config, base=DEFAULT_CONFIG)
    if not include_api_key:
        preset_cfg.pop("api_key", None)
        routes = preset_cfg.get("api_routes")
        if isinstance(routes, list):
            for route in routes:
                if isinstance(route, dict):
                    route.pop("api_key", None)

    presets = _load_presets()
    presets[name] = {"saved_at": datetime.now().isoformat(), "config": preset_cfg}
    try:
        _save_presets(presets)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to save preset: {e}"}, status_code=500)

    return {"status": "ok", "name": name}


@app.delete("/api/presets/{preset_name}")
def api_delete_preset(preset_name: str):
    name = _sanitize_preset_name(preset_name)
    if not name:
        return JSONResponse({"status": "error", "message": "Invalid preset name"}, status_code=400)

    presets = _load_presets()
    if name not in presets:
        return JSONResponse({"status": "error", "message": "Preset not found"}, status_code=404)

    presets.pop(name, None)
    try:
        _save_presets(presets)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to delete preset: {e}"}, status_code=500)

    return {"status": "ok"}


@app.post("/api/presets/{preset_name}/apply")
def api_apply_preset(preset_name: str):
    """Apply a preset as the active config (persisted)."""
    global config

    name = _sanitize_preset_name(preset_name)
    if not name:
        return JSONResponse({"status": "error", "message": "Invalid preset name"}, status_code=400)

    presets = _load_presets()
    entry = presets.get(name)
    if not isinstance(entry, dict):
        return JSONResponse({"status": "error", "message": "Preset not found"}, status_code=404)

    preset_cfg = entry.get("config")
    if not isinstance(preset_cfg, dict):
        return JSONResponse({"status": "error", "message": "Preset invalid"}, status_code=500)

    # Start from defaults, then apply preset. Preserve current api_key if preset omitted it.
    merged = _sanitize_config_values(preset_cfg, base=DEFAULT_CONFIG)
    if not _coerce_str(preset_cfg.get("api_key"), ""):
        merged["api_key"] = _coerce_str(config.get("api_key"), "")
    preset_routes = preset_cfg.get("api_routes")
    if isinstance(preset_routes, list):
        merged_routes = merged.get("api_routes")
        current_routes = config.get("api_routes")
        if isinstance(merged_routes, list) and isinstance(current_routes, list):
            for i, route in enumerate(merged_routes):
                if not isinstance(route, dict):
                    continue
                if _coerce_str(route.get("api_key"), ""):
                    continue
                if i >= len(current_routes):
                    continue
                current_route = current_routes[i]
                if not isinstance(current_route, dict):
                    continue
                route["api_key"] = _coerce_str(current_route.get("api_key"), "")

    try:
        audio_manager.stop_recording()
    except Exception:
        pass

    config = merged
    try:
        save_config(config)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to save settings: {e}"}, status_code=500)

    _apply_runtime_log_levels(config)
    init_llm_client_from_config()
    log_important(
        "preset.applied",
        preset=name,
        provider=config.get("api_provider"),
        model=config.get("model"),
    )
    return {"status": "ok", "config": config}


@app.post("/api/test_connection")
async def api_test_connection(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"status": "error", "message": "JSON body must be an object"}, status_code=400)

    merged_cfg = _sanitize_config_values(data, base=config)
    routes = _effective_api_routes(merged_cfg)
    if not routes:
        return JSONResponse(
            {
                "status": "error",
                "message": "Missing API key (set it in Settings, API routes, or via provider env var)",
            },
            status_code=400,
        )
    fallback_enabled = bool(merged_cfg.get("api_fallback_enabled", True))
    attempt_routes = routes if fallback_enabled else routes[:1]

    try:
        from openai import AsyncOpenAI

        failures: list[dict[str, object]] = []
        for idx, route in enumerate(attempt_routes):
            provider = str(route.get("provider") or "custom")
            api_key = str(route.get("api_key") or "")
            base_url = str(route.get("base_url") or "")
            model = str(route.get("model") or "")
            extra_headers = _coerce_headers(route.get("api_extra_headers"))

            t0 = time.monotonic()
            client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)

            model_list_ok = False
            model_found = None
            model_list_error = None
            model_retrieve_error = None
            try:
                resp = await asyncio.wait_for(client.models.list(), timeout=6.0)
                model_list_ok = True
                ids = [getattr(m, "id", None) for m in getattr(resp, "data", [])]
                ids = [i for i in ids if isinstance(i, str)]
                if model:
                    model_found = any(_model_id_matches(model, mid) for mid in ids)
                else:
                    model_found = None
            except Exception as e:
                model_list_error = str(e)

            # Some providers return aliased IDs from /models (or no usable list).
            # If exact list matching says False, verify directly with retrieve().
            if model and model_found is False:
                try:
                    _ = await asyncio.wait_for(client.models.retrieve(model), timeout=6.0)
                    model_found = True
                except Exception as e:
                    model_retrieve_error = str(e)

            chat_ok = False
            chat_error = None
            try:
                await asyncio.wait_for(
                    client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a connection test. Reply with exactly: OK"},
                            {"role": "user", "content": "ping"},
                        ],
                        max_tokens=1,
                        temperature=0,
                        stream=False,
                    ),
                    timeout=10.0,
                )
                chat_ok = True
            except Exception as e:
                chat_error = str(e)

            latency_ms = int((time.monotonic() - t0) * 1000)
            if chat_ok:
                return {
                    "status": "ok",
                    "message": "Connection OK" if idx == 0 else f"Connection OK via fallback #{idx + 1}",
                    "latency_ms": latency_ms,
                    "model_list_ok": model_list_ok,
                    "model_found": model_found,
                    "model_list_error": model_list_error,
                    "model_retrieve_error": model_retrieve_error,
                    "provider": provider,
                    "base_url": base_url,
                    "model": model,
                    "route_index": idx + 1,
                    "attempted_routes": len(attempt_routes),
                    "fallback_used": idx > 0,
                    "failed_routes": failures,
                }

            failures.append(
                {
                    "route_index": idx + 1,
                    "provider": provider,
                    "base_url": base_url,
                    "model": model,
                    "latency_ms": latency_ms,
                    "message": chat_error or "Connection test failed",
                    "model_list_ok": model_list_ok,
                    "model_found": model_found,
                    "model_list_error": model_list_error,
                    "model_retrieve_error": model_retrieve_error,
                }
            )

        last = failures[-1] if failures else {}
        return {
            "status": "error",
            "message": str(last.get("message") or "Connection test failed"),
            "latency_ms": int(last.get("latency_ms") or 0),
            "model_list_ok": bool(last.get("model_list_ok")),
            "model_found": last.get("model_found"),
            "model_list_error": last.get("model_list_error"),
            "model_retrieve_error": last.get("model_retrieve_error"),
            "provider": last.get("provider"),
            "base_url": last.get("base_url"),
            "model": last.get("model"),
            "route_index": last.get("route_index"),
            "attempted_routes": len(attempt_routes),
            "fallback_used": False,
            "failed_routes": failures,
        }
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Test failed: {e}"}, status_code=500)


# Session endpoints
@app.get("/api/sessions")
def api_list_sessions():
    return {"status": "ok", "sessions": list_sessions()}


@app.get("/api/sessions/current")
def api_get_current_session():
    if current_session:
        return {"status": "ok", "session": current_session.to_dict()}
    return {"status": "error", "message": "No active session"}


@app.post("/api/sessions/new")
def api_new_session():
    session = create_new_session()
    return {"status": "ok", "session": session.to_dict()}


@app.get("/api/sessions/{session_id}")
def api_get_session(session_id: str):
    session = load_session(session_id)
    if session:
        return {"status": "ok", "session": session.to_dict()}
    return JSONResponse({"status": "error", "message": "Session not found"}, status_code=404)


@app.post("/api/sessions/{session_id}/load")
def api_load_session(session_id: str):
    """Load a previous session as the current active session (resume in Live Session)."""
    global current_session

    session = load_session(session_id)
    if not session:
        return JSONResponse({"status": "error", "message": "Session not found"}, status_code=404)

    current_session = session

    # Rebuild LLM history for policy/notes/response/fact-check context from transcript.
    _rebuild_llm_history_from_session(session)

    return {"status": "ok", "session": session.to_dict()}


@app.delete("/api/sessions/{session_id}")
def api_delete_session(session_id: str):
    if delete_session(session_id):
        return {"status": "ok"}
    return JSONResponse({"status": "error", "message": "Failed to delete session"}, status_code=500)


@app.delete("/api/sessions")
def api_delete_all_sessions():
    """Delete all saved sessions (bulk clear history)."""
    deleted = 0
    failed = 0
    try:
        for f in _sessions_dir().glob("*.json"):
            try:
                f.unlink()
                deleted += 1
            except Exception:
                failed += 1
    except Exception:
        logger.exception("Failed to clear session history")
        return JSONResponse({"status": "error", "message": "Failed to clear history"}, status_code=500)
    return {"status": "ok", "deleted": deleted, "failed": failed}


# Notes endpoints
@app.post("/api/notes")
async def api_add_note(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)
    
    content = _normalize_note_content(data.get("content", ""), max_chars=1500)
    if not content:
        return JSONResponse({"status": "error", "message": "Note content required"}, status_code=400)
    
    note = NoteItem(
        id=str(uuid.uuid4())[:8],
        content=content,
        timestamp=datetime.now().strftime("%H:%M:%S"),
        source=_normalize_note_source(data.get("source", "manual"), default="manual"),
        category=_normalize_note_category(data.get("category", "general")),
        pinned=bool(data.get("pinned", False)),
        session_id=current_session.id if current_session else None,
    )
    
    if current_session:
        current_session.notes.append(note)
        save_session(current_session)
    
    return {"status": "ok", "note": asdict(note)}


@app.patch("/api/notes/{note_id}")
async def api_update_note(note_id: str, request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)
    
    if current_session:
        for note in current_session.notes:
            if note.id == note_id:
                if "content" in data:
                    normalized = _normalize_note_content(data.get("content"), max_chars=1500)
                    if normalized:
                        note.content = normalized
                if "pinned" in data:
                    note.pinned = bool(data["pinned"])
                if "completed" in data:
                    note.completed = bool(data["completed"])
                if "category" in data:
                    note.category = _normalize_note_category(data.get("category"))
                save_session(current_session)
                return {"status": "ok", "note": asdict(note)}
    
    return JSONResponse({"status": "error", "message": "Note not found"}, status_code=404)


@app.delete("/api/notes/{note_id}")
def api_delete_note(note_id: str):
    if current_session:
        for i, note in enumerate(current_session.notes):
            if note.id == note_id:
                current_session.notes.pop(i)
                save_session(current_session)
                return {"status": "ok"}
    return JSONResponse({"status": "error", "message": "Note not found"}, status_code=404)


@app.post("/api/notes/clear")
def api_clear_notes():
    if current_session:
        current_session.notes = []
        save_session(current_session)
    return {"status": "ok"}


# ============================================
# WEBSOCKET
# ============================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global _TRANSCRIPT_ACTIVITY_SEQ
    await websocket.accept()
    logger.info("WebSocket connected")
    log_important("ws.connected")

    state = ConnectionState()
    send_lock = asyncio.Lock()
    response_lock = asyncio.Lock()
    notes_lock = asyncio.Lock()
    fact_check_lock = asyncio.Lock()
    transcript_cleanup_lock = asyncio.Lock()
    response_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    response_task = asyncio.create_task(run_response_loop(websocket, state, send_lock, response_lock, response_queue))
    notes_trigger_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    notes_trigger_task = asyncio.create_task(run_notes_trigger_loop(websocket, send_lock, notes_trigger_queue, notes_lock))
    fact_check_trigger_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    fact_check_trigger_task = asyncio.create_task(
        run_fact_check_trigger_loop(websocket, send_lock, fact_check_trigger_queue, fact_check_lock)
    )
    transcript_cleanup_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    transcript_cleanup_task = asyncio.create_task(
        run_transcript_cleanup_loop(websocket, send_lock, transcript_cleanup_queue, transcript_cleanup_lock)
    )

    # Send current session info
    if current_session:
        await _ws_send_json(websocket, {
            "type": "session_info",
            "session_id": current_session.id,
            "started_at": current_session.started_at,
        }, send_lock)
        await _ws_send_json(websocket, {"type": "session_context", "context": (current_session.context or "")}, send_lock)
        await _ws_send_json(websocket, {"type": "notes_update", "notes": [asdict(n) for n in current_session.notes]}, send_lock)
        await _ws_send_json(websocket, {"type": "response_update", "responses": [asdict(r) for r in current_session.responses]}, send_lock)
        await _ws_send_json(websocket, {"type": "fact_checks_update", "fact_checks": [asdict(f) for f in current_session.fact_checks]}, send_lock)

    def _audio_runtime_enabled() -> bool:
        return os.environ.get("AI_ASSISTANT_ENABLE_AUDIO") == "1"

    def _on_off(flag: bool) -> str:
        return "on" if flag else "off"

    def _audio_snapshot_status() -> str:
        if not _audio_runtime_enabled():
            return "Audio: capture unavailable in this run. Set AI_ASSISTANT_ENABLE_AUDIO=1 to enable (may be unstable)."

        requested_mic = bool(config.get("mic_enabled"))
        requested_loopback = bool(config.get("loopback_enabled"))
        active_mic = bool(getattr(audio_manager, "mic_active", False))
        active_loopback = bool(getattr(audio_manager, "loopback_active", False))
        active = bool(getattr(audio_manager, "is_recording", False)) and (active_mic or active_loopback)

        if active:
            return f"Audio: capture active (mic={_on_off(active_mic)}, system={_on_off(active_loopback)})."

        if requested_mic or requested_loopback:
            return (
                f"Audio: ready (capture idle). Configured: "
                f"mic={_on_off(requested_mic)}, system={_on_off(requested_loopback)}."
            )

        return "Audio: ready (capture idle). Both mic and system audio toggles are off."

    def _drain_queue(q: asyncio.Queue | None) -> None:
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

    await _ws_send_json(websocket, {"type": "status", "message": _audio_snapshot_status()}, send_lock)
    
    global transcription_model
    transcription_tasks_started = False
    enable_loopback = False
    audio_disabled_warned = False

    async def start_audio_capture():
        global transcription_model
        nonlocal transcription_tasks_started, enable_loopback, audio_disabled_warned
        if transcription_tasks_started or audio_manager.is_recording:
            log_important(
                "audio.start.skip",
                reason="already-running",
                recording=bool(audio_manager.is_recording),
            )
            await _ws_send_json(
                websocket,
                {"type": "status", "message": _audio_snapshot_status()},
                send_lock,
            )
            return

        # Audio capture via soundcard is currently unstable on some Windows/Python setups (can hard-crash).
        # Keep it opt-in via env var so the UI doesn't immediately close.
        if not _audio_runtime_enabled():
            if not audio_disabled_warned:
                audio_disabled_warned = True
                log_important(
                    "audio.start.blocked",
                    level=logging.WARNING,
                    reason="runtime-disabled",
                )
                await _ws_send_json(
                    websocket,
                    {
                        "type": "status",
                        "message": "Audio: capture disabled in this run. Set AI_ASSISTANT_ENABLE_AUDIO=1 to enable (may be unstable).",
                    },
                    send_lock,
                )
            return

        enable_loopback = bool(config.get("loopback_enabled"))
        enable_mic = bool(config.get("mic_enabled"))

        # IMPORTANT: On Windows, loading multiple faster-whisper models in-process can cause
        # native heap corruption (0xc0000374) depending on Python/ctranslate2 builds.
        # Use a single shared model instance; transcription calls are serialized by an internal lock.
        global transcription_model
        if (enable_mic or enable_loopback) and transcription_model is None:
            logger.info("Loading transcription model...")
            model_size = str(config.get("whisper_model_size", "tiny") or "tiny").strip() or "tiny"
            runtime_cfg = _resolve_transcription_runtime_config(config)
            transcription_model = TranscriptionEngine(
                model_size=model_size,
                device=str(runtime_cfg.get("device") or "cpu"),
                preprocess=_speech_preprocess_from_config(config),
                whisper_vad_filter=bool(config.get("whisper_vad_filter", True)),
                beam_size=int(runtime_cfg.get("beam_size") or 1),
                chunk_duration_s=float(runtime_cfg.get("chunk_duration_s") or 3.2),
                chunk_overlap_s=float(runtime_cfg.get("chunk_overlap_s") or 0.16),
                condition_on_previous_text=bool(runtime_cfg.get("condition_on_previous_text")),
                runtime_profile=str(runtime_cfg.get("resolved_profile") or "manual"),
            )
            log_important(
                "transcription.backend",
                model=model_size,
                device=getattr(transcription_model, "device", "unknown"),
                compute=getattr(transcription_model, "compute_type", "unknown"),
                requested_profile=str(runtime_cfg.get("requested_profile") or "auto"),
                profile=str(runtime_cfg.get("resolved_profile") or "manual"),
                beam=int(runtime_cfg.get("beam_size") or 1),
                chunk_s=f"{float(runtime_cfg.get('chunk_duration_s') or 0.0):.2f}",
                overlap_s=f"{float(runtime_cfg.get('chunk_overlap_s') or 0.0):.2f}",
            )

        mic_name = config.get("mic_device")
        loopback_name = config.get("loopback_device")
        # If the stored device is a numeric index, we can't safely infer its name here.
        # Only apply the system-mix heuristic for name-based selections.
        if enable_mic and mic_name and (not str(mic_name).strip().isdigit()) and _looks_like_system_mix_device_name(mic_name):
            await _ws_send_json(
                websocket,
                {
                    "type": "status",
                    "message": "Audio: selected mic looks like system audio (e.g. Stereo Mix); ignoring. Pick a microphone in Settings.",
                },
                send_lock,
            )
            mic_name = None

        if not enable_mic and not enable_loopback:
            log_important(
                "audio.start.skip",
                reason="no-inputs-enabled",
                mic=enable_mic,
                loopback=enable_loopback,
            )
            await _ws_send_json(
                websocket,
                {"type": "status", "message": "Audio: nothing to capture (both mic and system audio disabled)."},
                send_lock,
            )
            return

        await _ws_send_json(
            websocket,
            {
                "type": "status",
                "message": f"Audio: starting capture (mic={'on' if enable_mic else 'off'}, loopback={'on' if enable_loopback else 'off'}).",
            },
            send_lock,
        )

        try:
            report = audio_manager.start_recording(
                mic_name=mic_name,
                loopback_name=loopback_name,
                start_mic=enable_mic,
                start_loopback=enable_loopback,
            )
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
            log_important("audio.start.failed", level=logging.ERROR, error=e)
            await _ws_send_json(
                websocket,
                {"type": "error", "message": f"Failed to start audio recording: {e}"},
                send_lock,
            )
            return

        # If specific streams failed, notify the UI (loopback may still be active).
        if enable_mic and not getattr(audio_manager, "mic_active", False):
            await _ws_send_json(
                websocket,
                {"type": "error", "message": "Audio: microphone failed to start. Check Windows mic privacy permissions and pick a working mic in Settings."},
                send_lock,
            )
        if enable_loopback and not getattr(audio_manager, "loopback_active", False):
            await _ws_send_json(
                websocket,
                {"type": "error", "message": "Audio: system audio (loopback) failed to start. Check loopback device selection in Settings."},
                send_lock,
            )

        if not audio_manager.is_recording:
            log_important(
                "audio.start.failed",
                level=logging.ERROR,
                reason="no-streams-opened",
                requested_mic=enable_mic,
                requested_loopback=enable_loopback,
            )
            await _ws_send_json(
                websocket,
                {
                    "type": "error",
                    "message": "Audio: recording did not start (no streams opened). Check device selection.",
                },
                send_lock,
            )
            return

        log_important(
            "audio.start.ok",
            requested_mic=bool(enable_mic),
            requested_loopback=bool(enable_loopback),
            active_mic=bool(getattr(audio_manager, "mic_active", False)),
            active_loopback=bool(getattr(audio_manager, "loopback_active", False)),
            recording=bool(getattr(audio_manager, "is_recording", False)),
            mic_first_chunk=bool((report or {}).get("mic", {}).get("first_chunk")),
            loopback_first_chunk=bool((report or {}).get("loopback", {}).get("first_chunk")),
        )

        if getattr(audio_manager, "mic_active", False):
            asyncio.create_task(
                run_transcription_loop(
                    websocket,
                    "user",
                    state,
                    send_lock,
                    response_queue,
                    notes_trigger_queue,
                    fact_check_trigger_queue,
                    transcript_cleanup_queue,
                )
            )
        if getattr(audio_manager, "loopback_active", False):
            asyncio.create_task(
                run_transcription_loop(
                    websocket,
                    "third_party",
                    state,
                    send_lock,
                    response_queue,
                    notes_trigger_queue,
                    fact_check_trigger_queue,
                    transcript_cleanup_queue,
                )
            )
        transcription_tasks_started = True
        await _ws_send_json(websocket, {"type": "status", "message": _audio_snapshot_status()}, send_lock)

    async def stop_audio_capture():
        nonlocal transcription_tasks_started
        if audio_manager.is_recording:
            audio_manager.stop_recording()
            log_important("audio.stop", had_recording=True)
            await _ws_send_json(websocket, {"type": "status", "message": "Audio: capture stopped."}, send_lock)
        else:
            log_important("audio.stop", had_recording=False)
            await _ws_send_json(websocket, {"type": "status", "message": "Audio: capture already stopped."}, send_lock)
        transcription_tasks_started = False

    # Start periodic generators (separate AI processes with their own settings)
    notes_task = asyncio.create_task(run_notes_loop(websocket, send_lock, notes_lock))
    fact_check_task = asyncio.create_task(run_fact_check_loop(websocket, send_lock, fact_check_lock))

    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                msg = json.loads(data_text)
                msg_type = msg.get("type")
                
                if msg_type == "manual_input":
                    text = msg.get("text")
                    await process_transcription(
                        websocket,
                        text,
                        "user",
                        state,
                        send_lock,
                        message_kind="manual",
                        response_queue=response_queue,
                        notes_queue=notes_trigger_queue,
                        fact_check_queue=fact_check_trigger_queue,
                        transcript_cleanup_queue=transcript_cleanup_queue,
                    )
                elif msg_type == "ai_run_now":
                    _enqueue_response_request(response_queue, {"trigger": "run_now", "ts": time.time()})
                    _enqueue_fact_check_trigger(fact_check_trigger_queue)
                elif msg_type == "start_audio":
                    await start_audio_capture()
                elif msg_type == "stop_audio":
                    await stop_audio_capture()
                elif msg_type == "new_session":
                    # Drop pending jobs from previous session before switching.
                    _drain_queue(response_queue)
                    _drain_queue(notes_trigger_queue)
                    _drain_queue(fact_check_trigger_queue)
                    _drain_queue(transcript_cleanup_queue)
                    state.last_policy_eval_ts = 0.0
                    state.last_response_ts = 0.0
                    state.last_fact_check_ts = 0.0
                    session = create_new_session()
                    await _ws_send_json(websocket, {
                        "type": "session_info",
                        "session_id": session.id,
                        "started_at": session.started_at,
                    }, send_lock)
                    await _ws_send_json(websocket, {"type": "session_context", "context": (session.context or "")}, send_lock)
                    await _ws_send_json(websocket, {"type": "notes_update", "notes": []}, send_lock)
                    await _ws_send_json(websocket, {"type": "response_update", "responses": []}, send_lock)
                    await _ws_send_json(websocket, {"type": "fact_checks_update", "fact_checks": []}, send_lock)
                elif msg_type == "set_session_context":
                    ctx = msg.get("context", "")
                    if ctx is not None and not isinstance(ctx, str):
                        ctx = str(ctx)
                    ctx = (ctx or "")
                    if current_session is not None:
                        current_session.context = ctx[:8000]
                        save_session_throttled(current_session)
                        await _ws_send_json(websocket, {"type": "session_context", "context": current_session.context}, send_lock)
                elif msg_type == "update_transcript_message":
                    message_id = msg.get("id", msg.get("message_id"))
                    text = msg.get("text", "")
                    if message_id is not None and not isinstance(message_id, str):
                        message_id = str(message_id)
                    message_id = (message_id or "").strip()
                    if text is not None and not isinstance(text, str):
                        text = str(text)
                    text = (text or "").strip()

                    if current_session is None or not message_id:
                        await _ws_send_json(websocket, {"type": "error", "message": "Edit failed: no active session/message."}, send_lock)
                        continue
                    if not text:
                        await _ws_send_json(websocket, {"type": "error", "message": "Edit failed: message text cannot be empty."}, send_lock)
                        continue

                    idx = _find_transcript_index_by_id(current_session, message_id)
                    if idx < 0:
                        await _ws_send_json(websocket, {"type": "error", "message": "Edit failed: message not found."}, send_lock)
                        continue

                    tm = current_session.transcript[idx]
                    tm.text = text[:8000]
                    tm.clean_text = None
                    tm.timestamp = datetime.now().strftime("%H:%M:%S")
                    _TRANSCRIPT_LAST_UPDATE_TS[tm.id] = time.time()
                    _TRANSCRIPT_ACTIVITY_SEQ += 1
                    save_session(current_session)
                    _rebuild_llm_history_from_session(current_session)

                    if transcript_cleanup_queue is not None and config.get("ai_enabled", False) and _get_transcript_ai_mode(config) != "off":
                        _enqueue_transcript_cleanup(transcript_cleanup_queue, tm.id)
                    if config.get("ai_enabled", False) and config.get("notes_enabled", True) and (
                        config.get("notes_live_on_message", True) or config.get("notes_on_interaction_only", False)
                    ):
                        _enqueue_notes_trigger(notes_trigger_queue)
                    if config.get("ai_enabled", False) and config.get("fact_check_enabled", True) and (
                        config.get("fact_check_live_on_message", True) or config.get("fact_check_on_interaction_only", False)
                    ):
                        _enqueue_fact_check_trigger(fact_check_trigger_queue)

                    await _ws_send_json(
                        websocket,
                        {
                            "type": "transcription_update",
                            "id": tm.id,
                            "text": tm.text,
                            "clean_text": tm.clean_text,
                            "source": tm.source,
                            "timestamp": tm.timestamp,
                            "speaker_id": tm.speaker_id,
                            "speaker_label": tm.speaker_label,
                        },
                        send_lock,
                    )
                elif msg_type == "delete_transcript_message":
                    message_id = msg.get("id", msg.get("message_id"))
                    if message_id is not None and not isinstance(message_id, str):
                        message_id = str(message_id)
                    message_id = (message_id or "").strip()

                    if current_session is None or not message_id:
                        await _ws_send_json(websocket, {"type": "error", "message": "Delete failed: no active session/message."}, send_lock)
                        continue

                    idx = _find_transcript_index_by_id(current_session, message_id)
                    if idx < 0:
                        await _ws_send_json(websocket, {"type": "error", "message": "Delete failed: message not found."}, send_lock)
                        continue

                    removed = current_session.transcript.pop(idx)
                    with suppress(Exception):
                        _TRANSCRIPT_LAST_UPDATE_TS.pop(removed.id, None)
                    _TRANSCRIPT_ACTIVITY_SEQ += 1
                    save_session(current_session)
                    _rebuild_llm_history_from_session(current_session)

                    if config.get("ai_enabled", False) and config.get("notes_enabled", True) and (
                        config.get("notes_live_on_message", True) or config.get("notes_on_interaction_only", False)
                    ):
                        _enqueue_notes_trigger(notes_trigger_queue)
                    if config.get("ai_enabled", False) and config.get("fact_check_enabled", True) and (
                        config.get("fact_check_live_on_message", True) or config.get("fact_check_on_interaction_only", False)
                    ):
                        _enqueue_fact_check_trigger(fact_check_trigger_queue)

                    await _ws_send_json(
                        websocket,
                        {
                            "type": "transcription_deleted",
                            "id": removed.id,
                            "message_count": len(current_session.transcript),
                        },
                        send_lock,
                    )
                elif msg_type == "rename_speaker":
                    speaker_id = msg.get("speaker_id", msg.get("speakerId"))
                    name = msg.get("name", msg.get("speaker_label", msg.get("speakerLabel", "")))
                    if speaker_id is not None and not isinstance(speaker_id, str):
                        speaker_id = str(speaker_id)
                    speaker_id = (speaker_id or "").strip()

                    if name is not None and not isinstance(name, str):
                        name = str(name)
                    name = " ".join((name or "").split()).strip()

                    if not speaker_id or current_session is None:
                        await _ws_send_json(websocket, {"type": "error", "message": "Rename failed: no active session/speaker."}, send_lock)
                    else:
                        # Empty name => clear custom mapping back to default.
                        if not name:
                            with suppress(Exception):
                                current_session.speaker_names.pop(speaker_id, None)
                            effective_label = _default_speaker_label_from_id(speaker_id)
                        else:
                            # Keep it short and readable.
                            if len(name) > 32:
                                name = name[:32].strip()
                            current_session.speaker_names[speaker_id] = name
                            effective_label = name

                        # Update existing transcript entries for immediate UI/history consistency.
                        with suppress(Exception):
                            for tm in (current_session.transcript or []):
                                if getattr(tm, "speaker_id", None) == speaker_id:
                                    tm.speaker_label = effective_label
                        save_session_throttled(current_session)

                        # Update LLM history names so notes/policy use the renamed speaker consistently.
                        if llm_client is not None:
                            safe_id = _safe_openai_name_component(speaker_id, max_len=24)
                            safe_label = _safe_openai_name_component(effective_label, max_len=28) if name else ""
                            new_name = f"third_party_{safe_label}__{safe_id}" if (safe_label and safe_id) else (f"third_party_{safe_id}" if safe_id else "third_party")
                            with suppress(Exception):
                                for hm in (getattr(llm_client, "history", None) or []):
                                    if not isinstance(hm, dict):
                                        continue
                                    hm_name = hm.get("name")
                                    if not isinstance(hm_name, str):
                                        continue
                                    hn = hm_name.strip()
                                    if not hn.startswith("third_party_"):
                                        continue
                                    if hn == f"third_party_{safe_id}" or hn.endswith(f"__{safe_id}"):
                                        hm["name"] = new_name

                        await _ws_send_json(
                            websocket,
                            {"type": "speaker_renamed", "speaker_id": speaker_id, "speaker_label": effective_label},
                            send_lock,
                        )
                elif msg_type == "add_note":
                    # Manual note from UI
                    content = _normalize_note_content(msg.get("content", ""), max_chars=1500)
                    if content and current_session:
                        note = NoteItem(
                            id=str(uuid.uuid4())[:8],
                            content=content,
                            timestamp=datetime.now().strftime("%H:%M:%S"),
                            source="manual",
                            category=_normalize_note_category(msg.get("category", "general")),
                            pinned=bool(msg.get("pinned", False)),
                            session_id=current_session.id,
                        )
                        current_session.notes.append(note)
                        save_session(current_session)
                        await _ws_send_json(websocket, {
                            "type": "note_added",
                            "note": asdict(note),
                        }, send_lock)
                        if config.get("ai_enabled", False) and config.get("notes_enabled", True) and config.get("notes_on_interaction_only", False):
                            _enqueue_notes_trigger(notes_trigger_queue)
                        if config.get("ai_enabled", False) and config.get("fact_check_enabled", True) and config.get("fact_check_on_interaction_only", False):
                            _enqueue_fact_check_trigger(fact_check_trigger_queue)
                elif msg_type == "update_note":
                    note_id = msg.get("note_id")
                    updates = msg.get("updates", {})
                    if current_session and note_id:
                        for note in current_session.notes:
                            if note.id == note_id:
                                if "pinned" in updates:
                                    note.pinned = bool(updates["pinned"])
                                if "completed" in updates:
                                    note.completed = bool(updates["completed"])
                                if "content" in updates:
                                    normalized = _normalize_note_content(updates.get("content"), max_chars=1500)
                                    if normalized:
                                        note.content = normalized
                                if "category" in updates:
                                    note.category = _normalize_note_category(updates.get("category"))
                                save_session(current_session)
                                await _ws_send_json(websocket, {
                                    "type": "note_updated",
                                    "note": asdict(note),
                                }, send_lock)
                                break
                elif msg_type == "delete_note":
                    note_id = msg.get("note_id")
                    if current_session and note_id:
                        for i, note in enumerate(current_session.notes):
                            if note.id == note_id:
                                current_session.notes.pop(i)
                                save_session(current_session)
                                await _ws_send_json(websocket, {
                                    "type": "note_deleted",
                                    "note_id": note_id,
                                }, send_lock)
                                break
                elif msg_type == "clear_notes":
                    if current_session:
                        current_session.notes = []
                        save_session(current_session)
                        await _ws_send_json(websocket, {"type": "notes_cleared"}, send_lock)
                elif msg_type == "refresh_notes":
                    # Force notes generation now
                    if not config.get("ai_enabled", False):
                        await _ws_send_json(websocket, {"type": "error", "message": "AI is disabled. Enable AI to generate notes."}, send_lock)
                    else:
                        await generate_notes(websocket, send_lock, notes_lock)
                elif msg_type == "refresh_response":
                    if not config.get("ai_enabled", False):
                        await _ws_send_json(websocket, {"type": "error", "message": "AI is disabled. Enable AI to generate responses."}, send_lock)
                    elif not config.get("response_enabled", True):
                        await _ws_send_json(websocket, {"type": "error", "message": "Response system is disabled in Settings."}, send_lock)
                    else:
                        _enqueue_response_request(response_queue, {"trigger": "run_now", "ts": time.time()})
                elif msg_type == "clear_responses":
                    if current_session:
                        current_session.responses = []
                        save_session(current_session)
                        await _ws_send_json(websocket, {"type": "responses_cleared"}, send_lock)
                elif msg_type == "refresh_fact_checks":
                    if not config.get("ai_enabled", False):
                        await _ws_send_json(websocket, {"type": "error", "message": "AI is disabled. Enable AI to run fact checks."}, send_lock)
                    elif not config.get("fact_check_enabled", True):
                        await _ws_send_json(websocket, {"type": "error", "message": "Fact check system is disabled in Settings."}, send_lock)
                    else:
                        _enqueue_fact_check_trigger(fact_check_trigger_queue)
                elif msg_type == "clear_fact_checks":
                    if current_session:
                        current_session.fact_checks = []
                        save_session(current_session)
                        await _ws_send_json(websocket, {"type": "fact_checks_cleared"}, send_lock)
                    
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected (code={getattr(e, 'code', None)})")
        log_important("ws.disconnected", code=getattr(e, "code", None))
    except Exception:
        logger.exception("WebSocket crashed")
        log_important("ws.crashed", level=logging.ERROR)
    finally:
        notes_task.cancel()
        with suppress(BaseException):
            await notes_task
        fact_check_task.cancel()
        with suppress(BaseException):
            await fact_check_task
        response_task.cancel()
        with suppress(BaseException):
            await response_task
        notes_trigger_task.cancel()
        with suppress(BaseException):
            await notes_trigger_task
        fact_check_trigger_task.cancel()
        with suppress(BaseException):
            await fact_check_trigger_task
        transcript_cleanup_task.cancel()
        with suppress(BaseException):
            await transcript_cleanup_task
        audio_manager.stop_recording()


# ============================================
# WEB SEARCH (Optional)
# ============================================

def _extract_web_search_query_from_history() -> Optional[str]:
    if not llm_client or not getattr(llm_client, "history", None):
        return None

    # Prefer the most recent "You" message; fall back to any user-role message.
    fallback = None
    for m in reversed(llm_client.history):
        if (m.get("role") or "").strip().lower() != "user":
            continue
        content = (m.get("content") or "").strip()
        if not content:
            continue
        name = (m.get("name") or "").strip().lower()
        if name == "you":
            fallback = content
            break
        if fallback is None:
            fallback = content

    if not fallback:
        return None

    query = " ".join(fallback.split())
    query = query[:220].strip()
    if len(query) < 6:
        return None
    return query


def _normalize_web_search_query(query: str, *, fallback: str = "") -> str:
    import re

    for raw in (query, fallback):
        text = str(raw or "").strip()
        if not text:
            continue

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if lines:
            # Prefer the most recent line when full transcript blocks are provided.
            text = lines[-1]

        text = re.sub(
            r"^(you|user|assistant|ai|third[ _-]?party(?:\s*[a-z0-9_-]+)?|speaker\s*\d+)\s*:\s*",
            "",
            text,
            flags=re.I,
        )
        text = re.sub(r"\s+", " ", text)
        text = text.strip(" \t\r\n:;,.!?-")
        text = text[:220].strip()
        if len(text) >= 2:
            return text[:120]

    return ""


def _fallback_web_search_query(seed_text: str) -> str:
    query = _normalize_web_search_query(seed_text)
    if query:
        return query
    return _normalize_web_search_query(_extract_web_search_query_from_history() or "")


async def _llm_chat_create(**kwargs):
    if llm_client is None:
        raise RuntimeError("LLM not configured")
    if hasattr(llm_client, "chat_create"):
        return await llm_client.chat_create(**kwargs)
    # Back-compat fallback if client implementation is older.
    model = kwargs.pop("model", getattr(llm_client, "model", ""))
    return await llm_client.client.chat.completions.create(model=model, **kwargs)


def _looks_like_context_overflow_error(error: Exception) -> bool:
    t = " ".join(str(error or "").split()).casefold()
    if not t:
        return False
    patterns = (
        "maximum context length",
        "context length exceeded",
        "too many tokens",
        "token limit",
        "prompt is too long",
        "request too large",
        "context_window_exceeded",
    )
    return any(p in t for p in patterns)


def _strip_html_tags(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s or "")


def _ddg_lite_search_sync(query: str, *, timeout_s: float = 6.0, max_results: int = 5) -> list[dict]:
    """
    Fetch quick web results via DuckDuckGo Lite HTML.
    Returns: [{title, snippet, url}]
    """
    import html as _html
    import re
    import urllib.parse
    import urllib.request

    q = (query or "").strip()
    if not q:
        return []

    url = "https://lite.duckduckgo.com/lite/?" + urllib.parse.urlencode({"q": q})
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=float(timeout_s)) as resp:
        raw = resp.read()

    page = raw.decode("utf-8", errors="replace")

    titles: list[str] = []
    urls: list[str] = []
    for m in re.finditer(r"href=\"//duckduckgo\.com/l/\?uddg=([^&\"]+)[^\"]*\"[^>]*class='result-link'>(.*?)</a>", page, flags=re.I | re.S):
        uddg = urllib.parse.unquote(m.group(1))
        title = _strip_html_tags(_html.unescape(m.group(2))).strip()
        if title:
            titles.append(" ".join(title.split()))
            urls.append(uddg)
        if len(titles) >= int(max_results):
            break

    snippets: list[str] = []
    for m in re.finditer(r"class='result-snippet'>(.*?)</td>", page, flags=re.I | re.S):
        snip = _strip_html_tags(_html.unescape(m.group(1))).strip()
        snip = " ".join(snip.split())
        if snip:
            snippets.append(snip)
        if len(snippets) >= int(max_results):
            break

    results: list[dict] = []
    for i, title in enumerate(titles):
        results.append(
            {
                "title": title,
                "snippet": snippets[i] if i < len(snippets) else "",
                "url": urls[i] if i < len(urls) else "",
            }
        )
        if len(results) >= int(max_results):
            break

    cleaned: list[dict] = []
    seen = set()
    for r in results:
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        key = (title + "|" + url).casefold()
        if not title or key in seen:
            continue
        seen.add(key)
        cleaned.append({"title": title, "snippet": (r.get("snippet") or "").strip(), "url": url})
        if len(cleaned) >= int(max_results):
            break

    return cleaned


def _ddg_instant_answer_search_sync(query: str, *, timeout_s: float = 4.0, max_results: int = 5) -> list[dict]:
    import urllib.parse
    import urllib.request

    q = (query or "").strip()
    if not q:
        return []

    url = (
        "https://api.duckduckgo.com/?"
        + urllib.parse.urlencode(
            {
                "q": q,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1",
            }
        )
    )

    with urllib.request.urlopen(url, timeout=float(timeout_s)) as resp:
        raw = resp.read()

    data = json.loads(raw.decode("utf-8", errors="replace"))

    results: list[dict] = []
    abstract = (data.get("AbstractText") or "").strip()
    abstract_url = (data.get("AbstractURL") or "").strip()
    if abstract:
        results.append({"text": abstract, "url": abstract_url})

    def _walk_topics(topics) -> None:
        if not isinstance(topics, list):
            return
        for t in topics:
            if not isinstance(t, dict):
                continue
            if isinstance(t.get("Topics"), list):
                _walk_topics(t.get("Topics"))
                continue
            text = (t.get("Text") or "").strip()
            first_url = (t.get("FirstURL") or "").strip()
            if text:
                results.append({"text": text, "url": first_url})

    _walk_topics(data.get("RelatedTopics"))

    cleaned: list[dict] = []
    seen = set()
    for r in results:
        text = (r.get("text") or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"text": text, "url": (r.get("url") or "").strip()})
        if len(cleaned) >= int(max_results):
            break

    return cleaned


_WEB_SEARCH_CACHE: dict[str, tuple[float, str]] = {}
_WEB_SEARCH_LAST_TS: float = 0.0


def _web_search_cache_put(key: str, value: str, *, ts: float) -> None:
    _WEB_SEARCH_CACHE[key] = (ts, value)
    # Best-effort eviction to avoid unbounded growth.
    if len(_WEB_SEARCH_CACHE) <= 60:
        return
    try:
        oldest = sorted(_WEB_SEARCH_CACHE.items(), key=lambda kv: kv[1][0])[:20]
        for k, _ in oldest:
            _WEB_SEARCH_CACHE.pop(k, None)
    except Exception:
        pass


async def _decide_web_search(
    *,
    seed_text: str,
    purpose: str,
) -> tuple[bool, str]:
    """
    Ask the model whether to run a web search and what query to use.
    Returns: (should_search, query)
    """
    mode = (config.get("web_search_mode") or "auto").strip().lower()
    if mode not in ("auto", "always"):
        mode = "auto"

    seed = " ".join((seed_text or "").split()).strip()

    # In always mode, avoid a second LLM call and derive query directly.
    if mode == "always":
        query = _fallback_web_search_query(seed)
        return bool(query), query

    if not llm_client:
        return False, ""
    if len(seed) < 6:
        return False, ""

    system_prompt = (
        "You are deciding whether to call a web_search tool before answering.\n"
        f"Purpose: {purpose}\n\n"
        "Use web_search when it would materially improve accuracy: up-to-date facts, specific references, product docs, "
        "named entities, version-specific info, definitions you are unsure about.\n"
        "Do NOT web_search for purely personal/subjective content, or when the answer is already obvious from the transcript.\n\n"
        "Output ONLY valid JSON with keys:\n"
        "- search: boolean\n"
        "- query: string (2..120 chars)\n"
        "- reason: string (short)\n"
    )

    resp = await _llm_chat_create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": seed[:900]},
        ],
        stream=False,
        temperature=0,
        max_tokens=160,
        response_format={"type": "json_object"},
    )

    content = _extract_chat_content_best_effort(resp)

    def _parse_json_best_effort(s: str) -> dict:
        if not isinstance(s, str):
            return {}
        ss = s.strip()
        if not ss:
            return {}
        try:
            obj = json.loads(ss)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            pass
        # Some providers ignore response_format; try to extract a JSON object substring.
        start = ss.find("{")
        end = ss.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = ss[start : end + 1]
            try:
                obj = json.loads(snippet)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}

    data = _parse_json_best_effort(content)
    # Parse "search" strictly; avoid treating non-empty strings like "false" as truthy.
    should = _coerce_bool(data.get("search"), False)
    query = _normalize_web_search_query(data.get("query") or "", fallback=seed)
    if not query:
        query = _fallback_web_search_query(seed)
    if len(query) < 2:
        return False, ""
    return should, query


async def _maybe_build_web_search_context(*, purpose: str, seed_text: str) -> Optional[str]:
    if not config.get("web_search_enabled", False):
        return None

    global _WEB_SEARCH_LAST_TS
    now_ts = time.time()

    try:
        should, query = await _decide_web_search(seed_text=seed_text, purpose=purpose)
    except Exception as e:
        logger.warning(f"Web search decision failed: {e}")
        if (config.get("web_search_mode") or "auto").strip().lower() == "always":
            should, query = True, _fallback_web_search_query(seed_text)
        else:
            should, query = False, ""

    if not should or not query:
        return None

    ttl = float(config.get("web_search_cache_ttl_seconds", 180.0) or 180.0)
    ttl = max(10.0, min(3600.0, ttl))
    cache_key = _normalize_web_search_query(query).casefold()
    if not cache_key:
        return None
    cached = _WEB_SEARCH_CACHE.get(cache_key)
    if cached and (now_ts - cached[0]) < ttl:
        log_important(
            "web_search.cache_hit",
            purpose=purpose,
            query=query,
            dedupe_key=f"{purpose}|{cache_key}",
            dedupe_window_s=8.0,
        )
        return cached[1]

    min_interval = float(config.get("web_search_min_interval_seconds", 6.0) or 6.0)
    min_interval = max(0.0, min(60.0, min_interval))
    since_last = now_ts - _WEB_SEARCH_LAST_TS
    if since_last < min_interval:
        wait_s = max(0.0, min_interval - since_last)
        log_important(
            "web_search.cooldown_skip",
            purpose=purpose,
            wait_s=f"{wait_s:.2f}",
            dedupe_key=f"{purpose}|{cache_key}",
            dedupe_window_s=4.0,
        )
        return None

    max_results = int(config.get("web_search_max_results", 5) or 5)
    max_results = max(1, min(10, max_results))
    timeout_s = float(config.get("web_search_timeout_seconds", 6.0) or 6.0)
    timeout_s = max(2.0, min(20.0, timeout_s))
    _WEB_SEARCH_LAST_TS = now_ts
    log_important(
        "web_search.start",
        purpose=purpose,
        mode=(config.get("web_search_mode") or "auto"),
        query=query,
        timeout_s=f"{timeout_s:.1f}",
        max_results=max_results,
    )

    try:
        results = await asyncio.to_thread(
            _ddg_lite_search_sync,
            query,
            timeout_s=timeout_s,
            max_results=max_results,
        )
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        log_important("web_search.error", level=logging.WARNING, purpose=purpose, provider="ddg_lite", error=e)
        results = []

    # Fallback to instant answer API if Lite fails.
    if not results:
        try:
            ia = await asyncio.to_thread(
                _ddg_instant_answer_search_sync,
                query,
                timeout_s=min(6.0, timeout_s),
                max_results=max_results,
            )
            if ia:
                log_important(
                    "web_search.results",
                    purpose=purpose,
                    provider="ddg_instant",
                    count=len(ia),
                    dedupe_key=f"{purpose}|{cache_key}|ddg_instant",
                    dedupe_window_s=2.0,
                )
                lines = [
                    "Web search results (DuckDuckGo Instant Answer API; may be incomplete/outdated):",
                    f"Query: {query}",
                ]
                for i, r in enumerate(ia, start=1):
                    text = (r.get("text") or "").strip()
                    url = (r.get("url") or "").strip()
                    if url:
                        lines.append(f"{i}. {text}\n   Source: {url}")
                    else:
                        lines.append(f"{i}. {text}")
                ctx = "\n".join(lines)[:2200]
                _web_search_cache_put(cache_key, ctx, ts=now_ts)
                return ctx
        except Exception as e:
            logger.warning(f"Web search fallback failed: {e}")
            log_important("web_search.error", level=logging.WARNING, purpose=purpose, provider="ddg_instant", error=e)

        log_important(
            "web_search.empty",
            purpose=purpose,
            provider="ddg_lite_and_instant",
            dedupe_key=f"{purpose}|{cache_key}",
            dedupe_window_s=3.0,
        )
        return None

    log_important(
        "web_search.results",
        purpose=purpose,
        provider="ddg_lite",
        count=len(results),
        dedupe_key=f"{purpose}|{cache_key}|ddg_lite",
        dedupe_window_s=2.0,
    )
    lines = [
        "Web search results (DuckDuckGo Lite; verify details on the linked sources):",
        f"Query: {query}",
    ]
    for i, r in enumerate(results, start=1):
        title = (r.get("title") or "").strip()
        snippet = (r.get("snippet") or "").strip()
        url = (r.get("url") or "").strip()
        line = f"{i}. {title}"
        if snippet:
            line += f" - {snippet}"
        if url:
            line += f"\n   Source: {url}"
        lines.append(line)

    ctx = "\n".join(lines)[:2200]
    _web_search_cache_put(cache_key, ctx, ts=now_ts)
    return ctx


# ============================================
# RESPONSE / FACT CHECK LOOPS (Non-blocking)
# ============================================

def _latest_transcript_update_ts() -> float:
    try:
        if not _TRANSCRIPT_LAST_UPDATE_TS:
            return 0.0
        return max(float(v or 0.0) for v in _TRANSCRIPT_LAST_UPDATE_TS.values())
    except Exception:
        return 0.0


def _seconds_since_latest_transcript_update(now_ts: float | None = None) -> float:
    now = float(now_ts if now_ts is not None else time.time())
    last_ts = _latest_transcript_update_ts()
    if last_ts <= 0.0:
        return 1e9
    return max(0.0, now - last_ts)


async def _wait_for_transcript_settle(*, queue: asyncio.Queue | None = None) -> None:
    quiet_s = _cfg_float(
        config.get("ai_transcript_settle_seconds"),
        float(DEFAULT_CONFIG["ai_transcript_settle_seconds"]),
        min_v=0.0,
        max_v=20.0,
    )
    if quiet_s <= 0.0:
        return

    max_wait_s = _cfg_float(
        config.get("ai_transcript_settle_max_wait_seconds"),
        float(DEFAULT_CONFIG["ai_transcript_settle_max_wait_seconds"]),
        min_v=0.2,
        max_v=120.0,
    )
    deadline = time.time() + max_wait_s

    while True:
        now_ts = time.time()
        since_last = _seconds_since_latest_transcript_update(now_ts)
        if since_last >= quiet_s:
            return
        if now_ts >= deadline:
            return

        wait_s = min(
            0.35,
            max(0.05, quiet_s - since_last),
            max(0.05, deadline - now_ts),
        )
        if queue is None:
            await asyncio.sleep(wait_s)
            continue
        try:
            await asyncio.wait_for(queue.get(), timeout=wait_s)
        except asyncio.TimeoutError:
            continue
        # Keep queue coalesced while waiting.
        while True:
            try:
                queue.get_nowait()
            except Exception:
                break


async def _wait_for_transcript_settle_response(req: dict, response_queue: asyncio.Queue) -> dict:
    quiet_s = _cfg_float(
        config.get("ai_transcript_settle_seconds"),
        float(DEFAULT_CONFIG["ai_transcript_settle_seconds"]),
        min_v=0.0,
        max_v=20.0,
    )
    if quiet_s <= 0.0:
        return req

    max_wait_s = _cfg_float(
        config.get("ai_transcript_settle_max_wait_seconds"),
        float(DEFAULT_CONFIG["ai_transcript_settle_max_wait_seconds"]),
        min_v=0.2,
        max_v=120.0,
    )
    deadline = time.time() + max_wait_s
    latest_req = req

    while True:
        now_ts = time.time()
        since_last = _seconds_since_latest_transcript_update(now_ts)
        if since_last >= quiet_s:
            return latest_req
        if now_ts >= deadline:
            return latest_req

        wait_s = min(
            0.35,
            max(0.05, quiet_s - since_last),
            max(0.05, deadline - now_ts),
        )

        try:
            nxt = await asyncio.wait_for(response_queue.get(), timeout=wait_s)
            if isinstance(nxt, dict):
                latest_req = nxt
                nxt_trigger = str(latest_req.get("trigger") or "").strip().lower()
                if nxt_trigger in ("manual", "run_now"):
                    # Respect explicit user-triggered runs immediately.
                    return latest_req
        except asyncio.TimeoutError:
            continue

        # Coalesce queued requests to latest.
        while True:
            try:
                nxt = response_queue.get_nowait()
            except Exception:
                break
            if isinstance(nxt, dict):
                latest_req = nxt
                nxt_trigger = str(latest_req.get("trigger") or "").strip().lower()
                if nxt_trigger in ("manual", "run_now"):
                    return latest_req


def _enqueue_response_request(response_queue: asyncio.Queue, payload: dict) -> None:
    try:
        response_queue.put_nowait(payload)
    except asyncio.QueueFull:
        try:
            response_queue.get_nowait()
        except Exception:
            return
        try:
            response_queue.put_nowait(payload)
        except Exception:
            return


def _enqueue_notes_trigger(notes_queue: asyncio.Queue) -> None:
    try:
        notes_queue.put_nowait(time.time())
    except asyncio.QueueFull:
        try:
            notes_queue.get_nowait()
        except Exception:
            return
        try:
            notes_queue.put_nowait(time.time())
        except Exception:
            return


def _enqueue_fact_check_trigger(fact_check_queue: asyncio.Queue) -> None:
    try:
        fact_check_queue.put_nowait(time.time())
    except asyncio.QueueFull:
        try:
            fact_check_queue.get_nowait()
        except Exception:
            return
        try:
            fact_check_queue.put_nowait(time.time())
        except Exception:
            return


async def run_notes_trigger_loop(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    notes_queue: asyncio.Queue,
    notes_lock: asyncio.Lock,
):
    last_generated_ts = 0.0

    while True:
        await notes_queue.get()

        debounce = _cfg_float(config.get("notes_debounce_seconds"), 2.5, min_v=0.2, max_v=10.0)
        debounce = max(0.2, min(10.0, debounce))

        # Debounce: keep waiting while triggers continue.
        while True:
            try:
                await asyncio.wait_for(notes_queue.get(), timeout=debounce)
                continue
            except asyncio.TimeoutError:
                break

        if not config.get("ai_enabled", False):
            continue

        if not config.get("notes_enabled", True):
            continue

        await _wait_for_transcript_settle(queue=notes_queue)

        min_interval = _cfg_float(config.get("notes_trigger_min_interval_seconds"), 10.0, min_v=0.0, max_v=300.0)
        min_interval = max(0.0, min(300.0, min_interval))
        now_ts = time.time()
        if now_ts - last_generated_ts < min_interval:
            continue

        ok = await generate_notes(websocket, send_lock, notes_lock)
        if ok:
            last_generated_ts = time.time()


async def run_fact_check_trigger_loop(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    fact_check_queue: asyncio.Queue,
    fact_check_lock: asyncio.Lock,
):
    last_generated_ts = 0.0

    async def _coalesce_fact_check_triggers(wait_s: float) -> None:
        deadline = time.time() + max(0.0, float(wait_s))
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                await asyncio.wait_for(fact_check_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break

    while True:
        await fact_check_queue.get()

        debounce = _cfg_float(config.get("fact_check_debounce_seconds"), 1.0, min_v=0.2, max_v=10.0)
        debounce = max(0.2, min(10.0, debounce))

        # Debounce: keep waiting while triggers continue.
        while True:
            try:
                await asyncio.wait_for(fact_check_queue.get(), timeout=debounce)
                continue
            except asyncio.TimeoutError:
                break

        if not config.get("ai_enabled", False):
            continue

        if not config.get("fact_check_enabled", True):
            continue

        await _wait_for_transcript_settle(queue=fact_check_queue)

        min_interval = _cfg_float(config.get("fact_check_trigger_min_interval_seconds"), 4.0, min_v=0.0, max_v=300.0)
        min_interval = max(0.0, min(300.0, min_interval))
        now_ts = time.time()
        if now_ts - last_generated_ts < min_interval:
            await _coalesce_fact_check_triggers(min_interval - (now_ts - last_generated_ts))
            now_ts = time.time()
            if now_ts - last_generated_ts < min_interval:
                continue

        ok = await generate_fact_checks(websocket, send_lock, fact_check_lock=fact_check_lock)
        if ok:
            last_generated_ts = time.time()


def _text_requests_no_response(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False

    phrases = (
        "do not respond",
        "don't respond",
        "do not reply",
        "don't reply",
        "no response",
        "no reply",
        "do not answer",
        "don't answer",
    )
    return any(p in t for p in phrases)


def _latest_non_assistant_transcript() -> Optional[tuple[str, str, str, str, str]]:
    if not current_session or not current_session.transcript:
        return None

    for m in reversed(current_session.transcript):
        try:
            if getattr(m, "source", "") == "assistant":
                continue
            text = (getattr(m, "text", "") or "").strip()
            if text:
                return (
                    text,
                    (getattr(m, "source", "user") or "user"),
                    (getattr(m, "timestamp", "") or ""),
                    (getattr(m, "speaker_label", "") or ""),
                    (getattr(m, "id", "") or ""),
                )
        except Exception:
            continue

    return None


def _normalize_response_source(source: str) -> str:
    s = (source or "").strip().lower()
    if s in ("third_party", "third-party", "loopback"):
        return "third_party"
    if s in ("user", "you"):
        return "user"
    if s == "assistant":
        return "assistant"
    return s


def _is_counterparty_source(source: str) -> bool:
    return _normalize_response_source(source) == "third_party"


def _resolve_response_target(
    *,
    seed_text: str,
    source: str,
    target_timestamp: str,
    target_speaker: str,
    target_message_id: str,
) -> tuple[str, str, str, str, str]:
    """
    Pick the best transcript anchor for response generation.
    If the latest trigger came from the user, prefer the most recent third-party claim nearby
    so response cards reference what the user should answer.
    """
    base_text = " ".join((seed_text or "").split()).strip()
    base_source = (source or "user").strip() or "user"
    base_timestamp = (target_timestamp or "").strip()
    base_speaker = (target_speaker or "").strip()
    base_id = (target_message_id or "").strip()

    if not current_session or not (current_session.transcript or []):
        return base_text, base_source, base_timestamp, base_speaker, base_id

    transcript = current_session.transcript or []
    src_kind = _normalize_response_source(base_source)
    if src_kind == "third_party":
        return base_text, base_source, base_timestamp, base_speaker, base_id

    anchor_idx = -1
    if base_id:
        for i in range(len(transcript) - 1, -1, -1):
            if str(getattr(transcript[i], "id", "") or "") == base_id:
                anchor_idx = i
                break

    if anchor_idx < 0:
        if base_text:
            for i in range(len(transcript) - 1, -1, -1):
                tm_text = " ".join((getattr(transcript[i], "text", "") or "").split()).strip()
                if not tm_text:
                    continue
                if _normalize_response_source(str(getattr(transcript[i], "source", "") or "")) != src_kind:
                    continue
                if tm_text == base_text:
                    anchor_idx = i
                    break
        if anchor_idx < 0:
            anchor_idx = len(transcript) - 1

    candidate = None
    left = max(0, anchor_idx - 12)
    for i in range(anchor_idx - 1, left - 1, -1):
        tm = transcript[i]
        if not _is_counterparty_source(str(getattr(tm, "source", "") or "")):
            continue
        tm_text = " ".join((getattr(tm, "text", "") or "").split()).strip()
        if tm_text:
            candidate = tm
            break

    if candidate is None:
        for tm in reversed(transcript[-20:]):
            if not _is_counterparty_source(str(getattr(tm, "source", "") or "")):
                continue
            tm_text = " ".join((getattr(tm, "text", "") or "").split()).strip()
            if tm_text:
                candidate = tm
                break

    if candidate is None:
        return base_text, base_source, base_timestamp, base_speaker, base_id

    chosen_text = " ".join((getattr(candidate, "text", "") or "").split()).strip()
    if not chosen_text:
        return base_text, base_source, base_timestamp, base_speaker, base_id

    return (
        chosen_text,
        str(getattr(candidate, "source", "third_party") or "third_party"),
        str(getattr(candidate, "timestamp", "") or ""),
        str(getattr(candidate, "speaker_label", "") or ""),
        str(getattr(candidate, "id", "") or ""),
    )


async def run_response_loop(
    websocket: WebSocket,
    state: ConnectionState,
    send_lock: asyncio.Lock,
    response_lock: asyncio.Lock,
    response_queue: asyncio.Queue,
):
    async def _coalesce_latest_request(initial_req: dict, wait_s: float) -> dict:
        latest_req = initial_req
        deadline = time.time() + max(0.0, float(wait_s))
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(response_queue.get(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            if isinstance(nxt, dict):
                latest_req = nxt
        return latest_req

    while True:
        req = await response_queue.get()
        while True:
            trigger = (req.get("trigger") or "").strip().lower()

            if trigger == "audio":
                req = await _wait_for_transcript_settle_response(req, response_queue)
                trigger = (req.get("trigger") or "").strip().lower()

            # For high-frequency audio triggers, prefer the newest queued context.
            if trigger == "audio" and not response_queue.empty():
                break

            if not config.get("ai_enabled", False):
                break

            if not config.get("response_enabled", True):
                break

            if llm_client is None:
                if not state.llm_not_configured_warned:
                    state.llm_not_configured_warned = True
                    await _ws_send_json(websocket, {"type": "error", "message": "LLM not configured. Check settings."}, send_lock)
                break

            now_ts = time.time()

            # Audio auto-response cooldown:
            # defer and coalesce pending triggers instead of dropping valid requests.
            if trigger == "audio":
                min_ai_interval = _cfg_float(config.get("ai_min_interval_seconds"), 0.0, min_v=0.0, max_v=120.0)
                since_last = now_ts - state.last_response_ts
                if since_last < min_ai_interval:
                    req = await _coalesce_latest_request(req, min_ai_interval - since_last)
                    continue

            text = (req.get("text") or "").strip()
            source = (req.get("source") or "user").strip() or "user"
            target_timestamp = (req.get("timestamp") or "").strip() if isinstance(req.get("timestamp"), str) else ""
            target_speaker = (req.get("speaker_label") or "").strip() if isinstance(req.get("speaker_label"), str) else ""
            target_id = (req.get("message_id") or "").strip() if isinstance(req.get("message_id"), str) else ""
            if (not text) and trigger == "run_now":
                latest = _latest_non_assistant_transcript()
                if latest:
                    text, source, target_timestamp, target_speaker, target_id = latest

            if not text:
                break

            # Hard override: if user explicitly asked for no response, do not generate response guidance.
            if text and _text_requests_no_response(text):
                await _ws_send_json(
                    websocket,
                    {
                        "type": "assistant_policy",
                        "allow": False,
                        "urgency": "wait",
                        "reason": "User requested no response",
                        "confidence": 1.0,
                        "show_withheld": bool(config.get("policy_show_withheld", True)),
                    },
                    send_lock,
                )
                break

            resolved_text, resolved_source, resolved_ts, resolved_speaker, resolved_id = _resolve_response_target(
                seed_text=text,
                source=source,
                target_timestamp=target_timestamp,
                target_speaker=target_speaker,
                target_message_id=target_id,
            )
            if resolved_text:
                source = resolved_source or source
                text = resolved_text
                target_timestamp = resolved_ts
                target_speaker = resolved_speaker
                target_id = resolved_id

            # Policy gate (applies to manual, audio, and run-now).
            require_policy = bool(config.get("response_require_policy_gate", True))
            use_policy = bool(config.get("policy_enabled", False)) or require_policy
            if use_policy:
                min_policy_interval = _cfg_float(config.get("policy_min_interval_seconds"), 0.0, min_v=0.0, max_v=60.0)
                # Manual + run-now should evaluate immediately (don't skip on interval).
                if trigger in ("manual", "run_now"):
                    min_policy_interval = 0.0

                since_policy = now_ts - state.last_policy_eval_ts
                if since_policy < min_policy_interval:
                    if trigger == "audio":
                        req = await _coalesce_latest_request(req, min_policy_interval - since_policy)
                        continue
                    break

                state.last_policy_eval_ts = now_ts
                decision = await _policy_should_respond(text=text, source=source)
                if not await _ws_send_json(
                    websocket,
                    {
                        "type": "assistant_policy",
                        "allow": bool(decision.get("allow")),
                        "urgency": decision.get("urgency"),
                        "reason": decision.get("reason"),
                        "confidence": decision.get("confidence"),
                        "show_withheld": bool(config.get("policy_show_withheld", True)),
                    },
                    send_lock,
                ):
                    break

                min_conf = _cfg_float(config.get("response_policy_min_confidence"), 0.58, min_v=0.0, max_v=1.0)
                min_conf = max(0.0, min(1.0, min_conf))
                decision_conf = float(decision.get("confidence") or 0.0)
                if not decision.get("allow") or decision.get("urgency") != "now" or decision_conf < min_conf:
                    if decision_conf < min_conf and decision.get("allow") and decision.get("urgency") == "now":
                        await _ws_send_json(
                            websocket,
                            {
                                "type": "assistant_policy",
                                "allow": False,
                                "urgency": "wait",
                                "reason": "Low policy confidence",
                                "confidence": decision_conf,
                                "show_withheld": bool(config.get("policy_show_withheld", True)),
                            },
                            send_lock,
                        )
                    break

            ok = await generate_response(
                websocket,
                send_lock,
                response_lock=response_lock,
                seed_text=text,
                source=source,
                target_timestamp=target_timestamp,
                target_speaker=target_speaker,
                target_message_id=target_id,
            )
            if ok:
                state.last_response_ts = time.time()
            break


def _extract_chat_content_best_effort(response_obj: Any) -> str:
    def _extract_from_content_value(content_val: Any) -> str:
        if isinstance(content_val, str):
            return content_val
        if not isinstance(content_val, list):
            return ""

        parts: list[str] = []
        for item in content_val:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            txt = item.get("text")
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
                continue
            # Some SDK/providers nest text payloads under content blocks.
            maybe_nested = item.get("content")
            if isinstance(maybe_nested, str) and maybe_nested.strip():
                parts.append(maybe_nested)
        return "\n".join(p for p in parts if p).strip()

    try:
        choices = getattr(response_obj, "choices", None)
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    content_val = msg.get("content")
                else:
                    content_val = getattr(msg, "content", None)
            else:
                msg = getattr(first, "message", None)
                if isinstance(msg, dict):
                    content_val = msg.get("content")
                else:
                    content_val = getattr(msg, "content", None)
            out = _extract_from_content_value(content_val)
            if out:
                return out
    except Exception:
        pass

    if isinstance(response_obj, dict):
        try:
            choices = response_obj.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0] or {}
                if isinstance(first, dict):
                    msg = first.get("message")
                    if isinstance(msg, dict):
                        out = _extract_from_content_value(msg.get("content"))
                        if out:
                            return out
        except Exception:
            pass

    return ""


def _parse_json_object_best_effort(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}

    s = value.strip()
    if not s:
        return {}

    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_confidence(value: Any) -> str:
    v = (str(value or "").strip().lower())
    if v in ("high", "medium", "low"):
        return v
    if v in ("med", "mid"):
        return "medium"
    return ""


def _response_source_label(source: str, speaker_label: str = "") -> str:
    s = (source or "").strip().lower()
    if s in ("third_party", "third-party", "loopback"):
        return (speaker_label or "").strip() or "Third-Party"
    if s in ("user", "you"):
        return "You"
    return (source or "").strip() or "Unknown"


def _trim_excerpt(text: str, max_chars: int = 180) -> str:
    t = " ".join((text or "").split()).strip()
    if len(t) <= max_chars:
        return t
    return (t[: max(0, max_chars - 3)] + "...").strip()


def _cfg_float(value: Any, default: float, *, min_v: float | None = None, max_v: float | None = None) -> float:
    try:
        out = float(value if value is not None else default)
    except Exception:
        out = float(default)
    if min_v is not None:
        out = max(float(min_v), out)
    if max_v is not None:
        out = min(float(max_v), out)
    return out


def _cfg_int(value: Any, default: int, *, min_v: int | None = None, max_v: int | None = None) -> int:
    try:
        out = int(value if value is not None else default)
    except Exception:
        out = int(default)
    if min_v is not None:
        out = max(int(min_v), out)
    if max_v is not None:
        out = min(int(max_v), out)
    return out


def _response_signature(item: ResponseItem | dict) -> tuple:
    if isinstance(item, ResponseItem):
        d = asdict(item)
    else:
        d = dict(item or {})
    return (
        str(d.get("content") or "").strip(),
        str(d.get("responding_to_text") or "").strip(),
        str(d.get("responding_to_source") or "").strip(),
        str(d.get("responding_to_timestamp") or "").strip(),
        tuple(str(x).strip() for x in (d.get("basis_facts") or []) if str(x).strip()),
        tuple(str(x).strip() for x in (d.get("cautions") or []) if str(x).strip()),
        str(d.get("confidence") or "").strip().lower(),
    )


def _fact_check_signature(item: FactCheckItem | dict) -> tuple:
    if isinstance(item, FactCheckItem):
        d = asdict(item)
    else:
        d = dict(item or {})
    return (
        str(d.get("claim") or "").strip(),
        str(d.get("verdict") or "").strip().lower(),
        str(d.get("analysis") or "").strip(),
        tuple(str(x).strip() for x in (d.get("evidence") or []) if str(x).strip()),
        str(d.get("confidence") or "").strip().lower(),
    )


def _build_response_prompt() -> str:
    base = config.get("response_prompt", DEFAULT_CONFIG["response_prompt"])
    return f"""You are a strategic response coach for the user. {base}

Requirements:
- Ground advice in conversation facts and session context only.
- Keep tone calm, assertive, and legally cautious.
- Do not invent evidence, names, dates, or laws.
- Prefer short, defensible language the user can actually say out loud.
- The response must clearly reference the specific statement/event it addresses.
- Never create a response against the user; the user is your client.
- Never reply to the user. Only generate a response the user can use in the conversation.

Output ONLY valid JSON with this exact shape:
{{
  "response": "string",
  "basis_facts": ["string"],
  "cautions": ["string"],
  "confidence": "low|medium|high"
}}
"""


def _build_fact_check_prompt() -> str:
    base = config.get("fact_check_prompt", DEFAULT_CONFIG["fact_check_prompt"])
    return f"""You are a strict fact-check assistant. {base}

Task:
- Identify specific claims from the recent conversation.
- For each claim, decide verdict: supported, contradicted, or uncertain.
- Explain briefly using only transcript/context facts (and web results if present).
- If evidence is weak, say uncertain instead of guessing.
- If factual claims are present, return at least 1 check (do not return an empty list just because confidence is low).
- Prefer checking concrete, disputed statements first.

Output ONLY valid JSON with this exact shape:
{{
  "checks": [
    {{
      "claim": "string",
      "verdict": "supported|contradicted|uncertain",
      "analysis": "string",
      "evidence": ["string"],
      "confidence": "low|medium|high"
    }}
  ]
}}
"""


async def generate_response(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    *,
    response_lock: asyncio.Lock | None = None,
    seed_text: str = "",
    source: str = "user",
    target_timestamp: str = "",
    target_speaker: str = "",
    target_message_id: str = "",
) -> bool:
    if not config.get("ai_enabled", False):
        return False
    if not config.get("response_enabled", True):
        return False
    if llm_client is None or current_session is None:
        return False
    session = current_session
    session_id = str(getattr(session, "id", "") or "").strip()
    if not session_id:
        return False

    locked = False
    try:
        if response_lock is not None:
            await response_lock.acquire()
            locked = True

        context_size = _cfg_int(config.get("response_context_messages"), 14, min_v=0, max_v=5000)
        context_limit = None if context_size <= 0 else context_size
        context_max_chars = _cfg_int(
            config.get("response_context_max_chars"),
            int(DEFAULT_CONFIG["response_context_max_chars"]),
            min_v=4000,
            max_v=250000,
        )
        conversation_text = _format_recent_history_bounded(limit=context_limit, max_chars=context_max_chars)
        if not conversation_text.strip():
            return False

        web_seed = (seed_text or "").strip() or conversation_text[-500:]
        extra_messages: list[dict] = []
        web_ctx = await _maybe_build_web_search_context(purpose="response_generate", seed_text=web_seed)
        if web_ctx:
            extra_messages.append({"role": "system", "content": web_ctx})

        session_ctx = _get_session_context()
        if session_ctx:
            extra_messages.append({"role": "system", "content": f"Session context:\n{session_ctx[:4000]}"})

        payload = {
            "target_message_id": target_message_id,
            "new_message_source": source,
            "new_message_speaker": target_speaker,
            "new_message_timestamp": target_timestamp,
            "new_message": seed_text,
            "conversation": conversation_text,
        }

        request_messages = [
            {"role": "system", "content": _build_response_prompt()},
            *extra_messages,
            {"role": "user", "content": json.dumps(payload)},
        ]
        try:
            resp = await _llm_chat_create(
                messages=request_messages,
                stream=False,
                temperature=0.25,
                max_tokens=700,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            if not _looks_like_context_overflow_error(e):
                raise
            compact_max = max(4000, min(18000, int(context_max_chars * 0.5)))
            compact_text = _format_recent_history_bounded(limit=context_limit, max_chars=compact_max)
            if not compact_text or compact_text == conversation_text:
                raise
            payload["conversation"] = compact_text
            request_messages[-1] = {"role": "user", "content": json.dumps(payload)}
            resp = await _llm_chat_create(
                messages=request_messages,
                stream=False,
                temperature=0.25,
                max_tokens=700,
                response_format={"type": "json_object"},
            )

        data = _parse_json_object_best_effort(_extract_chat_content_best_effort(resp))

        content = (data.get("response") or "").strip()
        if not content:
            content = (data.get("recommended_response") or "").strip()
        content = " ".join(content.split())[:1400].strip()
        if not content:
            return False
        if not _is_active_session(session_id):
            return False

        source_label = _response_source_label(source, target_speaker)
        target_excerpt = _trim_excerpt(seed_text, max_chars=200)

        basis_facts = _coerce_string_list(data.get("basis_facts") or data.get("facts"))[:8]
        basis_facts = [" ".join(f.split())[:260].strip() for f in basis_facts if str(f).strip()]

        cautions = _coerce_string_list(data.get("cautions") or data.get("risk_notes"))[:6]
        cautions = [" ".join(c.split())[:260].strip() for c in cautions if str(c).strip()]

        confidence = _normalize_confidence(data.get("confidence"))

        item = ResponseItem(
            id=str(uuid.uuid4())[:8],
            content=content,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            responding_to_text=target_excerpt,
            responding_to_source=source_label,
            responding_to_speaker=(target_speaker or "").strip(),
            responding_to_timestamp=(target_timestamp or "").strip(),
            basis_facts=basis_facts,
            cautions=cautions,
            confidence=confidence,
            source="ai",
            session_id=session_id,
        )

        current = list(session.responses or [])
        if current and _response_signature(current[-1]) == _response_signature(item):
            return True
        if not current or (current[-1].content or "").strip().casefold() != content.casefold():
            current.append(item)
        else:
            # Replace the latest item to refresh facts/cautions if wording stayed the same.
            current[-1] = item

        max_items = int(config.get("response_max_items", 20) or 20)
        max_items = max(1, min(80, max_items))
        session.responses = current[-max_items:]
        save_session_throttled(session)
        if not _is_active_session(session_id):
            return False

        latest = session.responses[-1] if session.responses else None
        await _ws_send_json(
            websocket,
            {
                "type": "response_update",
                "responses": [asdict(r) for r in session.responses],
                "latest": asdict(latest) if latest else None,
            },
            send_lock,
        )
        log_important(
            "response.generated",
            source=source_label,
            confidence=(confidence or "-"),
            facts=len(basis_facts),
            cautions=len(cautions),
            total=len(session.responses),
        )
        return True
    except Exception as e:
        logger.error(f"Error generating response guidance: {e}")
        log_important("response.generate.error", level=logging.ERROR, error=e)
        return False
    finally:
        if locked and response_lock is not None and response_lock.locked():
            response_lock.release()


async def generate_fact_checks(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    *,
    fact_check_lock: asyncio.Lock | None = None,
) -> bool:
    if not config.get("ai_enabled", False):
        return False
    if not config.get("fact_check_enabled", True):
        return False
    if llm_client is None or current_session is None:
        return False
    session = current_session
    session_id = str(getattr(session, "id", "") or "").strip()
    if not session_id:
        return False

    locked = False
    try:
        if fact_check_lock is not None:
            await fact_check_lock.acquire()
            locked = True

        context_size = _cfg_int(config.get("fact_check_context_messages"), 18, min_v=0, max_v=5000)
        context_limit = None if context_size <= 0 else context_size
        context_max_chars = _cfg_int(
            config.get("fact_check_context_max_chars"),
            int(DEFAULT_CONFIG["fact_check_context_max_chars"]),
            min_v=4000,
            max_v=250000,
        )
        conversation_text = _format_recent_history_bounded(limit=context_limit, max_chars=context_max_chars)
        if not conversation_text.strip():
            if not _is_active_session(session_id):
                return False
            session.fact_checks = []
            save_session_throttled(session)
            await _ws_send_json(websocket, {"type": "fact_checks_update", "fact_checks": []}, send_lock)
            return True

        web_ctx = await _maybe_build_web_search_context(purpose="fact_check", seed_text=conversation_text[-500:])
        messages = [{"role": "system", "content": _build_fact_check_prompt()}]
        if web_ctx:
            messages.append({"role": "system", "content": web_ctx})

        session_ctx = _get_session_context()
        if session_ctx:
            messages.append({"role": "system", "content": f"Session context:\n{session_ctx[:4000]}"})
        messages.append({"role": "user", "content": conversation_text})

        try:
            resp = await _llm_chat_create(
                messages=messages,
                stream=False,
                temperature=0,
                max_tokens=900,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            if not _looks_like_context_overflow_error(e):
                raise
            compact_max = max(4000, min(20000, int(context_max_chars * 0.5)))
            compact_text = _format_recent_history_bounded(limit=context_limit, max_chars=compact_max)
            if not compact_text or compact_text == conversation_text:
                raise
            messages[-1] = {"role": "user", "content": compact_text}
            resp = await _llm_chat_create(
                messages=messages,
                stream=False,
                temperature=0,
                max_tokens=900,
                response_format={"type": "json_object"},
            )

        data = _parse_json_object_best_effort(_extract_chat_content_best_effort(resp))
        checks_raw = data.get("checks", [])
        if not isinstance(checks_raw, list):
            checks_raw = []

        normalized: list[FactCheckItem] = []
        max_items = int(config.get("fact_check_max_items", 12) or 12)
        max_items = max(1, min(50, max_items))
        for entry in checks_raw:
            if len(normalized) >= max_items:
                break
            if not isinstance(entry, dict):
                continue

            claim = " ".join(str(entry.get("claim") or "").split()).strip()[:280]
            if not claim:
                continue

            verdict = str(entry.get("verdict") or "").strip().lower()
            if verdict not in ("supported", "contradicted", "uncertain"):
                verdict = "uncertain"

            analysis = " ".join(str(entry.get("analysis") or "").split()).strip()[:520]
            if not analysis:
                analysis = "Insufficient evidence in available context."

            evidence = _coerce_string_list(entry.get("evidence"))[:6]
            evidence = [" ".join(ev.split())[:220].strip() for ev in evidence if str(ev).strip()]

            normalized.append(
                FactCheckItem(
                    id=str(uuid.uuid4())[:8],
                    claim=claim,
                    verdict=verdict,
                    analysis=analysis,
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                    evidence=evidence,
                    confidence=_normalize_confidence(entry.get("confidence")),
                    session_id=session_id,
                )
            )

        if not _is_active_session(session_id):
            return False

        existing = list(session.fact_checks or [])
        if len(existing) == len(normalized):
            same = True
            for i in range(len(normalized)):
                if _fact_check_signature(existing[i]) != _fact_check_signature(normalized[i]):
                    same = False
                    break
            if same:
                return True

        # Keep newest run as the current fact-check snapshot.
        session.fact_checks = normalized
        save_session_throttled(session)
        if not _is_active_session(session_id):
            return False
        await _ws_send_json(
            websocket,
            {"type": "fact_checks_update", "fact_checks": [asdict(fc) for fc in session.fact_checks]},
            send_lock,
        )
        supported = sum(1 for fc in normalized if (fc.verdict or "").lower() == "supported")
        contradicted = sum(1 for fc in normalized if (fc.verdict or "").lower() == "contradicted")
        uncertain = sum(1 for fc in normalized if (fc.verdict or "").lower() == "uncertain")
        log_important(
            "fact_checks.updated",
            total=len(normalized),
            supported=supported,
            contradicted=contradicted,
            uncertain=uncertain,
        )
        return True
    except Exception as e:
        logger.error(f"Error generating fact checks: {e}")
        log_important("fact_checks.error", level=logging.ERROR, error=e)
        return False
    finally:
        if locked and fact_check_lock is not None and fact_check_lock.locked():
            fact_check_lock.release()


# ============================================
# NOTES GENERATION (Separate AI Process)
# ============================================

def _build_notes_prompt() -> str:
    """Build the notes system prompt based on user settings."""
    base_prompt = config.get("notes_prompt", DEFAULT_CONFIG["notes_prompt"])
    
    categories = []
    if config.get("notes_extract_decisions", True):
        categories.append("decisions")
    if config.get("notes_extract_actions", True):
        categories.append("action_items")
    if config.get("notes_extract_risks", True):
        categories.append("risks")
    if config.get("notes_extract_facts", True):
        categories.append("key_facts")
    if not categories:
        categories = ["general context"]
    
    notes_format = config.get("notes_format", "bullets")
    
    if notes_format == "structured":
        return f"""You are a note-taking assistant. {base_prompt}

Return a JSON object with these keys (arrays of strings): {', '.join(categories)}.
Only include categories that have relevant content. Each item should be concise (1-2 sentences max)."""
    
    elif notes_format == "summary":
        return f"""You are a note-taking assistant. {base_prompt}

Return a JSON object with a single key "summary" containing a brief paragraph summarizing the key points.
Focus on: {', '.join(categories)}."""
    
    elif notes_format == "custom":
        return f"""You are a note-taking assistant. {base_prompt}

Return a JSON object with key "notes" as an array.
Each entry can be either:
- a string note, or
- an object: {{"content":"...", "category":"decision|action|risk|fact|general"}}

Keep notes concise and actionable. Focus on: {', '.join(categories)}."""
    
    else:  # bullets (default)
        return f"""You are a note-taking assistant. {base_prompt}

Return a JSON object with a single key "key_points" containing an array of bullet-point strings.
Each point should be concise (1-2 sentences max). Focus on: {', '.join(categories)}."""


def _build_smart_notes_prompt(*, target_count: int, max_count: int) -> str:
    return f"""You maintain a compact, high-signal notes list for an ongoing conversation.

You will be given:
- Transcript: recent conversation text
- Existing notes: the current AI notes list (may include redundancies)
- Protected notes: pinned/completed/manual notes that you must not modify (FYI only)

Task:
- Produce an updated AI notes list that is NOT flooded: merge duplicates, remove stale/irrelevant items, and summarize where appropriate.
- Prefer fewer, broader notes over many tiny ones; keep the most actionable/high-signal facts.
- Keep chronological details out unless critical; avoid repeating the transcript.
- Do not include any protected notes in your output.

Constraints:
- Aim for ~{target_count} notes, hard limit {max_count}.
- Each note should be concise (max ~200 chars), plain text (no markdown).
- Output ONLY valid JSON with this exact shape:
  {{\"notes\": [{{\"content\": \"...\", \"category\": \"decision|action|risk|fact|general\"}}]}}
"""


def _normalize_note_category(category: Any) -> str:
    raw = str(category or "").strip().lower().replace("-", "_").replace(" ", "_")
    category_map = {
        "decision": "decision",
        "decisions": "decision",
        "action": "action",
        "actions": "action",
        "action_item": "action",
        "action_items": "action",
        "risk": "risk",
        "risks": "risk",
        "fact": "fact",
        "facts": "fact",
        "key_fact": "fact",
        "key_facts": "fact",
        "general": "general",
        "summary": "general",
    }
    return category_map.get(raw, "general")


def _normalize_note_source(source: Any, *, default: str = "manual") -> str:
    raw = str(source or "").strip().lower()
    if raw == "ai":
        return "ai"
    if raw in ("manual", "user", "human"):
        return "manual"
    return default


def _normalize_note_content(content: Any, *, max_chars: int = 220) -> str:
    if content is None:
        return ""
    if not isinstance(content, str):
        content = str(content)
    out = " ".join(content.replace("\r", " ").replace("\n", " ").split()).strip()
    if not out:
        return ""
    return out[:max_chars].strip()


def _note_content_key(content: Any) -> str:
    return _normalize_note_content(content, max_chars=4000).casefold()


def _coerce_note_dict_list(value: Any, *, default_category: str = "general") -> list[dict]:
    out: list[dict] = []
    if value is None:
        return out

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                content = item.get("content")
                if not content:
                    content = item.get("text")
                if not content:
                    content = item.get("note")
                if not content:
                    content = item.get("summary")
                out.append(
                    {
                        "content": content,
                        "category": item.get("category", default_category),
                    }
                )
            else:
                out.append({"content": item, "category": default_category})
        return out

    if isinstance(value, dict):
        if "notes" in value:
            return _coerce_note_dict_list(value.get("notes"), default_category=default_category)
        return _coerce_note_dict_list([value], default_category=default_category)

    return [{"content": item, "category": default_category} for item in _coerce_string_list(value)]


def _extract_note_candidates_from_model_output(data: dict, *, notes_format: str) -> list[dict]:
    candidates: list[dict] = []
    if notes_format == "structured":
        cat_map = {
            "decisions": "decision",
            "action_items": "action",
            "risks": "risk",
            "key_facts": "fact",
        }
        for category in ("decisions", "action_items", "risks", "key_facts"):
            items = _coerce_string_list(data.get(category, []))
            for item in items:
                candidates.append({"content": item, "category": cat_map.get(category, "general")})
        return candidates

    if notes_format == "summary":
        summary = data.get("summary", "")
        if summary is not None and not isinstance(summary, str):
            summary = str(summary)
        summary = (summary or "").strip()
        if summary:
            candidates.append({"content": summary, "category": "general"})
        return candidates

    if notes_format == "custom":
        return _coerce_note_dict_list(data.get("notes", []), default_category="general")

    points = _coerce_string_list(data.get("key_points", []))
    if not points:
        # Fallback for models that return {"notes":[...]} even in bullets mode.
        return _coerce_note_dict_list(data.get("notes", []), default_category="general")
    return [{"content": p, "category": "general"} for p in points]


def _dedupe_note_dicts(notes: list[dict], *, max_items: int, seen_keys: set[str] | None = None) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set(seen_keys or set())
    for n in notes:
        if len(out) >= max_items:
            break
        if not isinstance(n, dict):
            continue
        content = _normalize_note_content(n.get("content"), max_chars=220)
        if not content:
            continue
        key = _note_content_key(content)
        if key in seen:
            continue
        seen.add(key)
        category = _normalize_note_category(n.get("category", "general"))
        out.append({"content": content, "category": category})
    return out


async def _smart_maintain_ai_notes(
    *,
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    conversation_text: str,
    existing_ai_notes: list[NoteItem],
    protected_notes: list[NoteItem],
    target_count: int,
    max_count: int,
) -> list[NoteItem]:
    if not llm_client:
        return []

    existing_payload = [{"content": n.content, "category": n.category} for n in existing_ai_notes if (n.content or "").strip()]
    protected_payload = [
        {"content": n.content, "source": n.source, "category": n.category, "pinned": n.pinned, "completed": n.completed}
        for n in protected_notes
        if (n.content or "").strip()
    ]

    system_prompt = _build_smart_notes_prompt(target_count=target_count, max_count=max_count)
    session_ctx = _get_session_context()
    if session_ctx:
        system_prompt = f"{system_prompt}\n\nSession context:\n{session_ctx}"

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "transcript": conversation_text,
                    "existing_notes": existing_payload,
                    "protected_notes": protected_payload,
                },
                ensure_ascii=False,
            ),
        },
    ]

    web_ctx = await _maybe_build_web_search_context(purpose="notes_maintain", seed_text=conversation_text)
    if web_ctx:
        messages.insert(1, {"role": "system", "content": web_ctx})

    response = await _llm_chat_create(
        messages=messages,
        stream=False,
        response_format={"type": "json_object"},
    )

    content = _extract_chat_content_best_effort(response)
    data = _parse_json_object_best_effort(content)
    notes_raw = data.get("notes", [])
    cleaned = _dedupe_note_dicts(
        _coerce_note_dict_list(notes_raw, default_category="general"),
        max_items=max_count,
    )
    timestamp = datetime.now().strftime("%H:%M:%S")
    new_notes: list[NoteItem] = []
    for item in cleaned:
        note = NoteItem(
            id=str(uuid.uuid4())[:8],
            content=item["content"],
            timestamp=timestamp,
            source="ai",
            category=item.get("category", "general") or "general",
            session_id=current_session.id if current_session else None,
        )
        new_notes.append(note)

    return new_notes


async def generate_notes(websocket: WebSocket, send_lock: asyncio.Lock, notes_lock: asyncio.Lock | None = None) -> bool:
    """Generate notes from recent conversation. Returns True if successful."""
    if not config.get("ai_enabled", False):
        return False
    if not llm_client or not current_session:
        return False
    
    if len(llm_client.history) == 0:
        return False
    
    try:
        if notes_lock is not None:
            await notes_lock.acquire()

        context_size = _cfg_int(config.get("notes_context_messages"), 10, min_v=0, max_v=5000)
        max_chars = int(config.get("notes_max_chars", 24000) or 24000)
        conversation_text = _format_recent_history_bounded(limit=context_size, max_chars=max_chars)
        if not conversation_text:
            return False

        smart_enabled = bool(config.get("notes_smart_enabled", True))
        if smart_enabled:
            try:
                target = int(config.get("notes_smart_target_ai_notes", 12) or 12)
            except Exception:
                target = 12
            target = max(3, min(50, target))

            try:
                max_count = int(config.get("notes_smart_max_ai_notes", 18) or 18)
            except Exception:
                max_count = 18
            max_count = max(3, min(80, max_count))
            if max_count < target:
                max_count = target

            protected = [
                n
                for n in (current_session.notes or [])
                if getattr(n, "source", None) != "ai" or bool(getattr(n, "pinned", False)) or bool(getattr(n, "completed", False))
            ]
            existing_ai = [
                n
                for n in (current_session.notes or [])
                if getattr(n, "source", None) == "ai" and not bool(getattr(n, "pinned", False)) and not bool(getattr(n, "completed", False))
            ]

            try:
                maintained = await _smart_maintain_ai_notes(
                    websocket=websocket,
                    send_lock=send_lock,
                    conversation_text=conversation_text,
                    existing_ai_notes=existing_ai,
                    protected_notes=protected,
                    target_count=target,
                    max_count=max_count,
                )
            except Exception as e:
                logger.error(f"Error generating smart notes: {e}")
                log_important("notes.smart.error", level=logging.ERROR, error=e)
                maintained = []

            if not maintained and existing_ai:
                # If model output is empty/invalid, keep existing AI notes rather than wiping.
                fallback: list[NoteItem] = []
                seen_existing: set[str] = set()
                for n in existing_ai:
                    key = _note_content_key(getattr(n, "content", ""))
                    if not key or key in seen_existing:
                        continue
                    seen_existing.add(key)
                    fallback.append(n)
                    if len(fallback) >= max_count:
                        break
                maintained = fallback

            before = [asdict(n) for n in (current_session.notes or [])]
            current_session.notes = list(protected) + list(maintained)
            save_session(current_session)

            after = [asdict(n) for n in (current_session.notes or [])]
            if after != before:
                await _ws_send_json(
                    websocket,
                    {"type": "notes_update", "notes": after, "new_notes": [asdict(n) for n in maintained]},
                    send_lock,
                )
                log_important(
                    "notes.updated",
                    mode="smart",
                    protected=len(protected),
                    ai=len(maintained),
                    total=len(current_session.notes),
                )
                return True
            return False
        
        session_ctx = _get_session_context()
        system_prompt = _build_notes_prompt()
        if session_ctx:
            system_prompt = f"{system_prompt}\n\nSession context:\n{session_ctx}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Transcript:\n{conversation_text}"},
        ]

        web_ctx = await _maybe_build_web_search_context(purpose="notes_generate", seed_text=conversation_text)
        if web_ctx:
            messages.insert(1, {"role": "system", "content": web_ctx})
        
        response = await _llm_chat_create(
            messages=messages,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        content = _extract_chat_content_best_effort(response)
        data = _parse_json_object_best_effort(content)
        
        notes_format = str(config.get("notes_format", "bullets") or "bullets").strip().lower()
        candidates = _extract_note_candidates_from_model_output(data, notes_format=notes_format)
        existing_keys = {
            _note_content_key(getattr(n, "content", ""))
            for n in (current_session.notes or [])
            if _normalize_note_content(getattr(n, "content", ""), max_chars=4000)
        }
        cleaned = _dedupe_note_dicts(candidates, max_items=60, seen_keys=existing_keys)
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_notes = [
            NoteItem(
                id=str(uuid.uuid4())[:8],
                content=item["content"],
                timestamp=timestamp,
                source="ai",
                category=item.get("category", "general") or "general",
                session_id=current_session.id,
            )
            for item in cleaned
        ]
        
        if new_notes:
            current_session.notes.extend(new_notes)
            save_session(current_session)
            
            # Send update to client
            await _ws_send_json(
                websocket,
                {
                    "type": "notes_update",
                    "notes": [asdict(n) for n in current_session.notes],
                    "new_notes": [asdict(n) for n in new_notes],
                },
                send_lock,
            )
            log_important(
                "notes.updated",
                mode="standard",
                added=len(new_notes),
                total=len(current_session.notes),
            )
            return True
        
    except Exception as e:
        logger.error(f"Error generating notes: {e}")
        log_important("notes.error", level=logging.ERROR, error=e)
    finally:
        if notes_lock is not None and notes_lock.locked():
            notes_lock.release()
    
    return False


def _label_from_history_message(message: dict) -> str:
    role = (message.get("role") or "").strip().lower()
    if role == "assistant":
        return "AI"
    name_raw = message.get("name") or ""
    name = name_raw.strip() if isinstance(name_raw, str) else str(name_raw).strip()
    name_low = name.lower()
    if name_low.startswith("third_party_") and name_low != "third_party":
        suffix = name[len("third_party_") :]
        if "__" in suffix:
            label_part = suffix.split("__", 1)[0]
            label = label_part.replace("_", " ").strip()
            if label:
                return label
        if suffix.lower().startswith("spk"):
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                return f"Speaker {digits}"
    if name_low.startswith("third_party_spk"):
        digits = "".join(ch for ch in name_low if ch.isdigit())
        if digits:
            return f"Speaker {digits}"
        return "Third-Party"
    if name_low == "third_party":
        return "Third-Party"
    if name_low == "you":
        return "You"
    return "User"


def _history_lines_for_context(*, limit: int | None = None) -> list[str]:
    if not llm_client:
        return []
    history = list(getattr(llm_client, "history", []) or [])
    if limit is not None:
        history = history[-int(limit) :]

    lines: list[str] = []
    for m in history:
        if not isinstance(m, dict):
            continue
        content = (m.get("content") or "")
        if not isinstance(content, str):
            content = str(content)
        content = " ".join(content.split()).strip()
        if not content:
            continue
        lines.append(f"{_label_from_history_message(m)}: {content}")
    return lines


def _tail_lines_within_chars(lines: list[str], budget_chars: int) -> list[str]:
    if budget_chars <= 0:
        return []
    out_rev: list[str] = []
    used = 0
    for line in reversed(lines):
        line_len = len(line) + (1 if out_rev else 0)
        if out_rev and used + line_len > budget_chars:
            break
        if (not out_rev) and len(line) > budget_chars:
            keep = max(0, budget_chars - len(" ...[truncated]"))
            out_rev.append((line[:keep] + " ...[truncated]").strip())
            break
        out_rev.append(line)
        used += line_len
    return list(reversed(out_rev))


def _get_pytextrank_nlp() -> Any:
    global _PYTEXTRANK_NLP, _PYTEXTRANK_INIT_ATTEMPTED
    if _PYTEXTRANK_INIT_ATTEMPTED:
        return _PYTEXTRANK_NLP
    _PYTEXTRANK_INIT_ATTEMPTED = True

    try:
        import spacy
    except Exception as e:
        logger.debug("PyTextRank unavailable (spacy import failed): %s", e)
        return None

    nlp = None
    for model_name in ("en_core_web_sm", "en_core_web_md"):
        try:
            nlp = spacy.load(model_name)
            break
        except Exception:
            continue

    if nlp is None:
        try:
            # Keep a best-effort fallback path if no pre-trained model is installed.
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")
        except Exception as e:
            logger.debug("PyTextRank unavailable (failed to init spaCy pipeline): %s", e)
            return None

    try:
        import pytextrank  # noqa: F401
        if "textrank" not in nlp.pipe_names:
            nlp.add_pipe("textrank")
    except Exception as e:
        logger.debug("PyTextRank unavailable (component init failed): %s", e)
        return None

    _PYTEXTRANK_NLP = nlp
    return _PYTEXTRANK_NLP


def _build_local_context_summary_pytextrank(
    older_lines: list[str],
    *,
    max_chars: int,
    max_points: int = 14,
) -> str:
    if not older_lines or max_chars <= 0:
        return ""

    nlp = _get_pytextrank_nlp()
    if nlp is None:
        return ""

    text = "\n".join(" ".join((ln or "").split()).strip() for ln in older_lines if str(ln or "").strip())
    if not text:
        return ""

    try:
        doc = nlp(text)
        spans = list(
            doc._.textrank.summary(  # type: ignore[attr-defined]
                limit_phrases=max(20, int(max_points) * 2),
                limit_sentences=max(1, int(max_points)),
            )
        )
    except Exception:
        return ""

    points: list[str] = []
    seen: set[str] = set()
    for sp in spans:
        s = " ".join(str(sp or "").split()).strip()
        if len(s) < 20:
            continue
        sig = s.casefold()
        if sig in seen:
            continue
        seen.add(sig)
        points.append(s[:240])
        if len(points) >= max(1, int(max_points)):
            break

    if not points:
        return ""

    summary_lines = [f"Earlier turns summarized: {len(older_lines)}", "Key points:"]
    for p in points:
        summary_lines.append(f"- {p}")

    out: list[str] = []
    used = 0
    for line in summary_lines:
        add = line if not out else ("\n" + line)
        if used + len(add) > max_chars:
            break
        out.append(line)
        used += len(add)

    return "\n".join(out).strip()


def _build_local_context_summary_heuristic(
    older_lines: list[str],
    *,
    max_chars: int,
    max_points: int = 14,
) -> str:
    import re

    if not older_lines or max_chars <= 0:
        return ""

    speaker_counts: dict[str, int] = {}
    points: list[tuple[int, int, str, str]] = []
    seen: set[str] = set()
    keyword_re = re.compile(
        r"\b("
        r"agree|agreed|decision|decide|plan|action|next step|deadline|must|should|need to|"
        r"because|evidence|claim|risk|issue|problem|request|policy|law|contract|payment|date"
        r")\b",
        flags=re.I,
    )

    for idx, raw in enumerate(older_lines):
        line = " ".join((raw or "").split()).strip()
        if not line:
            continue
        if ":" in line:
            speaker, body = line.split(":", 1)
            speaker = speaker.strip() or "User"
            body = body.strip()
        else:
            speaker = "User"
            body = line
        if not body:
            continue
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

        sentences = re.split(r"(?<=[.!?])\s+", body)
        if not sentences:
            sentences = [body]
        for sent in sentences:
            s = " ".join(sent.split()).strip(" -")
            if len(s) < 24:
                continue
            score = 0
            if any(ch.isdigit() for ch in s):
                score += 2
            if "?" in s:
                score += 1
            if keyword_re.search(s):
                score += 2
            if len(s) >= 120:
                score += 1
            if score <= 0:
                continue

            sig = re.sub(r"[^a-z0-9 ]+", "", s.casefold()).strip()
            if not sig or sig in seen:
                continue
            seen.add(sig)
            points.append((score, idx, speaker, s[:220]))

    if not points:
        for idx, raw in enumerate(older_lines[-6:]):
            line = " ".join((raw or "").split()).strip()
            if not line:
                continue
            points.append((1, idx, "Context", line[:180]))

    selected = sorted(points, key=lambda x: (-x[0], -x[1]))[: max(1, int(max_points))]
    selected.sort(key=lambda x: x[1])

    speaker_meta = ", ".join(f"{k} {v}" for k, v in sorted(speaker_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4])
    summary_lines = [f"Earlier turns summarized: {len(older_lines)}"]
    if speaker_meta:
        summary_lines.append(f"Speaker activity: {speaker_meta}")
    summary_lines.append("Key points:")
    for _, _, speaker, text in selected:
        summary_lines.append(f"- {speaker}: {text}")

    out: list[str] = []
    used = 0
    for line in summary_lines:
        add = line if not out else ("\n" + line)
        if used + len(add) > max_chars:
            break
        out.append(line)
        used += len(add)

    return "\n".join(out).strip()


def _build_local_context_summary(
    older_lines: list[str],
    *,
    max_chars: int,
    max_points: int = 14,
) -> str:
    method = str(config.get("context_local_summary_method", DEFAULT_CONFIG["context_local_summary_method"]) or "").strip().lower()
    if method not in ("pytextrank", "heuristic", "auto"):
        method = str(DEFAULT_CONFIG["context_local_summary_method"])

    if method in ("pytextrank", "auto"):
        out = _build_local_context_summary_pytextrank(
            older_lines,
            max_chars=max_chars,
            max_points=max_points,
        )
        if out:
            return out

    return _build_local_context_summary_heuristic(
        older_lines,
        max_chars=max_chars,
        max_points=max_points,
    )


def _format_recent_history_bounded(*, limit: int | None = 10, max_chars: int = 24000) -> str:
    if not llm_client:
        return ""

    try:
        max_chars = int(max_chars)
    except Exception:
        max_chars = 24000
    max_chars = max(2000, min(250000, max_chars))

    try:
        limit_n = int(limit) if limit is not None else None
    except Exception:
        limit_n = 10
    if limit_n is not None:
        # 0 or negative means "all available messages".
        if limit_n <= 0:
            limit_n = None
        else:
            limit_n = max(1, min(5000, limit_n))

    lines = _history_lines_for_context(limit=limit_n)
    if not lines:
        return ""

    full = "\n".join(lines).strip()
    if len(full) <= max_chars:
        return full

    # Fallback mode: keep newest verbatim lines when local summary is disabled.
    if not bool(config.get("context_local_summary_enabled", True)):
        return "\n".join(_tail_lines_within_chars(lines, max_chars)).strip()

    tail_budget = max(1200, min(max_chars - 700, int(max_chars * 0.68)))
    recent_lines = _tail_lines_within_chars(lines, tail_budget)
    if not recent_lines:
        recent_lines = _tail_lines_within_chars(lines, max_chars)
    recent_text = "\n".join(recent_lines).strip()
    if not recent_text:
        return ""

    older_count = max(0, len(lines) - len(recent_lines))
    if older_count <= 0:
        return recent_text[:max_chars].strip()

    older_lines = lines[:older_count]
    summary_budget = max(450, max_chars - len(recent_text) - 80)
    summary = _build_local_context_summary(older_lines, max_chars=summary_budget)
    if not summary:
        return recent_text[:max_chars].strip()

    composed = (
        "Earlier context summary (local, non-LLM):\n"
        f"{summary}\n\n"
        "Recent context (verbatim):\n"
        f"{recent_text}"
    ).strip()
    if len(composed) <= max_chars:
        return composed

    return composed[:max_chars].rstrip()


def _get_session_context(*, max_chars: int = 4000) -> str:
    if not current_session:
        return ""
    ctx = current_session.context if isinstance(current_session.context, str) else str(current_session.context or "")
    ctx = ctx.strip()
    if not ctx:
        return ""
    try:
        max_chars = int(max_chars)
    except Exception:
        max_chars = 4000
    max_chars = max(500, min(20000, max_chars))
    return ctx[:max_chars].strip()


def _coerce_string_list(value: Any) -> list[str]:
    """Best-effort normalize model output into a list of strings (avoid iterating over characters)."""
    if value is None:
        return []

    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif item is not None:
                s = str(item).strip()
                if s:
                    out.append(s)
        return out

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []

        # Sometimes the model returns a JSON-encoded string; try to decode it.
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")) or (s.startswith('"') and s.endswith('"')):
            try:
                decoded = json.loads(s)
                if decoded is not value:
                    return _coerce_string_list(decoded)
            except Exception:
                pass

        # Otherwise, treat as a multi-line/bulleted blob.
        lines: list[str] = []
        for line in s.splitlines():
            ln = line.strip()
            if not ln:
                continue
            ln = ln.lstrip("-* \t").strip()
            if ln:
                lines.append(ln)
        return lines or [s]

    # Fallback: single item.
    s = str(value).strip()
    return [s] if s else []


async def run_notes_loop(websocket: WebSocket, send_lock: asyncio.Lock, notes_lock: asyncio.Lock):
    """Periodically generates notes from conversation history."""
    logger.info("Starting notes generation loop")
    last_generated_activity_seq = -1

    while True:
        interval = _cfg_int(config.get("notes_interval_seconds"), 30, min_v=5, max_v=600)
        await asyncio.sleep(interval)
        
        if not config.get("ai_enabled", False):
            continue

        if not config.get("notes_enabled", True):
            continue

        if config.get("notes_on_interaction_only", False):
            continue

        current_seq = int(_TRANSCRIPT_ACTIVITY_SEQ)
        if current_seq == last_generated_activity_seq:
            continue

        await _wait_for_transcript_settle()

        ok = await generate_notes(websocket, send_lock, notes_lock)
        if ok:
            last_generated_activity_seq = current_seq


async def run_fact_check_loop(websocket: WebSocket, send_lock: asyncio.Lock, fact_check_lock: asyncio.Lock):
    """Periodically generates fact checks from conversation history."""
    logger.info("Starting fact check loop")
    last_generated_activity_seq = -1

    while True:
        interval = _cfg_int(config.get("fact_check_interval_seconds"), 15, min_v=5, max_v=600)
        await asyncio.sleep(interval)

        if not config.get("ai_enabled", False):
            continue

        if not config.get("fact_check_enabled", True):
            continue

        if config.get("fact_check_on_interaction_only", False):
            continue

        current_seq = int(_TRANSCRIPT_ACTIVITY_SEQ)
        if current_seq == last_generated_activity_seq:
            continue

        await _wait_for_transcript_settle()

        ok = await generate_fact_checks(websocket, send_lock, fact_check_lock=fact_check_lock)
        if ok:
            last_generated_activity_seq = current_seq


# ============================================
# TRANSCRIPTION & LLM PROCESSING
# ============================================

async def run_transcription_loop(
    websocket: WebSocket,
    source: str,
    state: ConnectionState,
    send_lock: asyncio.Lock,
    response_queue: asyncio.Queue,
    notes_queue: asyncio.Queue,
    fact_check_queue: asyncio.Queue,
    transcript_cleanup_queue: asyncio.Queue | None = None,
):
    """
    Consumes audio from audio_manager, runs transcription, sends to WS.
    Uses a shared transcription model instance (calls are internally serialized).
    """
    logger.info(f"Starting transcription loop for {source}")
    log_important("transcription.loop.start", source=source)
    source_label = "microphone" if source == "user" else ("system audio" if source in ("third_party", "loopback") else source)
    if not await _ws_send_json(
        websocket,
        {"type": "status", "message": f"Audio: transcribing from {source_label}."},
        send_lock,
    ):
        return
    
    def transcription_worker():
        if transcription_model is None:
            logger.error("Transcription model not initialized!")
            log_important("transcription.loop.error", level=logging.ERROR, source=source, reason="model-not-initialized")
            return

        diarize = bool(config.get("speaker_diarization_enabled", False)) and source in ("third_party", "loopback")
        global third_party_diarizer
        if diarize and third_party_diarizer is None:
            third_party_diarizer = SpeakerDiarizer()

        # Route audio based on source.
        if source == "user":
            audio_gen = audio_manager.get_mic_data()
        elif source in ("third_party", "loopback"):
            audio_gen = audio_manager.get_loopback_data()
        else:
            logger.warning(f"Unknown transcription source '{source}', defaulting to mic")
            audio_gen = audio_manager.get_mic_data()
        gen = transcription_model.transcribe_stream(audio_gen, return_voiceprint=diarize)
        for item in gen:
            speaker_id = None
            speaker_label = None
            text = item
            if isinstance(item, TranscriptionChunk):
                text = item.text
                if diarize and third_party_diarizer is not None:
                    try:
                        speaker_id, speaker_label = third_party_diarizer.assign(item.voiceprint)
                    except Exception:
                        speaker_id = None
                        speaker_label = None
            asyncio.run_coroutine_threadsafe(
                process_transcription(
                    websocket,
                    text,
                    source,
                    state,
                    send_lock,
                    message_kind="audio",
                    response_queue=response_queue,
                    notes_queue=notes_queue,
                    fact_check_queue=fact_check_queue,
                    transcript_cleanup_queue=transcript_cleanup_queue,
                    speaker_id=speaker_id,
                    speaker_label=speaker_label,
                ),
                loop,
            )

    loop = asyncio.get_running_loop()
    threading.Thread(target=transcription_worker, daemon=True).start()


def _should_merge_transcript(prev_text: str, new_text: str) -> bool:
    a = (prev_text or "").strip()
    b = (new_text or "").strip()
    if not a or not b:
        return False

    if b.startswith(("#", "@", "http://", "https://")):
        return False

    # Whisper often emits trailing "..." when speech is cut off mid-flow by
    # the chunk boundary.  This is a very strong continuation signal.
    if a.endswith("..."):
        return True

    if a and a[-1] not in ".?!":
        return True

    if b[0].islower():
        return True

    starters = (
        "and ",
        "but ",
        "so ",
        "to ",
        "because ",
        "also ",
        "then ",
    )
    b_cf = b.casefold()
    if any(b_cf.startswith(s) for s in starters):
        return True

    if len(b) <= 18:
        return True

    return False


def _is_strong_transcript_continuation(prev_text: str, new_text: str) -> bool:
    a = (prev_text or "").strip()
    b = (new_text or "").strip()
    if not a or not b:
        return False

    if b.startswith(("#", "@", "http://", "https://")):
        return False

    # Whisper trailing "..." means the chunk was cut mid-speech flow.
    if a.endswith("..."):
        return True

    if a and a[-1] not in ".?!":
        return True

    if a and a[-1] in "-,:;":
        return True

    if b and b[0].islower():
        return True

    starters = (
        "and ",
        "but ",
        "so ",
        "to ",
        "because ",
        "also ",
        "then ",
        "or ",
        "if ",
    )
    b_cf = b.casefold()
    return any(b_cf.startswith(s) for s in starters)


def _strip_prefix_word_tokens(text: str, token_count: int) -> str:
    import re

    if token_count <= 0:
        return text

    matches = list(re.finditer(r"[a-z0-9']+", text, flags=re.I))
    if token_count > len(matches):
        return text

    cut = matches[token_count - 1].end()
    remainder = text[cut:]
    return re.sub(r"^[\s,;:\-.!?]+", "", remainder)


def _merge_transcript_text(prev_text: str, new_text: str) -> str:
    import re

    a = (prev_text or "").rstrip()
    b = (new_text or "").lstrip()
    if not a:
        return b
    if not b:
        return a

    # Handle chopped word boundaries like "will-" + "will be".
    if a.endswith("-"):
        a = a[:-1].rstrip()

    a_tokens = re.findall(r"[a-z0-9']+", a.casefold(), flags=re.I)
    b_tokens = re.findall(r"[a-z0-9']+", b.casefold(), flags=re.I)
    max_overlap = min(8, len(a_tokens), len(b_tokens))
    overlap = 0
    for n in range(max_overlap, 1, -1):
        if a_tokens[-n:] == b_tokens[:n]:
            overlap = n
            break

    if overlap > 0:
        trimmed = _strip_prefix_word_tokens(b, overlap).strip()
        if trimmed:
            b = trimmed
        else:
            b = ""
    elif a_tokens and b_tokens:
        # Conservative 1-token seam dedupe (e.g. "cars. cars", "everyone Everyone").
        tail = a_tokens[-1]
        head = b_tokens[0]
        if tail == head and len(tail) >= 3:
            trimmed = _strip_prefix_word_tokens(b, 1).strip()
            if trimmed:
                b = trimmed
        elif len(head) >= 4 and tail.endswith(head) and tail != head:
            a = re.sub(r"[.?!]\s*$", "", a).rstrip()
            trimmed = _strip_prefix_word_tokens(b, 1).strip()
            if trimmed:
                b = trimmed
        elif (
            len(tail) >= 2
            and len(tail) <= 4
            and len(head) >= 4
            and head.startswith(tail)
            and tail != head
            and (len(head) - len(tail)) <= 8
        ):
            # Chopped stem seam: "by. biased" -> "biased"
            a = re.sub(rf"\b{re.escape(tail)}[.?!]\s*$", "", a, flags=re.I).rstrip()
        elif (
            tail in {"at", "by", "for", "from", "in", "of", "on", "with"}
            and len(head) >= 4
            and head[:1] == tail[:1]
        ):
            # Orphan connector seam: "by. biased" -> "biased"
            a = re.sub(rf"\b{re.escape(tail)}[.?!]\s*$", "", a, flags=re.I).rstrip()

    if not b:
        return a

    merged = (a + " " + b).strip()
    return re.sub(r"\s+([,.;:!?])", r"\1", merged)


def _count_word_tokens(text: str) -> int:
    import re

    t = " ".join((text or "").split()).strip()
    if not t:
        return 0
    return len(re.findall(r"[a-z0-9']+", t, flags=re.I))


def _transcript_append_delta(reference_text: str, current_text: str) -> str:
    import re

    ref = " ".join((reference_text or "").split()).strip()
    cur = " ".join((current_text or "").split()).strip()
    if not cur:
        return ""
    if not ref:
        return cur
    if cur == ref:
        return ""
    if cur.startswith(ref):
        return cur[len(ref) :].strip(" \t\r\n,;:.!?-")

    a_tokens = re.findall(r"[a-z0-9']+", ref.casefold(), flags=re.I)
    b_tokens = re.findall(r"[a-z0-9']+", cur.casefold(), flags=re.I)
    if not a_tokens or not b_tokens:
        return cur

    max_overlap = min(12, len(a_tokens), len(b_tokens))
    overlap = 0
    for n in range(max_overlap, 1, -1):
        if a_tokens[-n:] == b_tokens[:n]:
            overlap = n
            break

    if overlap > 0:
        trimmed = _strip_prefix_word_tokens(cur, overlap).strip()
        if trimmed:
            return trimmed

    return cur


def _prune_ai_trigger_baseline_cache(max_items: int = 4096) -> None:
    max_n = int(max(128, max_items))
    if len(_AI_TRIGGER_BASELINE_TEXT) <= max_n:
        return
    to_remove = len(_AI_TRIGGER_BASELINE_TEXT) - max_n
    for k in list(_AI_TRIGGER_BASELINE_TEXT.keys())[:to_remove]:
        _AI_TRIGGER_BASELINE_TEXT.pop(k, None)


def _should_enqueue_ai_from_transcript_update(
    feature: str,
    *,
    message_id: str,
    current_text: str,
    previous_text: str = "",
    merged: bool = False,
    message_kind: str = "audio",
) -> bool:
    kind = (message_kind or "audio").strip().lower()
    cur = " ".join((current_text or "").split()).strip()
    if not cur:
        return False

    sid = "-"
    if current_session is not None:
        try:
            sid = (current_session.id or "").strip() or "-"
        except Exception:
            sid = "-"
    mid = (message_id or "").strip() or "-"
    key = f"{sid}:{feature}:{mid}"

    # Manual triggers should stay immediate and explicit.
    if kind == "manual":
        _AI_TRIGGER_BASELINE_TEXT[key] = cur
        _prune_ai_trigger_baseline_cache()
        return True

    baseline = " ".join((_AI_TRIGGER_BASELINE_TEXT.get(key, "") or "").split()).strip()
    prev = " ".join((previous_text or "").split()).strip()
    reference = baseline or (prev if merged else "")
    delta = _transcript_append_delta(reference, cur) if reference else cur
    delta = " ".join((delta or "").split()).strip()
    if not delta:
        return False

    ends_sentence = cur.endswith((".", "?", "!"))
    delta_chars = len(delta)
    delta_words = _count_word_tokens(delta)
    full_chars = len(cur)
    full_words = _count_word_tokens(cur)
    is_cleanup = (feature == "transcript_cleanup")

    # New transcript chunks need stronger evidence before triggering downstream AI.
    if is_cleanup and not reference:
        substantive = (
            full_chars >= 18
            or full_words >= 3
            or (full_words >= 2 and ends_sentence)
        )
    elif not reference:
        substantive = (
            full_chars >= 42
            or full_words >= 8
            or (full_words >= 5 and ends_sentence)
        )
    elif is_cleanup:
        substantive = (
            delta_chars >= 10
            or delta_words >= 2
            or (delta_words >= 1 and ends_sentence)
        )
    else:
        substantive = (
            delta_chars >= 28
            or delta_words >= 6
            or (delta_words >= 4 and ends_sentence)
        )
        if not substantive and feature in ("response", "fact_check"):
            # Response/fact-check generation should not starve when STT appends tiny deltas.
            # If the cumulative transcript became substantive, allow one trigger.
            non_trivial_delta = (
                delta_chars >= 14
                or delta_words >= 3
                or (delta_words >= 2 and ends_sentence)
            )
            if non_trivial_delta:
                substantive = (
                    full_chars >= 56
                    or full_words >= 10
                    or (full_words >= 6 and ends_sentence)
                )

    if not substantive and feature in ("response", "fact_check", "notes"):
        # On merged STT updates, avoid starvation when growth happens in very small chunks.
        # If this message has never triggered before and is now clearly substantive,
        # allow one trigger even when the latest delta is tiny.
        if merged and not baseline:
            substantive = (
                full_chars >= 56
                or full_words >= 10
                or (full_words >= 6 and ends_sentence)
            )
    if not substantive and is_cleanup:
        # Cleanup/paraphrase should keep up with merged incremental STT chunks.
        if merged and not baseline:
            substantive = (
                full_chars >= 30
                or full_words >= 5
                or (full_words >= 4 and ends_sentence)
            )

    if not substantive:
        # Keep an initial anchor for this message so future merged deltas can be
        # evaluated cumulatively instead of only against the immediately prior chunk.
        if kind != "manual" and not baseline:
            _AI_TRIGGER_BASELINE_TEXT[key] = (prev or cur)
            _prune_ai_trigger_baseline_cache()
        return False

    _AI_TRIGGER_BASELINE_TEXT[key] = cur
    _prune_ai_trigger_baseline_cache()
    return True


async def _transcript_cleanup_text(text: str) -> str:
    t = " ".join((text or "").split()).strip()
    if not t:
        return ""
    fallback = _local_transcript_rewrite_fallback(t, mode="cleanup")

    if not llm_client:
        return fallback

    system_prompt = (
        "You clean up speech-to-text transcript fragments for readability.\n"
        "Rules:\n"
        "- Do NOT add new facts or change meaning.\n"
        "- Keep the same intent (questions stay questions).\n"
        "- Keep pronouns and references; do not invent names.\n"
        "- Fix casing/punctuation and remove obvious stutters/repeats.\n"
        "- Output ONLY valid JSON: {\"clean_text\": \"...\"}\n"
    )

    resp = await _llm_chat_create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": t[:1200]},
        ],
        stream=False,
        temperature=0,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    out = _extract_transcript_rewrite_text(resp, fallback_input=t)
    return out or fallback


def _get_transcript_ai_mode(cfg: dict) -> str:
    mode = (cfg.get("transcript_ai_mode") or "").strip().lower()
    if mode in ("off", "cleanup", "paraphrase"):
        return mode

    # Back-compat: older boolean setting.
    if bool(cfg.get("transcript_ai_cleanup_enabled", False)):
        return "cleanup"

    return "off"


async def _transcript_ai_rewrite(text: str, *, mode: str) -> str:
    """
    mode:
      - cleanup: punctuation/casing/fluency, no re-interpretation
      - paraphrase: rewrite into what the speaker likely meant, but preserve meaning and don't add facts
    """
    if mode == "cleanup":
        return await _transcript_cleanup_text(text)

    if mode != "paraphrase":
        return ""

    t = " ".join((text or "").split()).strip()
    if not t:
        return ""
    fallback = _local_transcript_rewrite_fallback(t, mode="paraphrase")

    if not llm_client:
        return fallback

    system_prompt = (
        "You rewrite speech-to-text transcript fragments into a clearer paraphrase of what the speaker likely meant.\n"
        "Rules:\n"
        "- Preserve meaning; do NOT add new facts.\n"
        "- If ambiguous, keep it ambiguous (do not guess specifics).\n"
        "- Keep perspective (I/you/they) and intent (questions remain questions).\n"
        "- Output ONLY valid JSON: {\"clean_text\": \"...\"}\n"
    )

    resp = await _llm_chat_create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": t[:1200]},
        ],
        stream=False,
        temperature=0.2,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    out = _extract_transcript_rewrite_text(resp, fallback_input=t)
    return out or fallback


def _local_transcript_rewrite_fallback(text: str, *, mode: str) -> str:
    import re

    t = " ".join((text or "").split()).strip()
    if not t:
        return ""

    # Remove obvious adjacent duplicated words from STT seams.
    words = t.split(" ")
    out_words: list[str] = []
    prev_norm = ""
    for w in words:
        norm = re.sub(r"[^a-z0-9']+", "", (w or "").casefold())
        if norm and norm == prev_norm:
            continue
        out_words.append(w)
        if norm:
            prev_norm = norm
    s = " ".join(out_words).strip()

    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = s.strip()
    if not s:
        return ""

    # Conservative readability touch only (no paraphrase/fact changes).
    first_alpha = re.search(r"[A-Za-z]", s)
    if first_alpha:
        i = first_alpha.start()
        s = s[:i] + s[i].upper() + s[i + 1 :]

    # Keep punctuation as-is; do not force sentence endings for partial chunks.
    return s


def _extract_transcript_rewrite_text(response_obj: Any, *, fallback_input: str = "") -> str:
    import re

    def _as_clean_str(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            parts: list[str] = []
            for item in value:
                s = str(item or "").strip()
                if s:
                    parts.append(s)
            return " ".join(parts).strip()
        return str(value).strip()

    def _normalize(text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        # Strip common wrappers from provider outputs.
        if s.startswith("```"):
            s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s).strip()
            if s.endswith("```"):
                s = s[:-3].strip()
        s = re.sub(r"^\s*(clean(?:ed)?_?text|rewrite|paraphrase)\s*:\s*", "", s, flags=re.I).strip()
        s = s.strip("`")
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        s = s.replace("\\n", " ").replace("\\t", " ").replace('\\"', '"').replace("\\'", "'")
        return " ".join(s.split()).strip()

    baseline = " ".join((fallback_input or "").split()).strip()
    raw = _extract_chat_content_best_effort(response_obj)
    raw = str(raw or "").strip()
    if (not raw) and isinstance(response_obj, dict):
        for key in ("output_text", "text", "content"):
            if key in response_obj:
                raw = str(response_obj.get(key) or "").strip()
                if raw:
                    break
    if not raw:
        return ""

    parsed = _parse_json_object_best_effort(raw)
    key_candidates = ("clean_text", "cleanText", "text", "rewrite", "paraphrase", "output")
    for k in key_candidates:
        if k in parsed:
            out = _normalize(_as_clean_str(parsed.get(k)))
            if out:
                return out
    nested = parsed.get("data") if isinstance(parsed.get("data"), dict) else None
    if isinstance(nested, dict):
        for k in key_candidates:
            if k in nested:
                out = _normalize(_as_clean_str(nested.get(k)))
                if out:
                    return out

    # Handle malformed JSON-like responses from providers that ignore strict json_object mode.
    for k in ("clean_text", "cleanText", "text", "rewrite", "paraphrase", "output"):
        m = re.search(rf"""['"]{re.escape(k)}['"]\s*:\s*(['"])(.+?)\1""", raw, flags=re.I | re.S)
        if m:
            out = _normalize(m.group(2))
            if out:
                return out

    normalized_raw = _normalize(raw)
    if not normalized_raw:
        return ""
    if not re.search(r"[A-Za-z0-9]", normalized_raw):
        return ""
    if baseline:
        max_reasonable = min(2400, max(260, int(len(baseline) * 3.2)))
        if len(normalized_raw) > max_reasonable:
            return ""

    # Last-resort fallback: accept plain text if it doesn't look like a JSON blob.
    if not ((normalized_raw.startswith("{") and normalized_raw.endswith("}")) or (normalized_raw.startswith("[") and normalized_raw.endswith("]"))):
        return normalized_raw

    # If still JSON-like and unparsable, fail closed (no rewrite).
    return ""


def _enqueue_transcript_cleanup(q: asyncio.Queue, msg_id: str) -> None:
    try:
        q.put_nowait(msg_id)
    except asyncio.QueueFull:
        try:
            q.get_nowait()
        except Exception:
            return
        try:
            q.put_nowait(msg_id)
        except Exception:
            return


async def run_transcript_cleanup_loop(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    cleanup_queue: asyncio.Queue,
    cleanup_lock: asyncio.Lock,
):
    global _TRANSCRIPT_CLEANUP_LAST_TS

    while True:
        msg_id = await cleanup_queue.get()

        debounce = float(config.get("transcript_ai_cleanup_debounce_seconds", 1.25) or 1.25)
        debounce = max(0.25, min(8.0, debounce))

        while True:
            try:
                msg_id = await asyncio.wait_for(cleanup_queue.get(), timeout=debounce)
                continue
            except asyncio.TimeoutError:
                break

        if not config.get("ai_enabled", False):
            continue

        mode = _get_transcript_ai_mode(config)
        if mode == "off":
            continue
        if llm_client is None or current_session is None:
            continue

        await _wait_for_transcript_settle(queue=cleanup_queue)
        while True:
            try:
                msg_id = cleanup_queue.get_nowait()
            except Exception:
                break

        min_interval = float(config.get("transcript_ai_cleanup_min_interval_seconds", 3.0) or 3.0)
        min_interval = max(0.0, min(30.0, min_interval))
        now_ts = time.time()
        since_last = now_ts - float(_TRANSCRIPT_CLEANUP_LAST_TS or 0.0)
        if since_last < min_interval:
            wait_s = max(0.0, min_interval - since_last)
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            # Prefer newest pending message after cooldown.
            while True:
                try:
                    msg_id = cleanup_queue.get_nowait()
                except Exception:
                    break

        target = None
        for m in reversed(current_session.transcript or []):
            if getattr(m, "id", None) == msg_id:
                target = m
                break
        if target is None:
            continue
        if getattr(target, "source", None) == "assistant":
            continue

        try:
            async with cleanup_lock:
                cleaned = await _transcript_ai_rewrite(getattr(target, "text", "") or "", mode=mode)
        except Exception:
            logger.exception("Transcript cleanup failed")
            cleaned = _local_transcript_rewrite_fallback(getattr(target, "text", "") or "", mode=mode)

        if not cleaned:
            cleaned = _local_transcript_rewrite_fallback(getattr(target, "text", "") or "", mode=mode)
        if not cleaned:
            continue
        if cleaned == (getattr(target, "clean_text", None) or ""):
            continue

        target.clean_text = cleaned
        save_session_throttled(current_session)
        _TRANSCRIPT_CLEANUP_LAST_TS = time.time()

        await _ws_send_json(
            websocket,
            {
                "type": "transcription_update",
                "id": target.id,
                "clean_text": cleaned,
            },
            send_lock,
        )


async def _policy_should_respond(text: str, source: str) -> dict:
    if not llm_client:
        return {"allow": False, "urgency": "wait", "reason": "LLM not configured", "confidence": 1.0}

    transcript = llm_client.format_recent_history(limit=12)
    policy_prompt = config.get("policy_prompt") or ""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a policy gate for a response assistant. Decide whether the system should generate response guidance NOW. "
                "Be conservative: default to WAIT unless a response would be materially useful right now.\n"
                "Allow NOW only when ALL are true:\n"
                "1) There is a specific statement or claim to respond to.\n"
                "2) A response would improve the user's position, clarity, or de-escalation.\n"
                "3) The moment is appropriate (not random filler, tiny acknowledgements, or unclear context).\n"
                "If uncertain, choose WAIT.\n"
                "Follow the user's policy prompt exactly. Output ONLY valid JSON as an object with keys: "
                "allow (boolean), urgency ('now' or 'wait'), reason (string), confidence (number 0..1).\n\n"
                f"Policy prompt:\n{policy_prompt}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"New message source: {source}\n"
                f"New message: {text}\n\n"
                "Recent conversation:\n"
                f"{transcript}"
            ),
        },
    ]

    try:
        resp = await _llm_chat_create(
            messages=messages,
            stream=False,
            response_format={"type": "json_object"},
        )
        content = _extract_chat_content_best_effort(resp)
        data = _parse_json_object_best_effort(content)
        allow = bool(data.get("allow"))
        urgency = data.get("urgency")
        reason = str(data.get("reason") or "")
        confidence = data.get("confidence")
        try:
            confidence = float(confidence)
        except Exception:
            confidence = 0.5
        if urgency not in ("now", "wait"):
            urgency = "wait"
        return {"allow": allow, "urgency": urgency, "reason": reason, "confidence": max(0.0, min(1.0, confidence))}
    except Exception as e:
        logger.error(f"Policy gate error: {e}")
        return {"allow": False, "urgency": "wait", "reason": "Policy gate failed", "confidence": 0.0}


async def process_transcription(
    websocket: WebSocket,
    text: str,
    source: str = "user",
    state: ConnectionState | None = None,
    send_lock: asyncio.Lock | None = None,
    *,
    message_kind: str = "audio",  # "audio" or "manual"
    response_queue: asyncio.Queue | None = None,
    notes_queue: asyncio.Queue | None = None,
    fact_check_queue: asyncio.Queue | None = None,
    transcript_cleanup_queue: asyncio.Queue | None = None,
    speaker_id: str | None = None,
    speaker_label: str | None = None,
):
    global _TRANSCRIPT_ACTIVITY_SEQ

    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    if not text:
        return

    if speaker_id is not None and not isinstance(speaker_id, str):
        speaker_id = str(speaker_id)
    speaker_id = (speaker_id or "").strip() or None

    if speaker_label is not None and not isinstance(speaker_label, str):
        speaker_label = str(speaker_label)
    speaker_label = (speaker_label or "").strip() or None

    if current_session is not None and speaker_id:
        try:
            mapped = (current_session.speaker_names or {}).get(speaker_id)
        except Exception:
            mapped = None
        mapped = (mapped or "").strip()
        if mapped:
            speaker_label = mapped
        elif speaker_label is None and source in ("third_party", "loopback"):
            speaker_label = _default_speaker_label_from_id(speaker_id)

    logger.debug("Transcribed (%s): %s", source, text)
    _TRANSCRIPT_ACTIVITY_SEQ += 1
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Add to (or merge into) session transcript
    now_ts = time.time()
    merged = False
    previous_text_for_ai = ""
    msg: TranscriptMessage | None = None
    if current_session:
        merge_enabled = bool(config.get("transcript_merge_enabled", True))
        merge_window = float(config.get("transcript_merge_window_seconds", 4.0) or 4.0)
        merge_window = max(0.0, min(20.0, merge_window))
        continuation_window = float(
            config.get("transcript_merge_continuation_window_seconds", max(18.0, merge_window)) or max(18.0, merge_window)
        )
        continuation_window = max(merge_window, min(45.0, continuation_window))

        if (
            merge_enabled
            and (current_session.transcript or [])
            and str(source or "") == str(current_session.transcript[-1].source or "")
            and str((speaker_id or "") or "") == str((current_session.transcript[-1].speaker_id or "") or "")
        ):
            prev = current_session.transcript[-1]
            last_ts = float(_TRANSCRIPT_LAST_UPDATE_TS.get(prev.id, 0.0) or 0.0)
            time_gap = now_ts - last_ts
            should_merge = _should_merge_transcript(prev.text, text)

            # Rapid-merge: consecutive same-source chunks arriving within roughly
            # one transcription window are definitively continuous speech.
            # The timing signal is more reliable than sentence-boundary heuristics.
            rapid_window = float(
                config.get("transcription_chunk_duration_seconds", 5.6) or 5.6
            ) + 1.5
            hard_no_merge = (
                not (text or "").strip()
                or (text or "").strip().startswith(("#", "@", "http://", "https://"))
            )
            if not hard_no_merge and time_gap <= rapid_window:
                should_merge = True
                allowed_window = rapid_window
            elif should_merge:
                strong_continuation = _is_strong_transcript_continuation(prev.text, text)
                allowed_window = continuation_window if strong_continuation else merge_window
            else:
                allowed_window = merge_window

            if should_merge and time_gap <= allowed_window:
                previous_text_for_ai = str(prev.text or "")
                prev.text = _merge_transcript_text(prev.text, text)
                prev.clean_text = None  # Invalidate; will be regenerated if enabled.
                prev.timestamp = timestamp
                msg = prev
                merged = True
                _TRANSCRIPT_LAST_UPDATE_TS[prev.id] = now_ts
                save_session_throttled(current_session)

        if msg is None:
            msg = TranscriptMessage(
                id=str(uuid.uuid4())[:8],
                text=text,
                source=source,
                timestamp=timestamp,
                speaker_id=speaker_id,
                speaker_label=speaker_label,
            )
            current_session.transcript.append(msg)
            _TRANSCRIPT_LAST_UPDATE_TS[msg.id] = now_ts
            save_session_throttled(current_session)

    if msg is None:
        # Shouldn't happen, but keep behavior safe.
        msg = TranscriptMessage(
            id=str(uuid.uuid4())[:8],
            text=text,
            source=source,
            timestamp=timestamp,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
        )

    ai_signal_notes = _should_enqueue_ai_from_transcript_update(
        "notes",
        message_id=msg.id,
        current_text=msg.text,
        previous_text=previous_text_for_ai,
        merged=merged,
        message_kind=message_kind,
    )
    ai_signal_fact_check = _should_enqueue_ai_from_transcript_update(
        "fact_check",
        message_id=msg.id,
        current_text=msg.text,
        previous_text=previous_text_for_ai,
        merged=merged,
        message_kind=message_kind,
    )
    ai_signal_cleanup = _should_enqueue_ai_from_transcript_update(
        "transcript_cleanup",
        message_id=msg.id,
        current_text=msg.text,
        previous_text=previous_text_for_ai,
        merged=merged,
        message_kind=message_kind,
    )
    ai_signal_response = _should_enqueue_ai_from_transcript_update(
        "response",
        message_id=msg.id,
        current_text=msg.text,
        previous_text=previous_text_for_ai,
        merged=merged,
        message_kind=message_kind,
    )

    event_type = "transcription_update" if merged else "transcription"
    if not await _ws_send_json(
        websocket,
        {
            "type": event_type,
            "id": msg.id,
            "text": msg.text,
            "clean_text": msg.clean_text,
            "source": msg.source,
            "timestamp": msg.timestamp,
            "speaker_id": msg.speaker_id,
            "speaker_label": msg.speaker_label,
        },
        send_lock,
    ):
        return

    # Keep transcript history for notes/policy/response/fact-check context.
    if llm_client is not None:
        if merged and llm_client.history:
            try:
                last = llm_client.history[-1]
                last_role = (last.get("role") or "").strip().lower()
                last_name = (last.get("name") or "").strip().lower()
                ent = llm_client._normalize_entity(source)  # type: ignore[attr-defined]
                if ent == "third_party":
                    safe_id = _safe_openai_name_component(speaker_id or "", max_len=24).casefold()
                    same_speaker = False
                    if safe_id:
                        same_speaker = (last_name == f"third_party_{safe_id}") or last_name.endswith(f"__{safe_id}")
                    else:
                        same_speaker = (last_name == "third_party")
                    if last_role == "user" and same_speaker:
                        last["content"] = _merge_transcript_text(str(last.get("content") or ""), text)
                    else:
                        llm_client.add_transcript_message(source, text, speaker_id=speaker_id, speaker_label=speaker_label)
                elif last_role == "user" and last_name == "you":
                    last["content"] = _merge_transcript_text(str(last.get("content") or ""), text)
                else:
                    llm_client.add_transcript_message(source, text, speaker_id=speaker_id, speaker_label=speaker_label)
            except Exception:
                llm_client.add_transcript_message(source, text, speaker_id=speaker_id, speaker_label=speaker_label)
        else:
            llm_client.add_transcript_message(source, text, speaker_id=speaker_id, speaker_label=speaker_label)

    if notes_queue is not None and config.get("notes_enabled", True):
        if (
            ai_signal_notes
            and config.get("ai_enabled", False)
            and (config.get("notes_live_on_message", True) or config.get("notes_on_interaction_only", False))
        ):
            _enqueue_notes_trigger(notes_queue)

    if fact_check_queue is not None and config.get("fact_check_enabled", True):
        if (
            ai_signal_fact_check
            and config.get("ai_enabled", False)
            and (config.get("fact_check_live_on_message", True) or config.get("fact_check_on_interaction_only", False))
        ):
            _enqueue_fact_check_trigger(fact_check_queue)

    if (
        transcript_cleanup_queue is not None
        and ai_signal_cleanup
        and config.get("ai_enabled", False)
        and _get_transcript_ai_mode(config) != "off"
    ):
        _enqueue_transcript_cleanup(transcript_cleanup_queue, msg.id)

    # Trigger response generation non-blockingly.
    if config.get("ai_enabled", False):
        if llm_client is None:
            if state is None:
                state = ConnectionState()
            if not state.llm_not_configured_warned:
                state.llm_not_configured_warned = True
                await _ws_send_json(websocket, {"type": "error", "message": "LLM not configured. Check settings."}, send_lock)
            return

        if response_queue is None:
            return

        kind = (message_kind or "audio").strip().lower()
        auto_respond = bool(config.get("auto_respond", False))
        auto_source_allowed = _is_counterparty_source(source)
        should_reply = (kind == "manual") or (auto_respond and auto_source_allowed)
        if should_reply and ai_signal_response:
            _enqueue_response_request(
                response_queue,
                {
                    "trigger": ("manual" if kind == "manual" else "audio"),
                    "text": text,
                    "source": source,
                    "timestamp": msg.timestamp,
                    "speaker_label": (msg.speaker_label or ""),
                    "message_id": msg.id,
                    "ts": time.time(),
                },
            )


# ============================================
# SERVER STARTUP
# ============================================

def check_server_ready(url, timeout=15):
    import time
    import urllib.request
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    # Give the server a moment to fully initialize WebSocket handler
                    time.sleep(0.5)
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def find_available_port(host: str, preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def start_server(host: str, port: int):
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    try:
        server_host = "127.0.0.1"
        preferred_port = int(os.environ.get("AI_ASSISTANT_PORT", "8000"))
        server_port = find_available_port(server_host, preferred_port)

        url = f"http://{server_host}:{server_port}/static/index.html"

        # Pre-load model to avoid threading/GUI crashes
        logger.info("Pre-loading Whisper Model...")
        model_size = str(config.get("whisper_model_size", "tiny") or "tiny").strip() or "tiny"
        runtime_cfg = _resolve_transcription_runtime_config(config)
        transcription_model = TranscriptionEngine(
            model_size=model_size,
            device=str(runtime_cfg.get("device") or "cpu"),
            preprocess=_speech_preprocess_from_config(config),
            whisper_vad_filter=bool(config.get("whisper_vad_filter", True)),
            beam_size=int(runtime_cfg.get("beam_size") or 1),
            chunk_duration_s=float(runtime_cfg.get("chunk_duration_s") or 3.2),
            chunk_overlap_s=float(runtime_cfg.get("chunk_overlap_s") or 0.16),
            condition_on_previous_text=bool(runtime_cfg.get("condition_on_previous_text")),
            runtime_profile=str(runtime_cfg.get("resolved_profile") or "manual"),
        )
        logger.info("Model loaded.")
        log_important(
            "transcription.backend",
            model=model_size,
            device=getattr(transcription_model, "device", "unknown"),
            compute=getattr(transcription_model, "compute_type", "unknown"),
            requested_profile=str(runtime_cfg.get("requested_profile") or "auto"),
            profile=str(runtime_cfg.get("resolved_profile") or "manual"),
            beam=int(runtime_cfg.get("beam_size") or 1),
            chunk_s=f"{float(runtime_cfg.get('chunk_duration_s') or 0.0):.2f}",
            overlap_s=f"{float(runtime_cfg.get('chunk_overlap_s') or 0.0):.2f}",
        )

        use_webview = os.environ.get("AI_ASSISTANT_USE_WEBVIEW") == "1"
        if use_webview:
            # pywebview needs the server running in a background thread.
            t = threading.Thread(target=start_server, args=(server_host, server_port), daemon=True)
            t.start()

            logger.info("Waiting for server to start...")
            if check_server_ready(url):
                import webview

                logger.info("Server ready. Launching UI (pywebview).")
                webview.create_window("AI Assistant", url, width=1200, height=800)
                try:
                    webview.start(debug=True)
                    logger.info("UI closed.")
                except Exception:
                    logger.exception("UI crashed.")
                    raise
            else:
                logger.error("Server failed to start in time.")
        else:
            # Run uvicorn in the main thread (more stable under VS Code/debugpy).
            # Open the browser once the HTTP endpoint is reachable.
            def _open_browser_when_ready() -> None:
                logger.info("Waiting for server to start...")
                if check_server_ready(url):
                    logger.info(f"Server ready. Opening browser: {url}")
                    webbrowser.open(url)
                else:
                    logger.error("Server failed to start in time.")

            threading.Thread(target=_open_browser_when_ready, daemon=True).start()
            logger.info("Starting server...")
            try:
                start_server(server_host, server_port)
            except KeyboardInterrupt:
                logger.info("Stopping...")
    except Exception as e:
        logger.exception("Fatal error during startup:")
        print(f"\n\nFATAL ERROR: {e}\n")
        print("Press Enter to exit...")
        input()
        sys.exit(1)
