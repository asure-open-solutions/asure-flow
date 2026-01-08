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
class TranscriptMessage:
    id: str
    text: str
    source: str  # "user" (You), "third_party"/"loopback" (Third-Party), "assistant" (AI)
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
            "speaker_names": dict(self.speaker_names or {}),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        return cls(
            id=data["id"],
            started_at=data["started_at"],
            ended_at=data.get("ended_at"),
            title=data.get("title", "Untitled Session"),
            context=data.get("context", "") or "",
            transcript=[TranscriptMessage(**m) for m in data.get("transcript", [])],
            notes=[NoteItem(**n) for n in data.get("notes", [])],
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
    "web_search_enabled": False,
    "web_search_mode": "auto",  # "auto" or "always" (only used when web_search_enabled is True)
    "web_search_max_results": 5,
    "web_search_timeout_seconds": 6.0,
    "web_search_cache_ttl_seconds": 180.0,
    "web_search_min_interval_seconds": 6.0,
    "web_search_mode": "auto",  # "auto" or "always" (only used when web_search_enabled is True)
    "web_search_max_results": 5,
    "web_search_timeout_seconds": 6.0,
    "web_search_cache_ttl_seconds": 180.0,
    "web_search_min_interval_seconds": 6.0,
    
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
    # "off" = disabled, "cleanup" = punctuation/fluency only, "paraphrase" = rewrite into what speaker likely meant.
    "transcript_ai_mode": "off",
    "transcript_ai_cleanup_debounce_seconds": 1.25,
    "transcript_ai_cleanup_min_interval_seconds": 3.0,
    "transcript_display_mode": "raw",  # "raw" or "clean"
}


@dataclass
class ConnectionState:
    last_policy_eval_ts: float = 0.0
    last_ai_reply_ts: float = 0.0
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


def _effective_api_settings(cfg: dict) -> tuple[str, str, str, dict[str, str]]:
    base_url_raw = (cfg.get("base_url") or "").strip()
    provider = _normalize_api_provider(cfg.get("api_provider"))
    inferred = _infer_provider_from_base_url(base_url_raw)
    if provider == "custom" and inferred != "custom":
        provider = inferred
    preset = _API_PROVIDER_PRESETS.get(provider, {})

    api_key = (cfg.get("api_key") or "").strip()
    if not api_key:
        env_names = preset.get("api_key_envs")
        if isinstance(env_names, list):
            for env_name in env_names:
                if not isinstance(env_name, str):
                    continue
                api_key = (os.environ.get(env_name) or "").strip()
                if api_key:
                    break
        if not api_key:
            env_name = preset.get("api_key_env")
            if isinstance(env_name, str) and env_name:
                api_key = (os.environ.get(env_name) or "").strip()

    base_url = base_url_raw or (preset.get("base_url") if isinstance(preset.get("base_url"), str) else "") or DEFAULT_CONFIG["base_url"]
    model = (cfg.get("model") or "").strip() or (preset.get("model") if isinstance(preset.get("model"), str) else "") or DEFAULT_CONFIG["model"]
    extra_headers = _coerce_headers(cfg.get("api_extra_headers"))
    return api_key, base_url, model, extra_headers


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        if _CONFIG_PATH.is_file():
            data = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update(data)
    except Exception:
        logger.exception("Failed to load settings file")
    return cfg


def save_config(cfg: dict) -> None:
    try:
        _CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = _CONFIG_PATH.with_suffix(_CONFIG_PATH.suffix + ".tmp")
        tmp_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        tmp_path.replace(_CONFIG_PATH)
    except Exception:
        logger.exception("Failed to save settings file")
        raise


# Configuration
config = load_config()


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


def create_new_session() -> Session:
    global current_session, third_party_diarizer
    
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
    third_party_diarizer = None
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
            "mixage stéréo",
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
        vad_enabled=bool(cfg.get("speech_vad_enabled", True)),
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

    api_key, base_url, model, extra_headers = _effective_api_settings(config)
    if not api_key:
        llm_client = None
        return

    system_prompt = (config.get("system_prompt") or DEFAULT_CONFIG["system_prompt"])

    if (
        llm_client is not None
        and getattr(llm_client, "api_key", None) == api_key
        and getattr(llm_client, "base_url", None) == base_url
        and getattr(llm_client, "model", None) == model
        and getattr(llm_client, "default_headers", None) == extra_headers
    ):
        llm_client.set_system_prompt(system_prompt)
        return

    previous_history = list(getattr(llm_client, "history", []) or [])
    llm_client = LLMClient(api_key=api_key, base_url=base_url, model=model, default_headers=extra_headers)
    llm_client.history = previous_history
    llm_client.set_system_prompt(system_prompt)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Server starting...")
    init_llm_client_from_config()
    # Create initial session
    create_new_session()
    yield
    # Shutdown
    logger.info("Shutting down...")
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
    return Response(status_code=204)


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
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"status": "error", "message": "JSON body must be an object"}, status_code=400)

    config.update(data)
    try:
        save_config(config)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to save settings: {e}"}, status_code=500)

    init_llm_client_from_config()
    
    return {"status": "ok", "config": config}


@app.post("/api/settings/reset")
def api_reset_settings():
    """Reset settings to defaults and persist to disk."""
    global config, llm_client

    try:
        audio_manager.stop_recording()
    except Exception:
        pass

    config = dict(DEFAULT_CONFIG)
    try:
        save_config(config)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to save settings: {e}"}, status_code=500)

    init_llm_client_from_config()
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
        out.append(
            {
                "name": str(name),
                "saved_at": saved_at,
                "has_api_key": bool(isinstance(cfg, dict) and (cfg.get("api_key") or "").strip()),
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

    preset_cfg = dict(config)
    if not include_api_key:
        preset_cfg.pop("api_key", None)

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
    merged = dict(DEFAULT_CONFIG)
    merged.update(preset_cfg)
    if not (preset_cfg.get("api_key") or "").strip():
        merged["api_key"] = config.get("api_key", "")

    try:
        audio_manager.stop_recording()
    except Exception:
        pass

    config = merged
    try:
        save_config(config)
    except Exception as e:
        return JSONResponse({"status": "error", "message": f"Failed to save settings: {e}"}, status_code=500)

    init_llm_client_from_config()
    return {"status": "ok", "config": config}


@app.post("/api/test_connection")
async def api_test_connection(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"status": "error", "message": "Invalid JSON"}, status_code=400)

    if not isinstance(data, dict):
        return JSONResponse({"status": "error", "message": "JSON body must be an object"}, status_code=400)

    merged_cfg = dict(config)
    merged_cfg.update({k: v for k, v in data.items() if k in ("api_provider", "api_key", "base_url", "model", "api_extra_headers")})
    api_key, base_url, model, extra_headers = _effective_api_settings(merged_cfg)

    if not api_key:
        return JSONResponse(
            {
                "status": "error",
                "message": "Missing API key (set it in Settings or via provider env var)",
            },
            status_code=400,
        )

    try:
        from openai import AsyncOpenAI

        t0 = time.monotonic()
        client = AsyncOpenAI(api_key=api_key, base_url=base_url, default_headers=extra_headers)

        model_list_ok = False
        model_found = None
        model_list_error = None
        try:
            resp = await asyncio.wait_for(client.models.list(), timeout=6.0)
            model_list_ok = True
            ids = [getattr(m, "id", None) for m in getattr(resp, "data", [])]
            ids = [i for i in ids if isinstance(i, str)]
            model_found = model in ids if model else None
        except Exception as e:
            model_list_error = str(e)

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

        if not chat_ok:
            return {
                "status": "error",
                "message": chat_error or "Connection test failed",
                "latency_ms": latency_ms,
                "model_list_ok": model_list_ok,
                "model_found": model_found,
                "model_list_error": model_list_error,
            }

        return {
            "status": "ok",
            "message": "Connection OK",
            "latency_ms": latency_ms,
            "model_list_ok": model_list_ok,
            "model_found": model_found,
            "model_list_error": model_list_error,
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

    # Rebuild LLM history for policy/notes context using the 3-entity transcript.
    if llm_client:
        try:
            llm_client.clear_history()
            for m in (session.transcript or []):
                llm_client.add_transcript_message(
                    getattr(m, "source", "user"),
                    getattr(m, "text", ""),
                    speaker_id=getattr(m, "speaker_id", None),
                    speaker_label=getattr(m, "speaker_label", None),
                )
        except Exception:
            logger.exception("Failed to rebuild LLM history from loaded session")

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
    
    content = data.get("content", "").strip()
    if not content:
        return JSONResponse({"status": "error", "message": "Note content required"}, status_code=400)
    
    note = NoteItem(
        id=str(uuid.uuid4())[:8],
        content=content,
        timestamp=datetime.now().strftime("%H:%M:%S"),
        source=data.get("source", "manual"),
        category=data.get("category", "general"),
        pinned=data.get("pinned", False),
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
                    note.content = data["content"]
                if "pinned" in data:
                    note.pinned = data["pinned"]
                if "completed" in data:
                    note.completed = data["completed"]
                if "category" in data:
                    note.category = data["category"]
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
    await websocket.accept()
    logger.info("WebSocket connected")

    state = ConnectionState()
    send_lock = asyncio.Lock()
    notes_lock = asyncio.Lock()
    transcript_cleanup_lock = asyncio.Lock()
    ai_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    ai_task = asyncio.create_task(run_ai_loop(websocket, state, send_lock, ai_queue))
    notes_trigger_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    notes_trigger_task = asyncio.create_task(run_notes_trigger_loop(websocket, send_lock, notes_trigger_queue, notes_lock))
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

    # Let the UI know whether audio capture is enabled in this run.
    if os.environ.get("AI_ASSISTANT_ENABLE_AUDIO") == "1":
        await _ws_send_json(websocket, {"type": "status", "message": "Audio: capture enabled."}, send_lock)
    else:
        await _ws_send_json(
            websocket,
            {
                "type": "status",
                "message": "Audio: capture disabled. Set AI_ASSISTANT_ENABLE_AUDIO=1 to enable (may be unstable).",
            },
            send_lock,
        )
    
    global transcription_model
    transcription_tasks_started = False
    enable_loopback = False
    audio_disabled_warned = False

    async def start_audio_capture():
        global transcription_model
        nonlocal transcription_tasks_started, enable_loopback, audio_disabled_warned
        if transcription_tasks_started or audio_manager.is_recording:
            return

        # Audio capture via soundcard is currently unstable on some Windows/Python setups (can hard-crash).
        # Keep it opt-in via env var so the UI doesn't immediately close.
        if os.environ.get("AI_ASSISTANT_ENABLE_AUDIO") != "1":
            if not audio_disabled_warned:
                audio_disabled_warned = True
                await _ws_send_json(
                    websocket,
                    {
                        "type": "error",
                        "message": "Audio capture is disabled by default. Set AI_ASSISTANT_ENABLE_AUDIO=1 to enable (may be unstable).",
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
            transcription_model = TranscriptionEngine(
                model_size=model_size,
                device=_whisper_device_from_config(config),
                preprocess=_speech_preprocess_from_config(config),
                whisper_vad_filter=bool(config.get("whisper_vad_filter", True)),
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
            await _ws_send_json(
                websocket,
                {
                    "type": "error",
                    "message": "Audio: recording did not start (no streams opened). Check device selection.",
                },
                send_lock,
            )
            return

        if getattr(audio_manager, "mic_active", False):
            asyncio.create_task(run_transcription_loop(websocket, "user", state, send_lock, ai_queue, notes_trigger_queue, transcript_cleanup_queue))
        if getattr(audio_manager, "loopback_active", False):
            asyncio.create_task(run_transcription_loop(websocket, "third_party", state, send_lock, ai_queue, notes_trigger_queue, transcript_cleanup_queue))
        transcription_tasks_started = True
        await _ws_send_json(websocket, {"type": "status", "message": "Audio: capture started."}, send_lock)

    async def stop_audio_capture():
        nonlocal transcription_tasks_started
        if audio_manager.is_recording:
            audio_manager.stop_recording()
        transcription_tasks_started = False
        await _ws_send_json(websocket, {"type": "status", "message": "Audio: capture stopped."}, send_lock)

    # Start notes generation loop (separate AI process with its own settings)
    notes_task = asyncio.create_task(run_notes_loop(websocket, send_lock, notes_lock))

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
                        ai_queue=ai_queue,
                        notes_queue=notes_trigger_queue,
                        transcript_cleanup_queue=transcript_cleanup_queue,
                    )
                elif msg_type == "ai_run_now":
                    _enqueue_ai_request(ai_queue, {"trigger": "run_now", "ts": time.time()})
                elif msg_type == "start_audio":
                    await start_audio_capture()
                elif msg_type == "stop_audio":
                    await stop_audio_capture()
                elif msg_type == "new_session":
                    session = create_new_session()
                    await _ws_send_json(websocket, {
                        "type": "session_info",
                        "session_id": session.id,
                        "started_at": session.started_at,
                    }, send_lock)
                    await _ws_send_json(websocket, {"type": "session_context", "context": (session.context or "")}, send_lock)
                elif msg_type == "set_session_context":
                    ctx = msg.get("context", "")
                    if ctx is not None and not isinstance(ctx, str):
                        ctx = str(ctx)
                    ctx = (ctx or "").strip()
                    if current_session is not None:
                        current_session.context = ctx[:8000]
                        save_session_throttled(current_session)
                        await _ws_send_json(websocket, {"type": "session_context", "context": current_session.context}, send_lock)
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
                    content = msg.get("content", "").strip()
                    if content and current_session:
                        note = NoteItem(
                            id=str(uuid.uuid4())[:8],
                            content=content,
                            timestamp=datetime.now().strftime("%H:%M:%S"),
                            source="manual",
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
                elif msg_type == "update_note":
                    note_id = msg.get("note_id")
                    updates = msg.get("updates", {})
                    if current_session and note_id:
                        for note in current_session.notes:
                            if note.id == note_id:
                                if "pinned" in updates:
                                    note.pinned = updates["pinned"]
                                if "completed" in updates:
                                    note.completed = updates["completed"]
                                if "content" in updates:
                                    note.content = updates["content"]
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
                    
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected (code={getattr(e, 'code', None)})")
    except Exception:
        logger.exception("WebSocket crashed")
    finally:
        notes_task.cancel()
        with suppress(BaseException):
            await notes_task
        ai_task.cancel()
        with suppress(BaseException):
            await ai_task
        notes_trigger_task.cancel()
        with suppress(BaseException):
            await notes_trigger_task
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
    if not llm_client:
        return False, ""

    seed = " ".join((seed_text or "").split()).strip()
    if len(seed) < 6:
        return False, ""

    mode = (config.get("web_search_mode") or "auto").strip().lower()
    if mode not in ("auto", "always"):
        mode = "auto"

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

    if mode == "always":
        system_prompt += "\nMode is ALWAYS: set search=true (still choose the best query)."

    resp = await llm_client.client.chat.completions.create(
        model=llm_client.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": seed[:900]},
        ],
        stream=False,
        temperature=0,
        max_tokens=160,
        response_format={"type": "json_object"},
    )

    content = resp.choices[0].message.content or ""

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
    should = bool(data.get("search"))
    query = (data.get("query") or "").strip()
    query = " ".join(query.split())[:120]
    if not query:
        query = seed[:120]
    if mode == "always":
        should = True
    if len(query) < 2:
        return False, ""
    return should, query


async def _maybe_build_web_search_context(*, purpose: str, seed_text: str) -> Optional[str]:
    if not config.get("web_search_enabled", False):
        return None

    global _WEB_SEARCH_LAST_TS
    now_ts = time.time()
    min_interval = float(config.get("web_search_min_interval_seconds", 6.0) or 6.0)
    min_interval = max(0.0, min(60.0, min_interval))
    if now_ts - _WEB_SEARCH_LAST_TS < min_interval:
        return None

    try:
        should, query = await _decide_web_search(seed_text=seed_text, purpose=purpose)
    except Exception as e:
        logger.warning(f"Web search decision failed: {e}")
        should, query = False, ""

    if not should or not query:
        return None

    ttl = float(config.get("web_search_cache_ttl_seconds", 180.0) or 180.0)
    ttl = max(10.0, min(3600.0, ttl))
    cache_key = query.casefold()
    cached = _WEB_SEARCH_CACHE.get(cache_key)
    if cached and (now_ts - cached[0]) < ttl:
        _WEB_SEARCH_LAST_TS = now_ts
        return cached[1]

    max_results = int(config.get("web_search_max_results", 5) or 5)
    max_results = max(1, min(10, max_results))
    timeout_s = float(config.get("web_search_timeout_seconds", 6.0) or 6.0)
    timeout_s = max(2.0, min(20.0, timeout_s))

    try:
        results = await asyncio.to_thread(
            _ddg_lite_search_sync,
            query,
            timeout_s=timeout_s,
            max_results=max_results,
        )
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
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
                _WEB_SEARCH_LAST_TS = now_ts
                return ctx
        except Exception as e:
            logger.warning(f"Web search fallback failed: {e}")

        return None

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
            line += f" — {snippet}"
        if url:
            line += f"\n   Source: {url}"
        lines.append(line)

    ctx = "\n".join(lines)[:2200]
    _web_search_cache_put(cache_key, ctx, ts=now_ts)
    _WEB_SEARCH_LAST_TS = now_ts
    return ctx


# ============================================
# AI REPLY LOOP (Non-blocking)
# ============================================

def _enqueue_ai_request(ai_queue: asyncio.Queue, payload: dict) -> None:
    try:
        ai_queue.put_nowait(payload)
    except asyncio.QueueFull:
        try:
            ai_queue.get_nowait()
        except Exception:
            return
        try:
            ai_queue.put_nowait(payload)
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


async def run_notes_trigger_loop(
    websocket: WebSocket,
    send_lock: asyncio.Lock,
    notes_queue: asyncio.Queue,
    notes_lock: asyncio.Lock,
):
    last_generated_ts = 0.0

    while True:
        await notes_queue.get()

        debounce = float(config.get("notes_debounce_seconds", 2.5) or 2.5)
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

        min_interval = float(config.get("notes_trigger_min_interval_seconds", 10.0) or 10.0)
        min_interval = max(0.0, min(300.0, min_interval))
        now_ts = time.time()
        if now_ts - last_generated_ts < min_interval:
            continue

        ok = await generate_notes(websocket, send_lock, notes_lock)
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


def _latest_non_assistant_transcript() -> Optional[tuple[str, str]]:
    if not current_session or not current_session.transcript:
        return None

    for m in reversed(current_session.transcript):
        try:
            if getattr(m, "source", "") == "assistant":
                continue
            text = (getattr(m, "text", "") or "").strip()
            if text:
                return text, (getattr(m, "source", "user") or "user")
        except Exception:
            continue

    return None


async def run_ai_loop(
    websocket: WebSocket,
    state: ConnectionState,
    send_lock: asyncio.Lock,
    ai_queue: asyncio.Queue,
):
    while True:
        req = await ai_queue.get()
        trigger = (req.get("trigger") or "").strip().lower()

        if not config.get("ai_enabled", False):
            continue

        if llm_client is None:
            if not state.llm_not_configured_warned:
                state.llm_not_configured_warned = True
                await _ws_send_json(websocket, {"type": "error", "message": "LLM not configured. Check settings."}, send_lock)
            continue

        now_ts = time.time()

        text = (req.get("text") or "").strip()
        source = (req.get("source") or "user").strip() or "user"
        if (not text) and trigger == "run_now":
            latest = _latest_non_assistant_transcript()
            if latest:
                text, source = latest

        # Hard override: if user explicitly asked for no response, do not reply (still keep notes/history).
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
            continue

        # Policy gate (applies to manual, audio, and run-now).
        if config.get("policy_enabled", False):
            min_policy_interval = float(config.get("policy_min_interval_seconds", 0.0) or 0.0)
            # Manual + run-now should evaluate immediately (don't skip on interval).
            if trigger in ("manual", "run_now"):
                min_policy_interval = 0.0

            if now_ts - state.last_policy_eval_ts < min_policy_interval:
                continue

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
                continue

            if not decision.get("allow") or decision.get("urgency") != "now":
                continue

        # Rate limit only for audio-driven auto-respond; manual + run-now should run immediately.
        if trigger == "audio":
            min_ai_interval = float(config.get("ai_min_interval_seconds", 0.0) or 0.0)
            if now_ts - state.last_ai_reply_ts < min_ai_interval:
                continue

        extra_messages: list[dict] = []
        web_ctx = await _maybe_build_web_search_context(purpose="assistant_reply", seed_text=text)
        if web_ctx:
            extra_messages.append({"role": "system", "content": web_ctx})

        if not await _ws_send_json(websocket, {"type": "llm_start"}, send_lock):
            continue

        full_response = ""
        pending = ""
        last_flush = time.monotonic()
        session_ctx = _get_session_context()
        system_prompt = llm_client.system_prompt
        if session_ctx:
            system_prompt = f"{system_prompt}\n\nSession context:\n{session_ctx}"

        try:
            async for chunk in llm_client.stream_reply(extra_messages=extra_messages, system_prompt=system_prompt):
                full_response += chunk
                pending += chunk

                now_m = time.monotonic()
                if len(pending) >= 80 or (now_m - last_flush) >= 0.06:
                    if not await _ws_send_json(websocket, {"type": "llm_chunk", "text": pending}, send_lock):
                        pending = ""
                        break
                    pending = ""
                    last_flush = now_m
        finally:
            if pending:
                await _ws_send_json(websocket, {"type": "llm_chunk", "text": pending}, send_lock)
            await _ws_send_json(websocket, {"type": "llm_end"}, send_lock)

        if current_session and full_response.strip():
            state.last_ai_reply_ts = time.time()
            current_session.transcript.append(
                TranscriptMessage(
                    id=str(uuid.uuid4())[:8],
                    text=full_response.strip(),
                    source="assistant",
                    timestamp=datetime.now().strftime("%H:%M:%S"),
                )
            )
            save_session_throttled(current_session)


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
    
    notes_format = config.get("notes_format", "bullets")
    
    if notes_format == "structured":
        return f"""You are a note-taking assistant. {base_prompt}

Return a JSON object with these keys (arrays of strings): {', '.join(categories)}.
Only include categories that have relevant content. Each item should be concise (1-2 sentences max)."""
    
    elif notes_format == "summary":
        return f"""You are a note-taking assistant. {base_prompt}

Return a JSON object with a single key "summary" containing a brief paragraph summarizing the key points.
Focus on: {', '.join(categories)}."""
    
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


def _dedupe_note_dicts(notes: list[dict], *, max_items: int) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for n in notes:
        if len(out) >= max_items:
            break
        if not isinstance(n, dict):
            continue
        content = (n.get("content") or "")
        if not isinstance(content, str):
            content = str(content)
        content = " ".join(content.split()).strip()
        if not content:
            continue
        key = content.casefold()
        if key in seen:
            continue
        seen.add(key)
        category = (n.get("category") or "general")
        if not isinstance(category, str):
            category = str(category)
        category = category.strip().lower() or "general"
        if category not in ("decision", "action", "risk", "fact", "general"):
            category = "general"
        out.append({"content": content[:220].strip(), "category": category})
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

    response = await llm_client.client.chat.completions.create(
        model=llm_client.model,
        messages=messages,
        stream=False,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    notes_raw = data.get("notes", [])
    if not isinstance(notes_raw, list):
        notes_raw = []

    cleaned = _dedupe_note_dicts(notes_raw, max_items=max_count)
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

        context_size = int(config.get("notes_context_messages", 10))
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
                maintained = []

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
        
        response = await llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=messages,
            stream=False,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        data = json.loads(content)
        
        notes_format = config.get("notes_format", "bullets")
        timestamp = datetime.now().strftime("%H:%M:%S")
        new_notes = []
        
        if notes_format == "structured":
            # Handle structured format with categories
            for category in ["decisions", "action_items", "risks", "key_facts"]:
                items = _coerce_string_list(data.get(category, []))
                cat_map = {
                    "decisions": "decision",
                    "action_items": "action",
                    "risks": "risk",
                    "key_facts": "fact",
                }
                for item in items:
                    note = NoteItem(
                        id=str(uuid.uuid4())[:8],
                        content=item.strip(),
                        timestamp=timestamp,
                        source="ai",
                        category=cat_map.get(category, "general"),
                        session_id=current_session.id,
                    )
                    new_notes.append(note)
        
        elif notes_format == "summary":
            summary = data.get("summary", "")
            if summary is not None and not isinstance(summary, str):
                summary = str(summary)
            summary = (summary or "").strip()
            if summary:
                note = NoteItem(
                    id=str(uuid.uuid4())[:8],
                    content=summary,
                    timestamp=timestamp,
                    source="ai",
                    category="summary",
                    session_id=current_session.id,
                )
                new_notes.append(note)
        
        else:  # bullets
            points = _coerce_string_list(data.get("key_points", []))
            for point in points:
                note = NoteItem(
                    id=str(uuid.uuid4())[:8],
                    content=point.strip(),
                    timestamp=timestamp,
                    source="ai",
                    session_id=current_session.id,
                )
                new_notes.append(note)
        
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
            return True
        
    except Exception as e:
        logger.error(f"Error generating notes: {e}")
    finally:
        if notes_lock is not None and notes_lock.locked():
            notes_lock.release()
    
    return False


def _format_recent_history_bounded(*, limit: int = 10, max_chars: int = 24000) -> str:
    if not llm_client:
        return ""

    try:
        max_chars = int(max_chars)
    except Exception:
        max_chars = 24000
    max_chars = max(2000, min(250000, max_chars))

    try:
        limit = int(limit)
    except Exception:
        limit = 10
    limit = max(1, min(200, limit))

    def _label_from_message(message: dict) -> str:
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

    out_rev: list[str] = []
    used = 0
    taken = 0

    for m in reversed(getattr(llm_client, "history", []) or []):
        if taken >= limit:
            break

        if not isinstance(m, dict):
            continue

        content = (m.get("content") or "")
        if not isinstance(content, str):
            content = str(content)
        content = " ".join(content.split())
        if not content:
            continue

        label = _label_from_message(m)
        line = f"{label}: {content}"

        remaining = max_chars - used
        if remaining <= 0:
            break

        if len(line) > remaining:
            suffix = " …[truncated]"
            keep = max(0, remaining - len(suffix))
            line = (line[:keep] + suffix).strip()

        if not line:
            break

        out_rev.append(line)
        used += len(line) + 1
        taken += 1

        if used >= max_chars:
            break

    return "\n".join(reversed(out_rev)).strip()


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
            ln = ln.lstrip("-*• \t").strip()
            if ln:
                lines.append(ln)
        return lines or [s]

    # Fallback: single item.
    s = str(value).strip()
    return [s] if s else []


async def run_notes_loop(websocket: WebSocket, send_lock: asyncio.Lock, notes_lock: asyncio.Lock):
    """Periodically generates notes from conversation history."""
    logger.info("Starting notes generation loop")
    
    while True:
        interval = int(config.get("notes_interval_seconds", 30))
        await asyncio.sleep(interval)
        
        if not config.get("ai_enabled", False):
            continue

        if not config.get("notes_enabled", True):
            continue

        if config.get("notes_on_interaction_only", False):
            continue
        
        await generate_notes(websocket, send_lock, notes_lock)


# ============================================
# TRANSCRIPTION & LLM PROCESSING
# ============================================

async def run_transcription_loop(
    websocket: WebSocket,
    source: str,
    state: ConnectionState,
    send_lock: asyncio.Lock,
    ai_queue: asyncio.Queue,
    notes_queue: asyncio.Queue,
    transcript_cleanup_queue: asyncio.Queue | None = None,
):
    """
    Consumes audio from audio_manager, runs transcription, sends to WS.
    Uses a shared transcription model instance (calls are internally serialized).
    """
    logger.info(f"Starting transcription loop for {source}")
    if not await _ws_send_json(
        websocket,
        {"type": "status", "message": f"Audio: transcription loop started for {source}."},
        send_lock,
    ):
        return
    
    def transcription_worker():
        if transcription_model is None:
            logger.error("Transcription model not initialized!")
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
                    ai_queue=ai_queue,
                    notes_queue=notes_queue,
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


async def _transcript_cleanup_text(text: str) -> str:
    if not llm_client:
        return ""

    t = " ".join((text or "").split()).strip()
    if not t:
        return ""

    system_prompt = (
        "You clean up speech-to-text transcript fragments for readability.\n"
        "Rules:\n"
        "- Do NOT add new facts or change meaning.\n"
        "- Keep the same intent (questions stay questions).\n"
        "- Keep pronouns and references; do not invent names.\n"
        "- Fix casing/punctuation and remove obvious stutters/repeats.\n"
        "- Output ONLY valid JSON: {\"clean_text\": \"...\"}\n"
    )

    resp = await llm_client.client.chat.completions.create(
        model=llm_client.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": t[:1200]},
        ],
        stream=False,
        temperature=0,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    cleaned = data.get("clean_text", "")
    if cleaned is not None and not isinstance(cleaned, str):
        cleaned = str(cleaned)
    cleaned = " ".join((cleaned or "").split()).strip()
    return cleaned


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

    if not llm_client:
        return ""

    t = " ".join((text or "").split()).strip()
    if not t:
        return ""

    system_prompt = (
        "You rewrite speech-to-text transcript fragments into a clearer paraphrase of what the speaker likely meant.\n"
        "Rules:\n"
        "- Preserve meaning; do NOT add new facts.\n"
        "- If ambiguous, keep it ambiguous (do not guess specifics).\n"
        "- Keep perspective (I/you/they) and intent (questions remain questions).\n"
        "- Output ONLY valid JSON: {\"clean_text\": \"...\"}\n"
    )

    resp = await llm_client.client.chat.completions.create(
        model=llm_client.model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": t[:1200]},
        ],
        stream=False,
        temperature=0.2,
        max_tokens=220,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    cleaned = data.get("clean_text", "")
    if cleaned is not None and not isinstance(cleaned, str):
        cleaned = str(cleaned)
    cleaned = " ".join((cleaned or "").split()).strip()
    return cleaned


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

        min_interval = float(config.get("transcript_ai_cleanup_min_interval_seconds", 3.0) or 3.0)
        min_interval = max(0.0, min(30.0, min_interval))
        now_ts = time.time()
        if now_ts - float(_TRANSCRIPT_CLEANUP_LAST_TS or 0.0) < min_interval:
            continue

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
            continue

        if not cleaned or cleaned == (getattr(target, "text", "") or ""):
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
                "You are a policy gate for an assistant. Decide whether the assistant should respond NOW. "
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
        resp = await llm_client.client.chat.completions.create(
            model=llm_client.model,
            messages=messages,
            stream=False,
            response_format={"type": "json_object"},
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
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
    ai_queue: asyncio.Queue | None = None,
    notes_queue: asyncio.Queue | None = None,
    transcript_cleanup_queue: asyncio.Queue | None = None,
    speaker_id: str | None = None,
    speaker_label: str | None = None,
):
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

    logger.info(f"Transcribed ({source}): {text}")
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Add to (or merge into) session transcript
    now_ts = time.time()
    merged = False
    msg: TranscriptMessage | None = None
    if current_session:
        merge_enabled = bool(config.get("transcript_merge_enabled", True))
        merge_window = float(config.get("transcript_merge_window_seconds", 4.0) or 4.0)
        merge_window = max(0.0, min(20.0, merge_window))

        if (
            merge_enabled
            and (current_session.transcript or [])
            and str(source or "") == str(current_session.transcript[-1].source or "")
            and str((speaker_id or "") or "") == str((current_session.transcript[-1].speaker_id or "") or "")
        ):
            prev = current_session.transcript[-1]
            last_ts = float(_TRANSCRIPT_LAST_UPDATE_TS.get(prev.id, 0.0) or 0.0)
            if (now_ts - last_ts) <= merge_window and _should_merge_transcript(prev.text, text):
                prev.text = (prev.text.rstrip() + " " + text.lstrip()).strip()
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

    # Always keep transcript history for notes/policy context when configured.
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
                        last["content"] = (str(last.get("content") or "").rstrip() + " " + text.lstrip()).strip()
                    else:
                        llm_client.add_transcript_message(source, text, speaker_id=speaker_id, speaker_label=speaker_label)
                elif last_role == "user" and last_name == "you":
                    last["content"] = (str(last.get("content") or "").rstrip() + " " + text.lstrip()).strip()
                else:
                    llm_client.add_transcript_message(source, text, speaker_id=speaker_id, speaker_label=speaker_label)
            except Exception:
                llm_client.add_transcript_message(source, text, speaker_id=speaker_id, speaker_label=speaker_label)
        else:
            llm_client.add_transcript_message(source, text, speaker_id=speaker_id, speaker_label=speaker_label)

    if notes_queue is not None and config.get("notes_enabled", True):
        if config.get("ai_enabled", False) and (config.get("notes_live_on_message", True) or config.get("notes_on_interaction_only", False)):
            _enqueue_notes_trigger(notes_queue)

    if transcript_cleanup_queue is not None and config.get("ai_enabled", False) and _get_transcript_ai_mode(config) != "off":
        _enqueue_transcript_cleanup(transcript_cleanup_queue, msg.id)

    # Trigger AI replies non-blockingly.
    if config.get("ai_enabled", False):
        if llm_client is None:
            if state is None:
                state = ConnectionState()
            if not state.llm_not_configured_warned:
                state.llm_not_configured_warned = True
                await _ws_send_json(websocket, {"type": "error", "message": "LLM not configured. Check settings."}, send_lock)
            return

        if ai_queue is None:
            return

        kind = (message_kind or "audio").strip().lower()
        should_reply = (kind == "manual") or bool(config.get("auto_respond", False))
        if should_reply:
            _enqueue_ai_request(ai_queue, {"trigger": ("manual" if kind == "manual" else "audio"), "text": text, "source": source, "ts": time.time()})


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
        transcription_model = TranscriptionEngine(
            model_size=model_size,
            device=_whisper_device_from_config(config),
            preprocess=_speech_preprocess_from_config(config),
            whisper_vad_filter=bool(config.get("whisper_vad_filter", True)),
        )
        logger.info("Model loaded.")

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
