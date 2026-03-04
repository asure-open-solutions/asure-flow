"""REST API routes — sessions CRUD, health, config, audio devices."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from asure_flow.agent.router import get_router, init_router
from asure_flow.config import (
    ProviderEntry, settings, update_settings, reset_settings,
    update_provider as _update_provider, add_provider as _add_provider,
    remove_provider as _remove_provider, reorder_providers as _reorder_providers,
)
from asure_flow.profile import profile, update_profile, reset_profile
from asure_flow.sessions.manager import session_manager
from asure_flow.sessions.models import Session, SessionSettings, SessionSummary
from asure_flow.transcription.engine import whisper_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ── Health ──


@router.get("/health")
async def health():
    llm = get_router()
    provider_name = None
    if llm and llm.model_list:
        provider_name = llm.model_list[0].get("model_name", "unknown")
    return {
        "status": "ok",
        "llm_available": llm is not None,
        "llm_provider": provider_name,
    }


# ── Sessions ──


class CreateSessionRequest(BaseModel):
    name: Optional[str] = "Untitled Session"


@router.get("/sessions", response_model=list[SessionSummary])
async def list_sessions():
    return session_manager.list_sessions()


@router.post("/sessions", response_model=Session)
async def create_session(body: CreateSessionRequest):
    return session_manager.create(name=body.name or "Untitled Session")


@router.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not session_manager.delete(session_id):
        raise HTTPException(404, f"Session {session_id} not found")
    # Clear cached embedding index for the deleted session
    from asure_flow.search.index import clear_index_cache
    clear_index_cache(session_id)
    return {"deleted": True}


class UpdateSessionRequest(BaseModel):
    name: Optional[str] = None


@router.patch("/sessions/{session_id}")
async def update_session(session_id: str, body: UpdateSessionRequest):
    from asure_flow.sessions.models import _utcnow

    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    if body.name is not None:
        name = body.name.strip()
        if not name:
            raise HTTPException(422, "Session name cannot be empty")
        session.name = name
    session.updated_at = _utcnow()
    session_manager.save(session)
    return session.model_dump(mode="json")


# ── Transcript Entry Actions ──


class EditTranscriptRequest(BaseModel):
    text: str


@router.patch("/sessions/{session_id}/transcript/{entry_id}")
async def edit_transcript_entry(session_id: str, entry_id: str, body: EditTranscriptRequest):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    text = body.text.strip()
    if not text:
        raise HTTPException(422, "Text cannot be empty")
    entry = session.edit_transcript_entry(entry_id, text)
    if not entry:
        raise HTTPException(404, f"Transcript entry {entry_id} not found")
    session_manager.save(session)
    return entry.model_dump(mode="json")


@router.delete("/sessions/{session_id}/transcript/{entry_id}")
async def delete_transcript_entry(session_id: str, entry_id: str):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    if not session.delete_transcript_entry(entry_id):
        raise HTTPException(404, f"Transcript entry {entry_id} not found")
    session_manager.save(session)
    return {"deleted": True}


# ── Session Settings (per-session overrides) ──


@router.patch("/sessions/{session_id}/settings")
async def update_session_settings(session_id: str, body: SessionSettings):
    from asure_flow.sessions.models import _utcnow

    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    if session.settings is None:
        session.settings = body
    else:
        updates = body.model_dump(exclude_none=True)
        for key, value in updates.items():
            setattr(session.settings, key, value)
    session.updated_at = _utcnow()
    session_manager.save(session)
    return session.settings.model_dump(mode="json")


@router.delete("/sessions/{session_id}/settings")
async def clear_session_settings(session_id: str):
    from asure_flow.sessions.models import _utcnow

    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    session.settings = None
    session.updated_at = _utcnow()
    session_manager.save(session)
    return {"cleared": True}


@router.get("/sessions/{session_id}/export")
async def export_session(session_id: str):
    """Export a session as a full JSON document."""
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    return session.model_dump(mode="json")


@router.get("/sessions/{session_id}/export/markdown")
async def export_session_markdown(session_id: str):
    """Export a session as a Markdown document."""
    from fastapi.responses import PlainTextResponse

    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    md = _session_to_markdown(session)
    return PlainTextResponse(
        md,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{session.name}.md"'},
    )


# ── Participants ──


class UpdateParticipantRequest(BaseModel):
    display_name: str
    role: Optional[str] = None
    notes: Optional[str] = None


@router.get("/sessions/{session_id}/participants")
async def list_participants(session_id: str):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    return [p.model_dump(mode="json") for p in session.participants]


@router.put("/sessions/{session_id}/participants/{speaker_label}")
async def update_participant(session_id: str, speaker_label: str, body: UpdateParticipantRequest):
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    participant = session.rename_speaker(speaker_label, body.display_name, body.role)
    if body.notes is not None:
        participant.notes = body.notes
    session_manager.save(session)
    return participant.model_dump(mode="json")


# ── Follow-up Draft ──


class GenerateFollowupRequest(BaseModel):
    format: str = "email"  # "email" | "message" | "summary"


@router.post("/sessions/{session_id}/followup")
async def generate_followup_endpoint(session_id: str, body: GenerateFollowupRequest):
    """Generate a follow-up draft (email, message, or summary) for a session."""
    session = session_manager.get(session_id)
    if not session:
        raise HTTPException(404, f"Session {session_id} not found")
    llm_router = get_router()
    if not llm_router:
        raise HTTPException(503, "No LLM provider available")
    from asure_flow.agent.followup import generate_followup
    result = await generate_followup(llm_router, session, format=body.format)
    return result


# ── Config ──


class UpdateConfigRequest(BaseModel):
    """Server-admin settings (Tier 1): hardware, secrets, non-provider configuration.

    Provider configuration uses dedicated CRUD endpoints under /api/config/providers.
    User preferences (feature toggles, AI preset, privacy) are in PUT /api/profile.
    """
    # Transcription
    whisper_model: Optional[str] = None
    whisper_device: Optional[str] = None  # "cuda" | "cpu"
    # Audio capture (server-mode device IDs — used when audio_capture_source="server")
    audio_capture_source: Optional[str] = None  # "client" | "server"
    mic_device_id: Optional[str] = None
    system_device_id: Optional[str] = None
    # Diarization hardware
    hf_diarization_token: Optional[str] = None
    diarization_device: Optional[str] = None
    # LLM routing strategy
    routing_strategy: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    """User profile settings (Tier 2): portable preferences synced across client machines.

    These follow the user — changing them on one client machine affects all clients
    connected to the same server.
    """
    # Feature toggles
    fact_checking: Optional[bool] = None
    suggestions: Optional[bool] = None
    notes: Optional[bool] = None
    search_transcript: Optional[bool] = None
    search_sessions: Optional[bool] = None
    web_search: Optional[bool] = None
    format_code: Optional[bool] = None
    deep_think: Optional[str] = None  # "off" | "auto" | "always"
    # AI behaviour
    ai_preset: Optional[str] = None
    custom_system_prompt: Optional[str] = None
    # Diarization preference
    diarization_enabled: Optional[bool] = None
    # Safety
    pii_redaction: Optional[bool] = None
    privacy_mode: Optional[bool] = None


@router.get("/config")
async def get_config():
    return settings.to_client_config()


@router.put("/config")
async def update_config(body: UpdateConfigRequest):
    changes = body.model_dump(exclude_none=True)
    if not changes:
        return settings.to_client_config()

    # Reject changes to admin-locked fields
    if settings.locked_settings:
        blocked = [k for k in changes if k in settings.locked_settings]
        if blocked:
            raise HTTPException(
                403,
                f"Settings are locked by server admin: {', '.join(sorted(blocked))}",
            )

    device_change = changes.pop("whisper_device", None)
    model_change = changes.pop("whisper_model", None)
    routing_change = changes.pop("routing_strategy", None)

    # Apply config changes
    if changes:
        update_settings(**changes)
        logger.info("Config updated: %s", list(changes.keys()))

    # Apply routing strategy change (requires router rebuild)
    if routing_change and routing_change != settings.routing_strategy:
        update_settings(routing_strategy=routing_change)
        init_router()
        logger.info("Routing strategy changed to: %s, router rebuilt", routing_change)

    # Apply whisper model change (requires model reload)
    needs_reload = False
    if model_change and model_change != settings.whisper_model:
        update_settings(whisper_model=model_change)
        logger.info("Whisper model changed to: %s", model_change)
        needs_reload = True

    # Apply device change (requires model reload)
    if device_change and device_change != settings.detect_device():
        update_settings(whisper_device=device_change)
        logger.info("Reloading whisper model on device: %s", device_change)
        needs_reload = True

    if needs_reload:
        await whisper_engine.load()

    return settings.to_client_config()


@router.get("/config/presets")
async def list_presets():
    """List available AI behaviour presets."""
    from asure_flow.agent.presets import PRESETS
    return [
        {
            "id": p.id,
            "name": p.name,
            "description": p.description,
            "default_tools": p.default_tools,
        }
        for p in PRESETS.values()
    ]


@router.post("/config/reset")
async def reset_config():
    """Reset all server-admin settings and user profile to defaults, then reload."""
    reset_settings()
    reset_profile()
    init_router()
    await whisper_engine.load()
    logger.info("Config and profile reset to defaults, router and whisper reloaded")
    return settings.to_client_config()


# ── Provider CRUD ──


class ProviderUpdateRequest(BaseModel):
    """Merge-update a single provider."""
    name: Optional[str] = None
    litellm_prefix: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    enabled: Optional[bool] = None


class ProviderCreateRequest(BaseModel):
    """Create a new provider."""
    id: str
    name: str
    litellm_prefix: str = "openai"
    model: str = ""
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    enabled: bool = True


class ProviderOrderRequest(BaseModel):
    order: list[str]


@router.put("/config/providers/{provider_id}")
async def update_single_provider(provider_id: str, body: ProviderUpdateRequest):
    """Update a single provider entry (merge-style)."""
    changes = body.model_dump(exclude_none=True)
    if not changes:
        return settings.to_client_config()
    result = _update_provider(provider_id, **changes)
    if result is None:
        raise HTTPException(404, f"Provider '{provider_id}' not found")
    init_router()
    logger.info("Provider '%s' updated, router rebuilt", provider_id)
    return settings.to_client_config()


@router.post("/config/providers")
async def create_provider(body: ProviderCreateRequest):
    """Add a new provider entry."""
    entry = ProviderEntry(**body.model_dump())
    try:
        _add_provider(entry)
    except ValueError as e:
        raise HTTPException(409, str(e))
    init_router()
    logger.info("Provider '%s' added, router rebuilt", entry.id)
    return settings.to_client_config()


@router.delete("/config/providers/{provider_id}")
async def delete_provider(provider_id: str):
    """Remove a provider entry."""
    if not _remove_provider(provider_id):
        raise HTTPException(404, f"Provider '{provider_id}' not found")
    init_router()
    logger.info("Provider '%s' removed, router rebuilt", provider_id)
    return settings.to_client_config()


@router.put("/config/providers/order")
async def reorder_providers_endpoint(body: ProviderOrderRequest):
    """Reorder providers by ID list (position = fallback priority)."""
    _reorder_providers(body.order)
    init_router()
    logger.info("Provider order updated, router rebuilt")
    return settings.to_client_config()


# ── User Profile ──


@router.get("/profile")
async def get_profile():
    """Fetch user profile (portable preferences synced across client machines)."""
    return profile.to_dict()


@router.put("/profile")
async def update_profile_endpoint(body: UpdateProfileRequest):
    """Update user profile settings."""
    changes = body.model_dump(exclude_none=True)
    if not changes:
        return profile.to_dict()

    # Reject changes to admin-locked profile fields
    if settings.locked_settings:
        blocked = [k for k in changes if k in settings.locked_settings]
        if blocked:
            raise HTTPException(
                403,
                f"Settings are locked by server admin: {', '.join(sorted(blocked))}",
            )

    # Apply privacy_mode side-effects
    if changes.get("privacy_mode"):
        changes.setdefault("pii_redaction", True)
        changes.setdefault("web_search", False)

    update_profile(**changes)
    logger.info("Profile updated: %s", list(changes.keys()))
    return profile.to_dict()


# ── Audio Devices ──


@router.get("/audio/devices")
async def get_audio_devices():
    """List all audio devices available on the server machine."""
    try:
        from asure_flow.audio.capture import enumerate_devices
        devices = enumerate_devices()
        return {
            "available": True,
            "devices": [
                {
                    "id": d.id,
                    "name": d.name,
                    "channels": d.channels,
                    "sample_rate": d.sample_rate,
                    "is_input": d.is_input,
                    "is_output": d.is_output,
                    "is_loopback": d.is_loopback,
                }
                for d in devices
            ],
        }
    except Exception:
        return {"available": False, "devices": []}


# ── Search ──


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None  # None = search across all sessions
    speaker: Optional[str] = None
    max_results: int = 20


@router.post("/search")
async def search(body: SearchRequest):
    """User-initiated semantic search across transcripts."""
    from asure_flow.search.embeddings import embedding_engine
    from asure_flow.search.index import get_index
    from asure_flow.agent.tools import _ensure_index, _ts

    query = body.query.strip()
    if not query:
        return {"results": [], "search_type": "none"}

    results = []
    search_type = "substring"

    if body.session_id:
        # Single-session search
        session = session_manager.get(body.session_id)
        if not session:
            raise HTTPException(404, "Session not found")

        if embedding_engine.available and session.transcript:
            search_type = "semantic"
            await _ensure_index(session.id, session.transcript)
            idx = get_index(session.id)
            query_emb = await embedding_engine.embed_single(query)
            matches = idx.search(query_emb, top_k=body.max_results)
            entry_map = {e.id: e for e in session.transcript}

            for entry_id, score in matches:
                entry = entry_map.get(entry_id)
                if not entry:
                    continue
                if body.speaker and entry.speaker.lower() != body.speaker.lower():
                    continue
                results.append({
                    "session_id": session.id,
                    "session_name": session.name,
                    "entry_id": entry.id,
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "timestamp": _ts(entry),
                    "relevance": round(score, 3),
                })
        else:
            q = query.lower()
            for entry in session.transcript:
                if body.speaker and entry.speaker.lower() != body.speaker.lower():
                    continue
                if q in entry.text.lower():
                    results.append({
                        "session_id": session.id,
                        "session_name": session.name,
                        "entry_id": entry.id,
                        "speaker": entry.speaker,
                        "text": entry.text,
                        "timestamp": _ts(entry),
                    })
    else:
        # Cross-session search
        sessions = session_manager.list_sessions()[:50]

        if embedding_engine.available:
            search_type = "semantic"
            query_emb = await embedding_engine.embed_single(query)
            scored: list[tuple[float, dict]] = []

            for summary in sessions:
                sess = session_manager.get(summary.id)
                if not sess or not sess.transcript:
                    continue
                await _ensure_index(sess.id, sess.transcript)
                idx = get_index(sess.id)
                if idx.count == 0:
                    continue
                matches = idx.search(query_emb, top_k=body.max_results)
                entry_map = {e.id: e for e in sess.transcript}

                for entry_id, score in matches:
                    entry = entry_map.get(entry_id)
                    if not entry:
                        continue
                    if body.speaker and entry.speaker.lower() != body.speaker.lower():
                        continue
                    scored.append((score, {
                        "session_id": sess.id,
                        "session_name": sess.name,
                        "entry_id": entry.id,
                        "speaker": entry.speaker,
                        "text": entry.text,
                        "timestamp": _ts(entry),
                        "relevance": round(score, 3),
                    }))

            scored.sort(key=lambda x: x[0], reverse=True)
            results = [r for _, r in scored[:body.max_results]]
        else:
            q = query.lower()
            for summary in sessions:
                sess = session_manager.get(summary.id)
                if not sess:
                    continue
                for entry in sess.transcript:
                    if body.speaker and entry.speaker.lower() != body.speaker.lower():
                        continue
                    if q in entry.text.lower():
                        results.append({
                            "session_id": sess.id,
                            "session_name": sess.name,
                            "entry_id": entry.id,
                            "speaker": entry.speaker,
                            "text": entry.text,
                            "timestamp": _ts(entry),
                        })
                        if len(results) >= body.max_results:
                            break
                if len(results) >= body.max_results:
                    break

    return {"results": results[:body.max_results], "search_type": search_type}


# ── Helpers ──


def _session_to_markdown(session: Session) -> str:
    """Convert a session to a Markdown document."""
    parts: list[str] = [f"# {session.name}\n"]
    parts.append(f"**Created**: {session.created_at.isoformat()}  ")
    parts.append(f"**Status**: {session.status.value}\n")

    if session.topics:
        parts.append(f"**Topics**: {', '.join(session.topics)}\n")

    if session.participants:
        parts.append("## Participants\n")
        for p in session.participants:
            line = f"- **{p.display_name}**"
            if p.role:
                line += f" ({p.role})"
            parts.append(line)
        parts.append("")

    parts.append("## Transcript\n")
    for entry in session.transcript:
        display = session.get_display_name(entry.speaker)
        parts.append(f"**{display}**: {entry.text}\n")

    # Group notes by type
    note_groups = {
        "Action Items": [n for n in session.notes if n.type.value == "action_item"],
        "Decisions": [n for n in session.notes if n.type.value == "decision"],
        "Key Facts": [n for n in session.notes if n.type.value == "key_fact"],
        "Risks": [n for n in session.notes if n.type.value == "risk"],
    }

    has_notes = any(notes for notes in note_groups.values())
    if has_notes:
        parts.append("## Notes\n")
        for heading, notes in note_groups.items():
            if not notes:
                continue
            parts.append(f"### {heading}\n")
            for n in notes:
                content = f"~~{n.content}~~" if n.completed else n.content
                line = f"- {content}"
                if n.owner:
                    line += f" (Owner: {n.owner})"
                if n.due_date:
                    line += f" (Due: {n.due_date})"
                if n.completed:
                    line += " ✓"
                parts.append(line)
            parts.append("")

    return "\n".join(parts)
