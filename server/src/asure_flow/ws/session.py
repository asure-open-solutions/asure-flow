"""WebSocket endpoint for session-level AI events."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asure_flow.agent.context import build_context, generate_summary, needs_summary_refresh
from asure_flow.agent.loop import run_agent
from asure_flow.agent.presets import build_system_prompt, DEFAULT_PRESET
from asure_flow.agent.router import get_router
from asure_flow.config import settings
from asure_flow.search.embeddings import embedding_engine
from asure_flow.search.index import get_index
from asure_flow.sessions.manager import session_manager
from asure_flow.sessions.models import FactCheck, NoteEntry, NoteType

logger = logging.getLogger(__name__)
router = APIRouter()

AUTOSAVE_INTERVAL = 30  # seconds


@dataclass
class FeatureToggles:
    # AI features
    fact_checking: bool = True
    suggestions: bool = True
    notes: bool = True
    # AI tools
    search_transcript: bool = True
    search_sessions: bool = True
    web_search: bool = True
    format_code: bool = True
    # Deep think mode: "off", "auto", "always"
    deep_think: str = "off"


def _resolve_system_prompt(toggles: FeatureToggles) -> str:
    """Resolve the system prompt — custom prompt verbatim, or dynamic build from preset + toggles."""
    if settings.custom_system_prompt:
        return settings.custom_system_prompt

    preset_id = settings.ai_preset or DEFAULT_PRESET
    toggles_dict = {
        k: v for k, v in asdict(toggles).items() if k != "deep_think"
    }
    return build_system_prompt(
        preset_id,
        toggles_dict,
        deep_think_mode=toggles.deep_think,
    )


def _any_enabled(toggles: FeatureToggles) -> bool:
    """Check if any feature or tool is enabled."""
    return (
        toggles.fact_checking
        or toggles.suggestions
        or toggles.notes
        or toggles.search_transcript
        or toggles.search_sessions
        or toggles.web_search
        or toggles.format_code
    )


@router.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str):
    """
    Session WebSocket — receives transcription events and streams AI analysis.

    Client → Server messages:
      {"type": "transcription", "speaker": str, "text": str}
      {"type": "config", ...toggle fields...}
      {"type": "rename_speaker", "speaker_label": str, "display_name": str, "role"?: str}
      {"type": "end_session"}

    Server → Client messages:
      {"type": "ai_event", "event": {...}}   — agent events
      {"type": "speaker_renamed", "participant": {...}}
      {"type": "session_saved"}
      {"type": "error", "message": str}
    """
    await websocket.accept()
    logger.info("Session WebSocket connected: session=%s", session_id)

    session = session_manager.get(session_id)
    if not session:
        await websocket.send_json({"type": "error", "message": f"Session {session_id} not found"})
        await websocket.close()
        return

    # Initialize toggles from persisted session settings (if any)
    toggles = FeatureToggles()
    if session.settings:
        overrides = session.settings.model_dump(exclude_none=True)
        for k, v in overrides.items():
            if hasattr(toggles, k):
                setattr(toggles, k, v)
    agent_task: asyncio.Task | None = None
    rolling_summary: str | None = None
    last_summarized_index = 0

    # Auto-save task
    async def autosave_loop():
        while True:
            await asyncio.sleep(AUTOSAVE_INTERVAL)
            session_manager.save(session)
            try:
                await websocket.send_json({"type": "session_saved"})
            except Exception:
                break

    save_task = asyncio.create_task(autosave_loop())

    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "config":
                toggles.fact_checking = msg.get("fact_checking", toggles.fact_checking)
                toggles.suggestions = msg.get("suggestions", toggles.suggestions)
                toggles.notes = msg.get("notes", toggles.notes)
                toggles.search_transcript = msg.get("search_transcript", toggles.search_transcript)
                toggles.search_sessions = msg.get("search_sessions", toggles.search_sessions)
                toggles.web_search = msg.get("web_search", toggles.web_search)
                toggles.format_code = msg.get("format_code", toggles.format_code)
                toggles.deep_think = msg.get("deep_think", toggles.deep_think)
                # Session context update
                if "session_context" in msg:
                    session.context = msg["session_context"]
                # Privacy mode update
                if "privacy_mode" in msg:
                    try:
                        from asure_flow.config import update_settings
                        update_settings(privacy_mode=msg["privacy_mode"])
                        if msg["privacy_mode"]:
                            update_settings(pii_redaction=True)
                            toggles.web_search = False
                    except Exception:
                        logger.warning("Failed to persist privacy_mode setting", exc_info=True)
                if "pii_redaction" in msg:
                    try:
                        from asure_flow.config import update_settings
                        update_settings(pii_redaction=msg["pii_redaction"])
                    except Exception:
                        logger.warning("Failed to persist pii_redaction setting", exc_info=True)
                logger.info("Feature toggles updated: %s", toggles)

            elif msg_type == "transcription":
                speaker = msg.get("speaker", "Unknown")
                text = msg.get("text", "")
                if not text.strip():
                    continue

                # PII redaction
                if settings.pii_redaction or settings.privacy_mode:
                    from asure_flow.safety.pii import redact_pii
                    text, pii_matches = redact_pii(text)
                    if pii_matches:
                        logger.debug("Redacted %d PII items", len(pii_matches))

                # Privacy mode: force-disable web search
                effective_web_search = toggles.web_search
                if settings.privacy_mode:
                    effective_web_search = False

                # Resolve display name for speaker
                speaker = session.get_display_name(speaker)

                # Add to session transcript (include audio timing for diarization)
                audio_start = msg.get("audio_start")
                audio_end = msg.get("audio_end")
                entry_id = msg.get("entry_id")
                entry = session.add_transcript(
                    speaker, text, audio_start=audio_start, audio_end=audio_end,
                    entry_id=entry_id,
                )

                # Generate embedding for semantic search (fire-and-forget)
                if embedding_engine.available:
                    asyncio.create_task(_embed_entry(session.id, entry.id, text))

                # Run AI agent if any features are enabled and a router is available
                llm_router = get_router()
                if llm_router and _any_enabled(toggles):
                    # Refresh rolling summary if enough new entries have accumulated
                    if needs_summary_refresh(session, last_summarized_index):
                        new_end = len(session.transcript) - 5  # leave recent ones unsummarized
                        if new_end > last_summarized_index:
                            rolling_summary = await generate_summary(
                                router=llm_router,
                                session=session,
                                existing_summary=rolling_summary,
                                start_index=last_summarized_index,
                                end_index=new_end,
                            )
                            last_summarized_index = new_end

                    context = build_context(session, rolling_summary)
                    system_prompt = _resolve_system_prompt(toggles)
                    deep_think_enabled = toggles.deep_think != "off"
                    session_context = session.context

                    async def process_agent():
                        try:
                            async for event in run_agent(
                                router=llm_router,
                                transcript_text=f"[{speaker}]: {text}",
                                conversation_context=context,
                                session_context=session_context,
                                fact_checking=toggles.fact_checking,
                                suggestions=toggles.suggestions,
                                notes=toggles.notes,
                                search_transcript=toggles.search_transcript,
                                search_sessions=toggles.search_sessions,
                                web_search=effective_web_search,
                                format_code=toggles.format_code,
                                deep_think=deep_think_enabled,
                                system_prompt=system_prompt,
                                session=session,
                            ):
                                # Persist tool results into the session
                                if event["type"] == "tool_result":
                                    _persist_tool_result(session, entry.id, event)

                                await websocket.send_json({"type": "ai_event", "event": event})
                        except Exception:
                            logger.exception("Agent processing error")
                            try:
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "AI processing failed",
                                })
                            except Exception:
                                pass

                    # Cancel previous agent task if still running and wait for it
                    if agent_task and not agent_task.done():
                        agent_task.cancel()
                        try:
                            await agent_task
                        except asyncio.CancelledError:
                            pass
                    agent_task = asyncio.create_task(process_agent())

            elif msg_type == "relabel":
                # Speaker diarization relabel: update transcript entry speaker
                entry_id = msg.get("entry_id", "")
                new_speaker = msg.get("speaker", "")
                if entry_id and new_speaker:
                    # Match by server-side ID or by audio timing key
                    for te in session.transcript:
                        audio_key = (
                            f"{te.audio_start:.3f}-{te.audio_end:.3f}"
                            if te.audio_start is not None and te.audio_end is not None
                            else None
                        )
                        if te.id == entry_id or audio_key == entry_id:
                            te.speaker = new_speaker
                            break

            elif msg_type == "rename_speaker":
                speaker_label = msg.get("speaker_label", "")
                display_name = msg.get("display_name", "")
                role = msg.get("role")
                if speaker_label and display_name:
                    participant = session.rename_speaker(speaker_label, display_name, role)
                    session_manager.save(session)
                    await websocket.send_json({
                        "type": "speaker_renamed",
                        "participant": participant.model_dump(mode="json"),
                    })

            elif msg_type == "end_session":
                # Extract topics and entities before ending
                llm_router = get_router()
                if llm_router and session.transcript:
                    try:
                        from asure_flow.memory.topics import extract_topics
                        from asure_flow.memory.entities import extract_entities
                        session.topics = await extract_topics(llm_router, session)
                        session.entities = await extract_entities(llm_router, session)
                    except Exception:
                        logger.warning("Failed to extract topics/entities", exc_info=True)

                session_manager.end_session(session_id)
                await websocket.send_json({"type": "session_ended"})
                break

    except WebSocketDisconnect:
        logger.info("Session WebSocket disconnected: session=%s", session_id)
    except Exception:
        logger.exception("Session WebSocket error")
    finally:
        save_task.cancel()
        if agent_task and not agent_task.done():
            agent_task.cancel()
        # Final save
        session_manager.save(session)


def _persist_tool_result(session, transcript_id: str, event: dict) -> None:
    """Save tool results (fact-checks, notes) into the session model."""
    name = event.get("name", "")
    result = event.get("result", {})

    if name == "fact_check":
        claims = result.get("claims", [])
        checks = [
            FactCheck(
                claim=c["claim"],
                verdict=c["verdict"],
                reasoning=c["reasoning"],
                fallacy=c.get("fallacy"),
            )
            for c in claims
            if "claim" in c and "verdict" in c and "reasoning" in c
        ]
        session.add_fact_checks(transcript_id, checks)

    elif name == "extract_notes":
        note_entries: list[NoteEntry] = []
        # Handle action_items: may be strings or structured objects with owner/due_date
        for item in result.get("action_items", []):
            if isinstance(item, dict):
                content = item.get("content", "")
                if content:
                    note_entries.append(NoteEntry(
                        type=NoteType.ACTION_ITEM,
                        content=content,
                        owner=item.get("owner"),
                        due_date=item.get("due_date"),
                    ))
            elif isinstance(item, str) and item:
                note_entries.append(NoteEntry(type=NoteType.ACTION_ITEM, content=item))
        # decisions, key_facts, risks remain as string arrays
        for note_type, key in [
            (NoteType.DECISION, "decisions"),
            (NoteType.KEY_FACT, "key_facts"),
            (NoteType.RISK, "risks"),
        ]:
            for item in result.get(key, []):
                if item:
                    note_entries.append(NoteEntry(type=note_type, content=item))
        session.add_notes(note_entries)

    elif name == "web_search":
        # Persist web search results as key facts for reference
        web_results = result.get("results", [])
        if web_results:
            note_entries = []
            for r in web_results[:3]:
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                url = r.get("url", "")
                if title and snippet:
                    note_entries.append(NoteEntry(
                        type=NoteType.KEY_FACT,
                        content=f"[Web] {title}: {snippet} ({url})",
                    ))
            if note_entries:
                session.add_notes(note_entries)

    # search_transcript, search_sessions, format_code, deep_think — ephemeral, not persisted


async def _embed_entry(session_id: str, entry_id: str, text: str) -> None:
    """Background task: embed a transcript entry for semantic search."""
    try:
        embedding = await embedding_engine.embed_single(text)
        idx = get_index(session_id)
        idx.add(entry_id, embedding)
        idx.save()
    except Exception:
        logger.debug("Failed to embed entry %s", entry_id, exc_info=True)
