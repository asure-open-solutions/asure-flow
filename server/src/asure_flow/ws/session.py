"""WebSocket endpoint for session-level AI events."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, asdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asure_flow.agent.context import build_context, build_prior_outputs, generate_summary, needs_summary_refresh
from asure_flow.agent.loop import run_agent
from asure_flow.agent.presets import build_system_prompt, DEFAULT_PRESET
from asure_flow.agent.router import get_router
from asure_flow.agent.specialists import get_enabled_specialists, run_specialists
from asure_flow.config import settings
from asure_flow.profile import profile, update_profile
from asure_flow.search.embeddings import embedding_engine
from asure_flow.search.index import get_index
from asure_flow.sessions.manager import session_manager
from asure_flow.sessions.models import FactCheck, NoteEntry, NoteType, SuggestionEntry

logger = logging.getLogger(__name__)
router = APIRouter()

AUTOSAVE_INTERVAL = 5  # seconds — keep crash-loss small during live sessions
EMBEDDING_SAVE_DELAY = 2.0

_embedding_locks: dict[str, asyncio.Lock] = {}
_embedding_save_tasks: dict[str, asyncio.Task] = {}

# ── Smart trigger constants ──
MIN_FIRE_INTERVAL = 3.0            # Minimum seconds between agent fires (cost throttle)
MONOLOGUE_INTERVAL = 15.0          # Periodic fire during long unbroken speech
SUMMARY_CHECK_INTERVAL = 5.0       # How often to check if rolling summary needs refresh
RERUN_CONTEXT_ENTRIES = 5          # How many recent entries to use for rerun triggers
HEARTBEAT_INTERVAL = 20.0

# Debounce delays (lowered since trivial gate saves us from wasted calls)
DELAY_IMMEDIATE = 0.0              # Other asked a question
DELAY_FAST = 0.3                   # Other → User transition (suggestions needed)
DELAY_QUICK = 0.5                  # Speaker changed
DELAY_NORMAL = 1.0                 # Same speaker continues

# ── Suggestion echo / dedup detection ──
ECHO_THRESHOLD = 0.6
DEDUP_THRESHOLD = 0.6
ECHO_SUGGESTION_LOOKBACK = 5
# Safety expiry: auto-release the delivery-lock after this many agent fires if the
# other party never speaks, so suggestions can't be suppressed forever.
SUGGESTION_LOCK_MAX_FIRES = 4

# ── Trivial content patterns ──
_TRIVIAL_PATTERN = re.compile(
    r'^(um+|uh+|hmm+|ah+|oh+|ok(ay)?|yeah|yep|yea|nah|no|yes|sure|right'
    r'|hi|hey|hello|bye|goodbye|thanks|thank you|sorry|excuse me'
    r'|mhm|hm|uh-huh|mm|so|well|like|you know|i mean)[\.\?\!,]*$',
    re.IGNORECASE,
)

_INTERROGATIVE_WORDS = frozenset({
    "what", "who", "where", "when", "why", "how",
    "can", "could", "would", "should", "is", "are",
    "do", "does", "did", "will", "have", "has",
})

_MIN_SUBSTANTIVE_WORDS = 3


# ── Trigger signal analysis ──


@dataclass
class TriggerSignals:
    """Cheap-to-compute signals about pending entries."""
    word_count: int
    is_trivial: bool
    has_question: bool
    speaker_changed: bool
    other_to_user: bool
    user_just_spoke: bool
    seconds_since_last_fire: float


def _is_trivial_entry(text: str) -> bool:
    """Check if a single entry is trivial filler."""
    return bool(_TRIVIAL_PATTERN.match(text.strip()))


def _has_question(text: str) -> bool:
    """Detect if text contains a question."""
    if "?" in text:
        return True
    first_word = text.strip().split()[0].lower() if text.strip() else ""
    return first_word in _INTERROGATIVE_WORDS


def _compute_signals(
    entries: list[tuple[str, str, str, bool]],
    last_speaker: str | None,
    last_fire_time: float,
) -> TriggerSignals:
    """Compute trigger signals from pending entries."""
    combined_text = " ".join(t for _, t, _, _ in entries)
    words = combined_text.split()
    word_count = len(words)

    is_trivial = all(_is_trivial_entry(t) for _, t, _, _ in entries if t.strip())
    has_q = any(_has_question(t) for _, t, _, is_u in entries if not is_u)

    last_entry_is_user = entries[-1][3] if entries else False
    prev_was_other = last_speaker is not None and last_speaker != "User"

    return TriggerSignals(
        word_count=word_count,
        is_trivial=is_trivial,
        has_question=has_q,
        speaker_changed=(last_speaker is not None and entries[-1][0] != last_speaker) if entries else False,
        other_to_user=last_entry_is_user and prev_was_other,
        user_just_spoke=last_entry_is_user,
        seconds_since_last_fire=time.monotonic() - last_fire_time,
    )


def _compute_trigger_delay(
    signals: TriggerSignals,
    response_profile: str = "balanced",
) -> float | None:
    """Determine trigger delay from signals. Returns None to skip entirely."""
    # Gate: skip trivial content
    if signals.is_trivial:
        return None

    # Gate: skip very short non-questions
    if signals.word_count < _MIN_SUBSTANTIVE_WORDS and not signals.has_question:
        return None

    # Priority-based delay
    if signals.has_question:
        delay = DELAY_IMMEDIATE
    elif signals.other_to_user:
        delay = DELAY_FAST
    elif signals.speaker_changed:
        delay = DELAY_QUICK
    else:
        delay = DELAY_NORMAL

    # Interactive profiles trade a little extra API usage for prompt replies.
    interval = {
        "realtime": 0.25,
        "balanced": 1.0,
        "quality": MIN_FIRE_INTERVAL,
    }.get(response_profile, 1.0)
    if signals.seconds_since_last_fire < interval:
        remaining = interval - signals.seconds_since_last_fire
        delay = max(delay, remaining)

    return delay


# ── Text helpers ──


def _normalize_words(text: str) -> set[str]:
    return set(re.findall(r'\w+', text.lower()))


def _suggestion_echo_score(user_text: str, suggestion_text: str) -> float:
    user_words = _normalize_words(user_text)
    sug_words = _normalize_words(suggestion_text)
    if len(user_words) < 3:
        return 0.0
    return len(user_words & sug_words) / len(user_words)


def _suggestion_similarity(a: str, b: str) -> float:
    words_a = _normalize_words(a)
    words_b = _normalize_words(b)
    if not words_a or not words_b:
        return 0.0
    overlap = len(words_a & words_b)
    return overlap / len(words_a | words_b)


def _is_user_speaker(session, speaker: str) -> bool:
    """Recognize the user even after their display label has been renamed."""
    if speaker == "User":
        return True
    return any(
        p.speaker_label == "User" and p.display_name == speaker
        for p in session.participants
    )


# ── Feature toggles ──


@dataclass
class FeatureToggles:
    fact_checking: bool = True
    suggestions: bool = True
    notes: bool = True
    search_transcript: bool = True
    search_sessions: bool = True
    web_search: bool = True
    format_code: bool = True
    deep_think: str = "off"
    agent_mode: str = "unified"   # "unified" | "specialists"
    parallel_tools: bool = False  # parallel tool execution in unified mode


def _resolve_system_prompt(toggles: FeatureToggles) -> str:
    if profile.custom_system_prompt:
        return profile.custom_system_prompt
    preset_id = profile.ai_preset or DEFAULT_PRESET
    toggles_dict = {k: v for k, v in asdict(toggles).items() if k != "deep_think"}
    return build_system_prompt(preset_id, toggles_dict, deep_think_mode=toggles.deep_think)


def _any_enabled(toggles: FeatureToggles) -> bool:
    return (
        toggles.fact_checking or toggles.suggestions or toggles.notes
        or toggles.search_transcript or toggles.search_sessions
        or toggles.web_search or toggles.format_code
    )


# ── WebSocket endpoint ──


@router.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str):
    """
    Session WebSocket — receives transcription events and streams AI analysis.

    Client → Server messages:
      {"type": "transcription", "speaker": str, "text": str}
      {"type": "config", ...toggle fields...}
      {"type": "rerun"}                — re-trigger agent on recent context
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
    trigger_task: asyncio.Task | None = None
    monologue_task: asyncio.Task | None = None
    summary_bg_task: asyncio.Task | None = None
    pending_entries: list[tuple[str, str, str, bool]] = []  # (speaker, text, entry_id, is_user)
    rolling_summary: str | None = None
    last_summarized_index = 0
    last_speaker: str | None = None
    last_fire_time: float = 0.0  # monotonic timestamp of last agent fire
    # Set while the user is delivering a suggestion we gave them. Suppresses new
    # suggestions until the other party speaks again, so the AI stops rewriting the
    # answer out from under the user mid-delivery. Auto-managed; replaces manual pin.
    suggestions_locked: bool = False
    # Fires the lock has been held without the other party speaking. A safety
    # expiry so the lock can never wedge if they never respond (monologue,
    # mislabeled audio): it auto-releases after SUGGESTION_LOCK_MAX_FIRES.
    suggestions_lock_age: int = 0

    # ── Agent fire logic ──

    async def _do_fire_agent(max_iterations: int = 5, realtime_turn: bool = False) -> None:
        nonlocal agent_task, last_fire_time, suggestions_locked, suggestions_lock_age

        entries = pending_entries[:]
        pending_entries.clear()
        if not entries:
            return

        llm_router = get_router()
        if not llm_router or not _any_enabled(toggles):
            return

        # Cancel previous agent task if still running (preemption)
        if agent_task and not agent_task.done():
            agent_task.cancel()
            try:
                await agent_task
            except asyncio.CancelledError:
                pass
            # Notify client of preemption
            try:
                await websocket.send_json({
                    "type": "ai_event",
                    "event": {"type": "preempted"},
                })
            except Exception:
                logger.debug("Failed to send preemption notice", exc_info=True)

        last_fire_time = time.monotonic()

        # Use whatever rolling summary is available (updated by background task)
        use_realtime_route = realtime_turn and profile.ai_response_profile != "quality"
        context = build_context(
            session,
            rolling_summary,
            recent_token_budget=2500 if use_realtime_route else 4500,
        )
        prior_outputs = build_prior_outputs(
            session,
            token_budget=600 if use_realtime_route else 1500,
        )
        system_prompt = _resolve_system_prompt(toggles)
        deep_think_enabled = toggles.deep_think != "off"
        session_context = session.context
        effective_ws = toggles.web_search and not profile.privacy_mode

        transcript_text = "\n".join(f"[{s}]: {t}" for s, t, _, _ in entries)
        last_entry_id = entries[-1][2]
        fire_started = time.perf_counter()
        first_result_sent = False

        # ── Suggestion delivery-lock ──
        was_locked = suggestions_locked
        other_spoke = any(not is_user for _, _, _, is_user in entries)

        if other_spoke:
            # Conversation returned to the other party — release the lock and do
            # NOT re-lock from trailing user echo in this same batch; a fresh
            # suggestion is exactly what's wanted now.
            suggestions_locked = False
            suggestions_lock_age = 0
        elif suggestions_locked:
            # Still the user's turn while locked — age the lock so it can't wedge
            # if the other party never speaks (monologue / mislabeled audio).
            suggestions_lock_age += 1
            if suggestions_lock_age >= SUGGESTION_LOCK_MAX_FIRES:
                suggestions_locked = False
                suggestions_lock_age = 0

        # Echo detection: only when the other party did NOT just speak. If the
        # user's speech is tracking a suggestion we gave, they're delivering it —
        # lock so the AI doesn't rewrite it mid-sentence.
        if not other_spoke and session.suggestions:
            recent_sugs = [s.text for s in session.suggestions[-ECHO_SUGGESTION_LOOKBACK:]]
            user_texts = [t for _, t, _, is_user in entries if is_user]
            if user_texts:
                combined_user = " ".join(user_texts)
                for sug_text in recent_sugs:
                    if _suggestion_echo_score(combined_user, sug_text) >= ECHO_THRESHOLD:
                        transcript_text = (
                            "[The user's speech echoes a prior suggestion.]\n\n"
                            + transcript_text
                        )
                        if not suggestions_locked:
                            suggestions_lock_age = 0
                        suggestions_locked = True
                        logger.debug("Echo detected: locking suggestions until the other party responds")
                        break

        # Suppress new suggestions while the lock is held (the other features still run).
        effective_suggestions = toggles.suggestions and not suggestions_locked

        # Tell the client when the lock flips so the UI can freeze/unfreeze the live
        # suggestion card automatically — no manual pin required.
        if toggles.suggestions and suggestions_locked != was_locked:
            try:
                await websocket.send_json({
                    "type": "ai_event",
                    "event": {"type": "suggestion_lock", "locked": suggestions_locked},
                })
            except Exception:
                logger.debug("Failed to send suggestion_lock notice", exc_info=True)

        async def process_agent():
            nonlocal first_result_sent
            try:
                background_stream = None
                if toggles.agent_mode == "specialists":
                    # Parallel specialist agents mode
                    specialists = get_enabled_specialists(
                        fact_checking=toggles.fact_checking,
                        suggestions=effective_suggestions,
                        notes=toggles.notes,
                        search_transcript=toggles.search_transcript,
                        search_sessions=toggles.search_sessions,
                        web_search=effective_ws,
                        format_code=toggles.format_code,
                        deep_think_mode=toggles.deep_think,
                    )
                    preset_id = profile.ai_preset or "general"
                    event_stream = run_specialists(
                        router=llm_router,
                        specialists=specialists,
                        transcript_text=transcript_text,
                        conversation_context=context,
                        session_context=session_context,
                        prior_outputs=prior_outputs,
                        preset_id=preset_id,
                        deep_think_mode=toggles.deep_think,
                        session=session,
                        parallel_tools=toggles.parallel_tools,
                    )
                else:
                    # Unified agent mode (default)
                    fast_lane = (
                        use_realtime_route
                        and effective_suggestions
                    )
                    event_stream = run_agent(
                        router=llm_router,
                        transcript_text=transcript_text,
                        conversation_context=context,
                        session_context=session_context,
                        prior_outputs=prior_outputs,
                        fact_checking=False if fast_lane else toggles.fact_checking,
                        suggestions=effective_suggestions,
                        notes=False if fast_lane else toggles.notes,
                        search_transcript=False if fast_lane else toggles.search_transcript,
                        search_sessions=False if fast_lane else toggles.search_sessions,
                        web_search=False if fast_lane else effective_ws,
                        format_code=False if fast_lane else toggles.format_code,
                        deep_think=False if fast_lane else deep_think_enabled,
                        system_prompt=(
                            system_prompt
                            + "\nFor this realtime pass, produce only a concise response suggestion."
                            if fast_lane else system_prompt
                        ),
                        session=session,
                        max_iterations=max_iterations,
                        parallel_tools=toggles.parallel_tools,
                        model_name="assistant_realtime" if use_realtime_route else "assistant",
                        request_timeout=12 if use_realtime_route else None,
                        max_completion_tokens=450 if use_realtime_route else None,
                        fallback_suggestion=fast_lane,
                    )

                    if fast_lane and (
                        toggles.fact_checking or toggles.notes or toggles.search_transcript
                        or toggles.search_sessions or effective_ws or toggles.format_code
                    ):
                        background_stream = run_agent(
                            router=llm_router,
                            transcript_text=transcript_text,
                            conversation_context=context,
                            session_context=session_context,
                            prior_outputs=prior_outputs,
                            fact_checking=toggles.fact_checking,
                            suggestions=False,
                            notes=toggles.notes,
                            search_transcript=toggles.search_transcript,
                            search_sessions=toggles.search_sessions,
                            web_search=effective_ws,
                            format_code=toggles.format_code,
                            deep_think=deep_think_enabled,
                            system_prompt=system_prompt,
                            session=session,
                            max_iterations=2,
                            parallel_tools=toggles.parallel_tools,
                            model_name="assistant",
                        )

                streams = [event_stream] + ([background_stream] if background_stream else [])
                for stream_index, stream in enumerate(streams):
                    async for event in stream:
                        if event.get("type") == "tool_result":
                            if not first_result_sent:
                                event["latency_ms"] = round(
                                    (time.perf_counter() - fire_started) * 1000,
                                    1,
                                )
                                first_result_sent = True
                            event["transcript_id"] = last_entry_id
                            _persist_tool_result(session, last_entry_id, event)
                        if event.get("type") == "done":
                            usage = event.get("usage", {})
                            session.token_usage.prompt_tokens += usage.get("prompt_tokens", 0)
                            session.token_usage.completion_tokens += usage.get("completion_tokens", 0)
                            if stream_index < len(streams) - 1:
                                continue
                        await websocket.send_json({"type": "ai_event", "event": event})
            except Exception:
                logger.exception("Agent processing error")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "AI processing failed",
                    })
                except Exception:
                    logger.debug("Failed to send agent error notification", exc_info=True)

        agent_task = asyncio.create_task(process_agent())

    async def _schedule_fire(
        delay: float,
        max_iterations: int = 5,
        realtime_turn: bool = False,
    ) -> None:
        if delay > 0:
            await asyncio.sleep(delay)
        await _do_fire_agent(max_iterations, realtime_turn)

    # ── Background summary refresh (off critical path) ──

    async def _summary_refresh_loop() -> None:
        nonlocal rolling_summary, last_summarized_index
        consecutive_failures = 0
        while True:
            await asyncio.sleep(SUMMARY_CHECK_INTERVAL)
            if agent_task and not agent_task.done():
                continue
            if not needs_summary_refresh(session, last_summarized_index):
                continue
            llm_router = get_router()
            if not llm_router:
                continue
            new_end = len(session.transcript) - 5
            if new_end <= last_summarized_index:
                continue
            try:
                rolling_summary = await generate_summary(
                    router=llm_router,
                    session=session,
                    existing_summary=rolling_summary,
                    start_index=last_summarized_index,
                    end_index=new_end,
                )
                last_summarized_index = new_end
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                logger.exception("Background summary refresh failed (attempt %d)", consecutive_failures)
                if consecutive_failures == 3:
                    try:
                        await websocket.send_json({"type": "warning", "message": "Rolling summary generation is failing repeatedly"})
                    except Exception:
                        pass

    # ── Monologue check ──

    async def _monologue_check() -> None:
        nonlocal trigger_task
        while True:
            await asyncio.sleep(MONOLOGUE_INTERVAL)
            if pending_entries:
                if trigger_task and not trigger_task.done():
                    trigger_task.cancel()
                    try:
                        await trigger_task
                    except (asyncio.CancelledError, Exception):
                        pass
                await _do_fire_agent()

    def _cancel_trigger() -> None:
        nonlocal trigger_task
        if trigger_task and not trigger_task.done():
            trigger_task.cancel()

    def _start_trigger(
        delay: float,
        max_iterations: int = 5,
        realtime_turn: bool = False,
    ) -> None:
        nonlocal trigger_task
        _cancel_trigger()
        trigger_task = asyncio.create_task(
            _schedule_fire(delay, max_iterations, realtime_turn)
        )

    # Auto-save task
    async def autosave_loop():
        while True:
            await asyncio.sleep(AUTOSAVE_INTERVAL)
            await session_manager.save_async(session)
            try:
                await websocket.send_json({"type": "session_saved"})
            except Exception:
                logger.debug("Autosave WS notification failed, continuing saves")

    async def heartbeat_loop():
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            try:
                await websocket.send_json({"type": "heartbeat", "timestamp": time.time()})
            except Exception:
                return

    save_task = asyncio.create_task(autosave_loop())
    heartbeat_task = asyncio.create_task(heartbeat_loop())
    monologue_task = asyncio.create_task(_monologue_check())
    summary_bg_task = asyncio.create_task(_summary_refresh_loop())

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Malformed JSON from client: %s", raw[:200])
                try:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON message"})
                except Exception:
                    logger.debug("Failed to send JSON error response", exc_info=True)
                continue
            if not isinstance(msg, dict):
                try:
                    await websocket.send_json({"type": "error", "message": "Expected JSON object"})
                except Exception:
                    logger.debug("Failed to send type error response", exc_info=True)
                continue
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
                toggles.agent_mode = msg.get("agent_mode", toggles.agent_mode)
                toggles.parallel_tools = msg.get("parallel_tools", toggles.parallel_tools)
                if "session_context" in msg:
                    session.context = msg["session_context"]
                if "privacy_mode" in msg:
                    try:
                        update_profile(privacy_mode=msg["privacy_mode"])
                        if msg["privacy_mode"]:
                            update_profile(pii_redaction=True, web_search=False)
                            toggles.web_search = False
                    except Exception:
                        logger.warning("Failed to persist privacy_mode setting", exc_info=True)
                if "pii_redaction" in msg:
                    try:
                        update_profile(pii_redaction=msg["pii_redaction"])
                    except Exception:
                        logger.warning("Failed to persist pii_redaction setting", exc_info=True)
                logger.info("Feature toggles updated: %s", toggles)

            elif msg_type == "transcription":
                raw_speaker = msg.get("speaker", "Unknown")
                text = msg.get("text", "")
                if not text.strip():
                    continue

                # PII redaction
                if profile.pii_redaction or profile.privacy_mode:
                    from asure_flow.safety.pii import redact_pii
                    text, pii_matches = redact_pii(text)
                    if pii_matches:
                        logger.debug("Redacted %d PII items", len(pii_matches))

                speaker = session.get_display_name(raw_speaker)

                audio_start = msg.get("audio_start")
                audio_end = msg.get("audio_end")
                entry_id = msg.get("entry_id")
                entry = session.add_transcript(
                    speaker, text, audio_start=audio_start, audio_end=audio_end,
                    entry_id=entry_id,
                )

                if embedding_engine.available:
                    asyncio.create_task(_embed_entry(session.id, entry.id, text))

                # ── Smart trigger: compute signals and decide ──
                is_user = raw_speaker == "User"
                pending_entries.append((speaker, text, entry.id, is_user))

                signals = _compute_signals(pending_entries, last_speaker, last_fire_time)
                last_speaker = raw_speaker

                delay = _compute_trigger_delay(signals, profile.ai_response_profile)
                if delay is not None:
                    # Use fast path (fewer iterations) for urgent scenarios
                    # Base on has_question directly — delay may be > 0 after throttle even for questions
                    interactive = signals.has_question or signals.speaker_changed
                    iters = 2 if signals.has_question else 5
                    if profile.ai_response_profile == "realtime" and interactive:
                        iters = 1
                    _start_trigger(
                        delay,
                        max_iterations=iters,
                        realtime_turn=interactive,
                    )
                else:
                    logger.debug("Skipping trivial segment: %r", text[:50])

            elif msg_type == "rerun":
                if session.transcript:
                    recent = session.transcript[-RERUN_CONTEXT_ENTRIES:]
                    pending_entries.clear()
                    for te in recent:
                        pending_entries.append((
                            te.speaker, te.text, te.id,
                            _is_user_speaker(session, te.speaker),
                        ))
                    _start_trigger(0.0)

            elif msg_type == "relabel":
                entry_id = msg.get("entry_id", "")
                new_speaker = msg.get("speaker", "")
                if entry_id and new_speaker:
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
                    await session_manager.save_async(session)
                    await websocket.send_json({
                        "type": "speaker_renamed",
                        "speaker_label": speaker_label,
                        "display_name": display_name,
                        "participant": participant.model_dump(mode="json"),
                    })

            elif msg_type == "end_session":
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
        for task in [save_task, heartbeat_task, trigger_task, monologue_task, agent_task, summary_bg_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        await _flush_embedding_index(session.id)
        await session_manager.save_async(session)


def _persist_tool_result(session, transcript_id: str, event: dict) -> None:
    """Save tool results (fact-checks, notes) into the session model."""
    event.setdefault("transcript_id", transcript_id)
    name = event.get("name", "")
    result = event.get("result", {})

    if name == "suggest_response":
        suggestion_text = result.get("suggestion", "")
        responding_to = result.get("responding_to", "")
        if suggestion_text:
            for recent in session.suggestions[-ECHO_SUGGESTION_LOOKBACK:]:
                if _suggestion_similarity(suggestion_text, recent.text) >= DEDUP_THRESHOLD:
                    logger.info("Suppressing duplicate suggestion (too similar to recent)")
                    event["result"] = {"suggestion": "", "suppressed": True}
                    return
            session.add_suggestion(suggestion_text, responding_to=responding_to)

    elif name == "fact_check":
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


async def _embed_entry(session_id: str, entry_id: str, text: str) -> None:
    try:
        lock = _embedding_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            embedding = await embedding_engine.embed_single(text)
            idx = get_index(session_id)
            if not idx.has_entry(entry_id):
                idx.add(entry_id, embedding)
        _schedule_embedding_save(session_id)
    except Exception:
        logger.debug("Failed to embed entry %s", entry_id, exc_info=True)


def _schedule_embedding_save(session_id: str) -> None:
    previous = _embedding_save_tasks.get(session_id)
    if previous and not previous.done():
        previous.cancel()

    async def save_later() -> None:
        try:
            await asyncio.sleep(EMBEDDING_SAVE_DELAY)
            lock = _embedding_locks.setdefault(session_id, asyncio.Lock())
            async with lock:
                await asyncio.to_thread(get_index(session_id).save)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("Failed to save embedding index for %s", session_id, exc_info=True)
        finally:
            if _embedding_save_tasks.get(session_id) is asyncio.current_task():
                _embedding_save_tasks.pop(session_id, None)

    _embedding_save_tasks[session_id] = asyncio.create_task(
        save_later(), name=f"embedding-save-{session_id}",
    )


async def _flush_embedding_index(session_id: str) -> None:
    pending = _embedding_save_tasks.pop(session_id, None)
    if pending and not pending.done():
        pending.cancel()
        try:
            await pending
        except asyncio.CancelledError:
            pass
    lock = _embedding_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        await asyncio.to_thread(get_index(session_id).save)
