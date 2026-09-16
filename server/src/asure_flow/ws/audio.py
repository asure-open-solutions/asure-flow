"""WebSocket endpoint for real-time audio streaming and transcription."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from asure_flow.config import settings
from asure_flow.profile import profile
from asure_flow.transcription.engine import AudioBuffer, pcm16_bytes_to_float32, whisper_engine

logger = logging.getLogger(__name__)
router = APIRouter()

STREAM_MIC = 0
STREAM_SYSTEM = 1


def _resolve_capture_mode(requested_capture: str) -> tuple[bool, bool]:
    """Return whether mic/system should use server-side capture."""
    force_client = requested_capture == "client"
    return (
        not force_client and settings.audio_capture_source == "server",
        not force_client and bool(settings.system_device_id),
    )


@router.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    """
    Receives multiplexed audio from the client.

    Binary messages: first byte = stream ID (0=mic, 1=system), rest = 16-bit PCM @ 16 kHz mono.
    Sends back JSON transcription events.

    In server capture mode, incoming client audio is ignored — transcription events
    come from the AudioCaptureManager instead.
    """
    await websocket.accept()
    logger.info("Audio WebSocket connected: %s", websocket.client)

    requested_capture = websocket.query_params.get("capture", "auto")
    server_mic, server_system = _resolve_capture_mode(requested_capture)
    logger.info(
        "Audio capture mode: requested=%s mic=%s system=%s",
        requested_capture, "server" if server_mic else "client",
        "server" if server_system else "client",
    )

    if server_mic:
        await _handle_server_capture(websocket)
    elif server_system:
        await _handle_mixed_capture(websocket)
    else:
        await _handle_client_capture(websocket)


async def _handle_server_capture(websocket: WebSocket) -> None:
    """Server-side capture: relay transcription from capture manager to WebSocket."""
    from asure_flow.audio.manager import audio_capture_manager

    async def send_transcription(seg):
        await websocket.send_json({
            "type": "transcription",
            "speaker": seg.speaker,
            "text": seg.text,
            "start": seg.start,
            "end": seg.end,
        })

    audio_capture_manager.on_transcription = send_transcription

    try:
        while True:
            # Keep connection alive; ignore incoming client audio
            await websocket.receive_bytes()
    except WebSocketDisconnect:
        audio_capture_manager.on_transcription = None
        logger.info("Audio WebSocket disconnected (server capture mode)")
    except Exception:
        audio_capture_manager.on_transcription = None
        logger.exception("Audio WebSocket error (server capture mode)")


async def _handle_mixed_capture(websocket: WebSocket) -> None:
    """Mixed mode: mic audio from client, system audio from server loopback.

    Client sends mic PCM (stream_id=0) which we transcribe locally.
    Server capture manager handles system audio and relays transcription events.
    """
    from asure_flow.audio.manager import audio_capture_manager

    mic_buffer = AudioBuffer(whisper_engine, speaker_label="User")
    mic_queue = asyncio.Queue[bytes](maxsize=500)
    send_lock = asyncio.Lock()

    # Ensure system capture is running
    if settings.system_device_id and not audio_capture_manager._system_capture:
        try:
            await audio_capture_manager.start_system_capture(int(settings.system_device_id))
        except Exception:
            logger.warning("Failed to start system loopback capture", exc_info=True)

    async def send_transcription(seg):
        try:
            async with send_lock:
                await websocket.send_json({
                    "type": "transcription",
                    "speaker": seg.speaker,
                    "text": seg.text,
                    "start": seg.start,
                    "end": seg.end,
                })
        except Exception:
            logger.debug("Failed to send transcription to WebSocket", exc_info=True)

    audio_capture_manager.on_transcription = send_transcription

    async def process_mic() -> None:
        while True:
            pcm_data = await mic_queue.get()
            try:
                mic_buffer.add_audio(pcm_data)
                if mic_buffer.ready:
                    for seg in await mic_buffer.flush():
                        await send_transcription(seg)
            finally:
                mic_queue.task_done()

    mic_worker = asyncio.create_task(process_mic(), name="mixed-mic-worker")

    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) < 2:
                continue

            stream_id = data[0]
            pcm_data = data[1:]

            # Only process mic audio from client; system audio is handled server-side
            if stream_id == STREAM_MIC:
                await mic_queue.put(pcm_data)

    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected (mixed capture mode)")
    except Exception:
        logger.exception("Audio WebSocket error (mixed capture mode)")
    finally:
        audio_capture_manager.on_transcription = None
        mic_worker.cancel()
        await asyncio.gather(mic_worker, return_exceptions=True)
        try:
            remaining = await mic_buffer.flush()
            for seg in remaining:
                await send_transcription(seg)
        except Exception:
            logger.debug("Failed to flush remaining buffer on disconnect", exc_info=True)


def _get_speaker_tracker():
    """Create a SpeakerTracker if diarization is enabled and available."""
    if not profile.diarization_enabled:
        return None
    try:
        from asure_flow.transcription.diarization import diarization_engine
        if diarization_engine.available:
            from asure_flow.transcription.speaker_tracker import SpeakerTracker
            return SpeakerTracker(diarization_engine)
    except ImportError:
        pass
    return None


async def _handle_client_capture(websocket: WebSocket) -> None:
    """Client capture with independent bounded workers for mic and system audio."""
    mic_buffer = AudioBuffer(whisper_engine, speaker_label="User")
    system_buffer = AudioBuffer(whisper_engine, speaker_label="Third Party")
    speaker_tracker = _get_speaker_tracker()
    buffers = {STREAM_MIC: mic_buffer, STREAM_SYSTEM: system_buffer}
    queues = {
        STREAM_MIC: asyncio.Queue[bytes](maxsize=500),
        STREAM_SYSTEM: asyncio.Queue[bytes](maxsize=500),
    }
    send_lock = asyncio.Lock()

    async def send_json(message: dict) -> None:
        async with send_lock:
            await websocket.send_json(message)

    async def process_stream(stream_id: int) -> None:
        buf = buffers[stream_id]
        queue = queues[stream_id]
        while True:
            pcm_data = await queue.get()
            try:
                buf.add_audio(pcm_data)
                if stream_id == STREAM_SYSTEM and speaker_tracker is not None:
                    speaker_tracker.add_audio(pcm16_bytes_to_float32(pcm_data))

                if buf.ready:
                    segments = await buf.flush()
                    for seg in segments:
                        await send_json({
                            "type": "transcription",
                            "speaker": seg.speaker,
                            "text": seg.text,
                            "start": seg.start,
                            "end": seg.end,
                        })
                        if stream_id == STREAM_SYSTEM and speaker_tracker is not None:
                            speaker_tracker.add_segment(
                                entry_id=f"{seg.start:.3f}-{seg.end:.3f}",
                                start=seg.start,
                                end=seg.end,
                            )

                if (
                    stream_id == STREAM_SYSTEM
                    and speaker_tracker is not None
                    and speaker_tracker.ready
                ):
                    for relabel in await speaker_tracker.flush():
                        await send_json({
                            "type": "relabel",
                            "entry_id": relabel.entry_id,
                            "speaker": relabel.new_speaker,
                        })
            finally:
                queue.task_done()

    workers = [
        asyncio.create_task(process_stream(STREAM_MIC), name="audio-mic-worker"),
        asyncio.create_task(process_stream(STREAM_SYSTEM), name="audio-system-worker"),
    ]

    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) < 2:
                continue

            stream_id = data[0]
            pcm_data = data[1:]
            if stream_id in queues:
                # Backpressure starts only after ~10 seconds per stream. Until
                # then, receiving continues while Whisper decodes another chunk.
                await queues[stream_id].put(pcm_data)

    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected: %s", websocket.client)
    except Exception:
        logger.exception("Audio WebSocket error")
    finally:
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        for buf in [mic_buffer, system_buffer]:
            try:
                remaining = await buf.flush()
                for seg in remaining:
                    await websocket.send_json({
                        "type": "transcription",
                        "speaker": seg.speaker,
                        "text": seg.text,
                        "start": seg.start,
                        "end": seg.end,
                    })
            except Exception:
                logger.debug("Failed to flush remaining buffer on disconnect", exc_info=True)
