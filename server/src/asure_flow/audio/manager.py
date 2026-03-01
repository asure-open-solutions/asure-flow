"""Manages server-side audio capture lifecycle and transcription integration."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from asure_flow.config import settings
from asure_flow.transcription.engine import AudioBuffer, TranscriptSegment, whisper_engine

logger = logging.getLogger(__name__)

STREAM_MIC = 0
STREAM_SYSTEM = 1


class AudioCaptureManager:
    """Singleton that manages server-side audio capture sessions."""

    def __init__(self) -> None:
        self._mic_capture = None
        self._system_capture = None
        self._mic_buffer: Optional[AudioBuffer] = None
        self._system_buffer: Optional[AudioBuffer] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self.on_transcription: Optional[Callable[[TranscriptSegment], asyncio.coroutine]] = None

    async def start(
        self,
        mic_device_id: int | None = None,
        system_device_id: int | None = None,
    ) -> None:
        """Start server-side capture for the given devices.

        Args:
            mic_device_id: Input device for mic capture. None to skip.
            system_device_id: Output device for WASAPI loopback capture. None to skip.
        """
        from asure_flow.audio.capture import ServerAudioCapture

        self._mic_buffer = AudioBuffer(whisper_engine, speaker_label="User")
        self._system_buffer = AudioBuffer(whisper_engine, speaker_label="Third Party")

        if mic_device_id is not None:
            self._mic_capture = ServerAudioCapture()
            self._mic_capture.on_audio = self._handle_mic_audio
            await self._mic_capture.start(
                device_id=mic_device_id,
                stream_id=STREAM_MIC,
            )

        if system_device_id is not None:
            self._system_capture = ServerAudioCapture()
            self._system_capture.on_audio = self._handle_system_audio
            await self._system_capture.start(
                device_id=system_device_id,
                stream_id=STREAM_SYSTEM,
                loopback=True,
            )

        self._running = True
        self._task = asyncio.create_task(self._transcription_loop())
        logger.info(
            "Audio capture manager started (mic=%s, system=%s)",
            mic_device_id, system_device_id,
        )

    async def start_system_capture(self, device_id: int) -> None:
        """Start system loopback capture only (for mixed mode)."""
        from asure_flow.audio.capture import ServerAudioCapture

        if self._system_capture:
            return

        if self._system_buffer is None:
            self._system_buffer = AudioBuffer(whisper_engine, speaker_label="Third Party")

        self._system_capture = ServerAudioCapture()
        self._system_capture.on_audio = self._handle_system_audio
        await self._system_capture.start(
            device_id=device_id,
            stream_id=STREAM_SYSTEM,
            loopback=True,
        )

        if not self._running:
            self._running = True
            self._task = asyncio.create_task(self._transcription_loop())

        logger.info("System loopback capture started: device=%d", device_id)

    def stop_system_capture(self) -> None:
        """Stop only the system capture stream."""
        if self._system_capture:
            self._system_capture.stop()
            self._system_capture = None
            logger.info("System loopback capture stopped")

    def _handle_mic_audio(self, stream_id: int, pcm_bytes: bytes) -> None:
        if self._mic_buffer:
            self._mic_buffer.add_audio(pcm_bytes)

    def _handle_system_audio(self, stream_id: int, pcm_bytes: bytes) -> None:
        if self._system_buffer:
            self._system_buffer.add_audio(pcm_bytes)

    async def _transcription_loop(self) -> None:
        """Periodically flush buffers and emit transcription events."""
        while self._running:
            for buf in [self._mic_buffer, self._system_buffer]:
                if buf and buf.ready:
                    try:
                        segments = await buf.flush()
                        for seg in segments:
                            if self.on_transcription:
                                await self.on_transcription(seg)
                    except Exception:
                        logger.exception("Transcription error in capture manager")
            await asyncio.sleep(0.1)

    def stop(self) -> None:
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        if self._mic_capture:
            self._mic_capture.stop()
            self._mic_capture = None
        if self._system_capture:
            self._system_capture.stop()
            self._system_capture = None
        logger.info("Audio capture manager stopped")


audio_capture_manager = AudioCaptureManager()
