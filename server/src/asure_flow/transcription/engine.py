"""Transcription engine — wraps faster-whisper with buffered real-time support."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from asure_flow.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# How often (in samples) to re-run the VAD check inside the `ready` property.
_VAD_CHECK_INTERVAL = int(0.5 * SAMPLE_RATE)  # ~500 ms
# Speech probability below this value is treated as silence by the flush gate.
_VAD_SILENCE_THRESHOLD = 0.35


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str = "Unknown"


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Convert raw 16-bit signed PCM bytes to float32 array normalised to [-1, 1]."""
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


class WhisperEngine:
    """Manages the faster-whisper model and provides buffered transcription."""

    def __init__(self) -> None:
        self._model = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        """Load the whisper model (call once at server startup)."""
        loop = asyncio.get_event_loop()
        self._model = await loop.run_in_executor(None, self._load_model)
        logger.info(
            "Whisper model loaded: %s on %s (%s)",
            settings.whisper_model,
            settings.detect_device(),
            settings.detect_compute_type(),
        )

    def _load_model(self):
        from faster_whisper import WhisperModel

        return WhisperModel(
            settings.whisper_model,
            device=settings.detect_device(),
            compute_type=settings.detect_compute_type(),
        )

    async def transcribe(
        self, audio: np.ndarray, initial_prompt: str | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe a float32 audio buffer. Returns a list of segments."""
        if self._model is None:
            raise RuntimeError("Whisper model not loaded — call load() first")

        loop = asyncio.get_event_loop()
        segments = await loop.run_in_executor(
            None, self._transcribe_sync, audio, initial_prompt,
        )
        return segments

    def _transcribe_sync(
        self, audio: np.ndarray, initial_prompt: str | None = None,
    ) -> list[TranscriptSegment]:
        kwargs: dict = dict(
            beam_size=5,
            # AudioBuffer already gates flushes with Silero VAD (speech-then-silence).
            # A second VAD pass here is redundant and harmful for short (~1-2 s) chunks
            # that arrive from remote clients — it aggressively strips them as "silence".
            vad_filter=False,
        )
        if settings.whisper_language:
            kwargs["language"] = settings.whisper_language
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        raw_segments, _info = self._model.transcribe(audio, **kwargs)
        results: list[TranscriptSegment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if text:
                results.append(TranscriptSegment(start=seg.start, end=seg.end, text=text))
        return results


class AudioBuffer:
    """Accumulates PCM audio and flushes for transcription when silence is detected.

    Instead of flushing at a fixed interval (which can cut speech mid-sentence),
    the buffer uses Silero VAD to detect trailing silence after speech.  It only
    triggers a flush when the speaker has paused, producing cleaner segments and
    eliminating overlap-induced duplication.
    """

    def __init__(self, engine: WhisperEngine, speaker_label: str = "Unknown") -> None:
        self.engine = engine
        self.speaker_label = speaker_label
        self._buffer = np.array([], dtype=np.float32)
        self._prev_text: str = ""  # last flush output — used as Whisper prompt context

        # Derived sample counts from settings
        self._min_samples = int(settings.vad_min_buffer_sec * SAMPLE_RATE)
        self._max_samples = int(settings.vad_max_buffer_sec * SAMPLE_RATE)
        self._silence_windows = max(1, int(settings.vad_silence_ms / 1000 * SAMPLE_RATE) // 512)

        # Rate-limiting state for VAD checks
        self._last_vad_len: int = 0
        self._cached_ready: bool = False
        self._has_speech: bool = False  # set True when any window exceeds threshold

    def add_audio(self, pcm_bytes: bytes) -> None:
        chunk = pcm16_bytes_to_float32(pcm_bytes)
        self._buffer = np.concatenate([self._buffer, chunk])

    @property
    def ready(self) -> bool:
        buf_len = len(self._buffer)
        if buf_len < self._min_samples:
            return False
        if buf_len >= self._max_samples:
            return True
        # Rate-limit: only re-run VAD after ~500 ms of new audio
        if buf_len - self._last_vad_len < _VAD_CHECK_INTERVAL:
            return self._cached_ready
        self._cached_ready = self._check_vad_state()
        self._last_vad_len = buf_len
        return self._cached_ready

    def _check_vad_state(self) -> bool:
        """Run Silero VAD and check for speech followed by trailing silence.

        Returns True only if the buffer contains at least some speech AND
        the trailing windows are silence (i.e. the speaker has paused).
        Pure-silence buffers are never flushed, preventing wasted Whisper calls.
        """
        from faster_whisper.vad import get_vad_model

        model = get_vad_model()
        audio = self._buffer.copy()

        # Pad to a multiple of 512 (VAD window size)
        remainder = len(audio) % 512
        if remainder:
            audio = np.pad(audio, (0, 512 - remainder))

        probs = model(audio).flatten()

        if len(probs) < self._silence_windows:
            return False

        # Track whether the buffer ever contained speech
        if not self._has_speech:
            self._has_speech = bool(np.any(probs >= _VAD_SILENCE_THRESHOLD))

        # Only flush when speech was detected AND trailing windows are now silent
        if not self._has_speech:
            return False

        return bool(np.all(probs[-self._silence_windows:] < _VAD_SILENCE_THRESHOLD))

    async def flush(self) -> list[TranscriptSegment]:
        """Transcribe the buffer and return segments.

        Because we flush at silence boundaries, no overlap is kept — a clean cut.
        """
        if len(self._buffer) == 0:
            return []

        audio = self._buffer.copy()
        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        logger.info(
            "AudioBuffer flush [%s]: %.2fs, RMS=%.4f, peak=%.4f, had_speech=%s",
            self.speaker_label, duration, rms, peak, self._has_speech,
        )

        # Clean cut — no overlap needed when flushing at silence boundaries
        self._buffer = np.array([], dtype=np.float32)
        self._last_vad_len = 0
        self._cached_ready = False
        self._has_speech = False

        # Pass previous text as prompt so Whisper keeps sentence context
        prompt = self._prev_text[-200:] if self._prev_text else None
        segments = await self.engine.transcribe(audio, initial_prompt=prompt)

        for seg in segments:
            seg.speaker = self.speaker_label

        # Update context for next flush
        if segments:
            self._prev_text = " ".join(s.text for s in segments)
        return segments

    def clear(self) -> None:
        self._buffer = np.array([], dtype=np.float32)
        self._prev_text = ""
        self._last_vad_len = 0
        self._cached_ready = False
        self._has_speech = False


# Singleton engine
whisper_engine = WhisperEngine()
