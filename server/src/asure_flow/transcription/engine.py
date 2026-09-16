"""Transcription engine — wraps faster-whisper with buffered real-time support."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from asure_flow.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# ── Whisper hallucination filters ──
# Segments with no_speech_prob above this are likely hallucinations on noise.
_NO_SPEECH_PROB_THRESHOLD = 0.55
# Segments with avg_logprob below this are low-confidence garbage.
_AVG_LOGPROB_THRESHOLD = -0.9
_COMPRESSION_RATIO_THRESHOLD = 2.4

# ── Cached VAD model (lazy-loaded once) ──
_vad_model = None
_vad_lock = threading.Lock()


def _get_cached_vad_model():
    global _vad_model
    if _vad_model is None:
        with _vad_lock:
            if _vad_model is None:
                from faster_whisper.vad import get_vad_model
                _vad_model = get_vad_model()
    return _vad_model


async def warm_vad() -> None:
    """Load and exercise Silero once so the first spoken segment stays fast."""
    def _warm() -> None:
        model = _get_cached_vad_model()
        model(np.zeros(512, dtype=np.float32))

    await asyncio.to_thread(_warm)


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
        # What actually loaded (may differ from what was requested after OOM fallback).
        self._loaded_model: str | None = None
        self._loaded_device: str | None = None
        self._loaded_compute_type: str | None = None
        self._degraded: bool = False  # True if we fell back to a weaker model/device
        self._loading: bool = False
        self._last_transcription_ms: float | None = None

    @property
    def status(self) -> dict:
        """Current engine state — surfaced via /health so the UI can warn on degradation."""
        return {
            "loaded": self._model is not None,
            "loading": self._loading,
            "requested_model": settings.whisper_model,
            "requested_device": settings.detect_device(),
            "model": self._loaded_model,
            "device": self._loaded_device,
            "compute_type": self._loaded_compute_type,
            "degraded": self._degraded,
            "last_transcription_ms": self._last_transcription_ms,
        }

    def _effective_beam_size(self) -> int:
        """Beam size: explicit setting wins; otherwise 5 on GPU, 1 on CPU.

        Beam search is much more accurate but costs decode time; a GPU has the
        headroom, a CPU does not. Keyed on the *actually loaded* device so an OOM
        fallback to CPU doesn't leave us doing slow beam search there.
        """
        if settings.whisper_beam_size > 0:
            return settings.whisper_beam_size
        return 5 if self._loaded_device == "cuda" else 1

    async def load(self, *, force: bool = False) -> None:
        """Load the Whisper model once, serialized across all audio streams."""
        async with self._lock:
            if self._model is not None and not force:
                return
            self._loading = True
            try:
                self._model = await asyncio.to_thread(self._load_model)
                if self._degraded:
                    logger.warning(
                        "Whisper running DEGRADED: requested %s on %s but loaded %s on %s (%s) "
                        "after out-of-memory fallback — transcription accuracy is reduced.",
                        settings.whisper_model, settings.detect_device(),
                        self._loaded_model, self._loaded_device, self._loaded_compute_type,
                    )
                else:
                    logger.info(
                        "Whisper model loaded: %s on %s (%s)",
                        self._loaded_model, self._loaded_device, self._loaded_compute_type,
                    )
            finally:
                self._loading = False

    def _load_model(self):
        from faster_whisper import WhisperModel

        _OOM_PHRASES = ("out of memory", "failed to allocate", "mkl_malloc")

        def _is_oom(exc: RuntimeError) -> bool:
            msg = str(exc).lower()
            return any(p in msg for p in _OOM_PHRASES)

        device = settings.detect_device()
        compute_type = settings.detect_compute_type()

        # Cascade: (model, device, compute_type)
        candidates = [(settings.whisper_model, device, compute_type)]
        if device == "cuda":
            candidates += [
                (settings.whisper_model, "cpu", "int8"),
                ("small", "cpu", "int8"),
            ]
        else:
            candidates.append(("small", "cpu", "int8"))

        last_exc: RuntimeError | None = None
        for model_name, dev, ct in candidates:
            is_fallback = (model_name, dev, ct) != (settings.whisper_model, device, compute_type)
            try:
                if is_fallback:
                    logger.warning(
                        "Retrying with model=%s device=%s compute_type=%s",
                        model_name, dev, ct,
                    )
                model = WhisperModel(model_name, device=dev, compute_type=ct)
                self._loaded_model = model_name
                self._loaded_device = dev
                self._loaded_compute_type = ct
                self._degraded = is_fallback
                return model
            except RuntimeError as exc:
                if _is_oom(exc):
                    last_exc = exc
                    continue
                raise
        raise RuntimeError(
            f"Could not load any Whisper model — all candidates OOM'd. "
            f"Last error: {last_exc}"
        ) from last_exc

    async def transcribe(
        self, audio: np.ndarray, initial_prompt: str | None = None,
    ) -> list[TranscriptSegment]:
        """Transcribe a float32 audio buffer. Returns a list of segments."""
        if self._model is None:
            await self.load()

        # faster-whisper/CTranslate2 model use is serialized. Mic and system
        # buffers can flush concurrently and sharing one model unsafely causes
        # intermittent corruption or device errors.
        async with self._lock:
            started = time.perf_counter()
            result = await asyncio.to_thread(
                self._transcribe_sync, audio, initial_prompt,
            )
            self._last_transcription_ms = round((time.perf_counter() - started) * 1000, 1)
            return result

    def _transcribe_sync(
        self, audio: np.ndarray, initial_prompt: str | None = None,
    ) -> list[TranscriptSegment]:
        kwargs: dict = dict(
            beam_size=self._effective_beam_size(),
            # AudioBuffer already gates flushes with Silero VAD (speech-then-silence).
            # A second VAD pass here is redundant and harmful for short (~1-2 s) chunks
            # that arrive from remote clients — it aggressively strips them as "silence".
            vad_filter=False,
            # Skip timestamp prediction — useless for pre-segmented chunks and saves decode time.
            without_timestamps=True,
            # Disable auto-conditioning on prior decoded text within a single transcribe call
            # to prevent repetition hallucinations. The explicit initial_prompt still provides context.
            condition_on_previous_text=False,
        )
        if settings.whisper_language:
            kwargs["language"] = settings.whisper_language
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        raw_segments, _info = self._model.transcribe(audio, **kwargs)
        results: list[TranscriptSegment] = []
        for seg in raw_segments:
            text = seg.text.strip()
            if not text:
                continue
            # Filter hallucinations: high no-speech probability
            if seg.no_speech_prob > _NO_SPEECH_PROB_THRESHOLD:
                logger.debug(
                    "Dropping segment (no_speech_prob=%.2f): %s",
                    seg.no_speech_prob, text,
                )
                continue
            # Filter hallucinations: very low confidence
            if seg.avg_logprob < _AVG_LOGPROB_THRESHOLD:
                logger.debug(
                    "Dropping segment (avg_logprob=%.2f): %s",
                    seg.avg_logprob, text,
                )
                continue
            if getattr(seg, "compression_ratio", 0.0) > _COMPRESSION_RATIO_THRESHOLD:
                logger.debug(
                    "Dropping repetitive segment (compression_ratio=%.2f): %s",
                    seg.compression_ratio, text,
                )
                continue
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
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples: int = 0
        self._prev_text: str = ""  # last flush output — used as Whisper prompt context

        # Derived sample counts from settings
        self._min_samples = int(settings.vad_min_buffer_sec * SAMPLE_RATE)
        self._max_samples = int(settings.vad_max_buffer_sec * SAMPLE_RATE)
        self._silence_windows = max(1, int(settings.vad_silence_ms / 1000 * SAMPLE_RATE) // 512)
        self._speech_windows = max(
            1,
            int(settings.vad_min_speech_ms / 1000 * SAMPLE_RATE) // 512,
        )
        self._vad_check_interval = max(
            512,
            int(settings.vad_check_interval_ms / 1000 * SAMPLE_RATE),
        )

        # Rate-limiting state for VAD checks
        self._last_vad_len: int = 0
        self._cached_ready: bool = False
        self._has_speech: bool = False  # set True when any window exceeds threshold

    def _get_buffer(self) -> np.ndarray:
        """Concatenate all queued chunks into a single array."""
        if not self._chunks:
            return np.array([], dtype=np.float32)
        if len(self._chunks) == 1:
            return self._chunks[0]
        return np.concatenate(list(self._chunks))

    def add_audio(self, pcm_bytes: bytes) -> None:
        chunk = pcm16_bytes_to_float32(pcm_bytes)
        self._chunks.append(chunk)
        self._total_samples += len(chunk)

    @property
    def ready(self) -> bool:
        buf_len = self._total_samples
        if buf_len < self._min_samples:
            return False
        if buf_len >= self._max_samples:
            return True
        # Rate-limit: only re-run VAD after a fraction of the silence window
        if buf_len - self._last_vad_len < self._vad_check_interval:
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
        model = _get_cached_vad_model()
        audio = self._get_buffer()

        # Pad to a multiple of 512 (VAD window size)
        remainder = len(audio) % 512
        if remainder:
            audio = np.pad(audio, (0, 512 - remainder))

        probs = model(audio).flatten()

        if len(probs) < self._silence_windows:
            return False

        speech_mask = probs >= settings.vad_speech_threshold

        # Require sustained speech rather than one noisy window. This rejects
        # keyboard clicks, bumps, and short background bursts before Whisper.
        if not self._has_speech:
            run = 0
            for is_speech in speech_mask:
                run = run + 1 if is_speech else 0
                if run >= self._speech_windows:
                    self._has_speech = True
                    break

        # Only flush when speech was detected AND trailing windows are now silent
        if not self._has_speech:
            return False

        return bool(np.all(probs[-self._silence_windows:] < settings.vad_speech_threshold))

    async def flush(self) -> list[TranscriptSegment]:
        """Transcribe the buffer and return segments.

        Because we flush at silence boundaries, no overlap is kept — a clean cut.
        """
        if self._total_samples == 0:
            return []

        audio = self._get_buffer()
        duration = len(audio) / SAMPLE_RATE
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        had_speech = self._has_speech
        logger.info(
            "AudioBuffer flush [%s]: %.2fs, RMS=%.4f, peak=%.4f, had_speech=%s",
            self.speaker_label, duration, rms, peak, had_speech,
        )

        # Clear buffer state before transcription — keep audio in local var
        self._chunks.clear()
        self._total_samples = 0
        self._last_vad_len = 0
        self._cached_ready = False
        self._has_speech = False

        # Skip transcription when VAD never detected speech (e.g. max-buffer
        # timeout on ambient noise) — avoids Whisper hallucinations.
        if not had_speech or rms < settings.audio_min_rms:
            if had_speech:
                logger.info(
                    "Dropping low-energy buffer [%s]: RMS %.4f < %.4f",
                    self.speaker_label, rms, settings.audio_min_rms,
                )
            return []

        # Pass previous text as prompt so Whisper keeps sentence context
        prompt = self._prev_text[-200:] if self._prev_text else None
        try:
            segments = await self.engine.transcribe(audio, initial_prompt=prompt)
        except Exception:
            # Re-inject audio so the next flush retries it
            self._chunks.append(audio)
            self._total_samples = len(audio)
            self._has_speech = True
            raise

        for seg in segments:
            seg.speaker = self.speaker_label

        # Update context for next flush
        if segments:
            self._prev_text = " ".join(s.text for s in segments)
        return segments

    def clear(self) -> None:
        self._chunks.clear()
        self._total_samples = 0
        self._prev_text = ""
        self._last_vad_len = 0
        self._cached_ready = False
        self._has_speech = False


# Singleton engine
whisper_engine = WhisperEngine()


