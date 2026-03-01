"""Speaker tracker — accumulates system audio for diarization and relabels segments."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import numpy as np

from asure_flow.config import settings
from asure_flow.transcription.diarization import DiarizationEngine, DiarSegment

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class PendingSegment:
    """A transcript segment waiting for speaker identification."""
    entry_id: str
    start: float  # offset within the rolling audio window
    end: float
    original_speaker: str


@dataclass
class RelabelEvent:
    """Emitted when a segment's speaker label is updated."""
    entry_id: str
    new_speaker: str


class SpeakerTracker:
    """Accumulates system audio in a rolling window, runs diarization,
    and produces relabel events for transcript entries.

    The rolling window is longer than the transcription buffer (default 20s)
    so pyannote has enough context for consistent speaker clustering.
    """

    def __init__(self, diarization_engine: DiarizationEngine) -> None:
        self._engine = diarization_engine
        self._buffer_sec = settings.diarization_buffer_sec
        self._buffer_size = int(self._buffer_sec * SAMPLE_RATE)
        self._audio = np.array([], dtype=np.float32)
        self._audio_offset: float = 0.0  # cumulative seconds trimmed

        # Segments waiting for diarization
        self._pending: list[PendingSegment] = []

        # Stable speaker name mapping: pyannote ID -> "Speaker N"
        self._speaker_map: dict[str, str] = {}
        self._next_speaker_id = 1

        self._running = False

    def add_audio(self, pcm_float: np.ndarray) -> None:
        """Append audio samples to the rolling window."""
        self._audio = np.concatenate([self._audio, pcm_float])

    def add_segment(self, entry_id: str, start: float, end: float) -> None:
        """Register a transcript segment for future speaker attribution.

        ``start``/``end`` are offsets relative to the current audio window.
        """
        abs_start = self._audio_offset + start
        abs_end = self._audio_offset + end
        self._pending.append(PendingSegment(
            entry_id=entry_id,
            start=abs_start,
            end=abs_end,
            original_speaker="Third Party",
        ))

    @property
    def ready(self) -> bool:
        """True when the audio window is full enough for diarization."""
        return len(self._audio) >= self._buffer_size

    async def flush(self) -> list[RelabelEvent]:
        """Run diarization on the current window and relabel pending segments.

        Trims the window to keep only the most recent overlap portion.
        """
        if self._running or len(self._audio) == 0:
            return []

        self._running = True
        try:
            audio = self._audio.copy()
            audio_duration = len(audio) / SAMPLE_RATE

            diar_segments = await self._engine.diarize(audio)
            if not diar_segments:
                return []

            relabels = self._match_and_relabel(diar_segments, audio_duration)

            # Trim buffer: keep last 5 seconds as overlap for continuity
            overlap_samples = int(5.0 * SAMPLE_RATE)
            if len(self._audio) > overlap_samples:
                trimmed = len(self._audio) - overlap_samples
                self._audio = self._audio[-overlap_samples:]
                self._audio_offset += trimmed / SAMPLE_RATE

            # Remove fully processed pending segments
            cutoff = self._audio_offset
            self._pending = [p for p in self._pending if p.end > cutoff]

            return relabels
        finally:
            self._running = False

    def _match_and_relabel(
        self,
        diar_segments: list[DiarSegment],
        audio_duration: float,
    ) -> list[RelabelEvent]:
        """Match pending transcript segments to diarization output by time overlap."""
        relabels: list[RelabelEvent] = []
        window_start = self._audio_offset

        for pending in self._pending:
            best_speaker: str | None = None
            best_overlap = 0.0

            # Convert pending times to window-relative
            p_start = pending.start - window_start
            p_end = pending.end - window_start

            if p_end < 0 or p_start > audio_duration:
                continue  # segment outside this window

            for ds in diar_segments:
                overlap_start = max(p_start, ds.start)
                overlap_end = min(p_end, ds.end)
                overlap = max(0.0, overlap_end - overlap_start)

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = ds.speaker

            if best_speaker is not None and best_overlap > 0:
                stable_name = self._stable_name(best_speaker)
                relabels.append(RelabelEvent(
                    entry_id=pending.entry_id,
                    new_speaker=stable_name,
                ))

        return relabels

    def _stable_name(self, pyannote_id: str) -> str:
        """Map a pyannote speaker ID to a stable 'Speaker N' label."""
        if pyannote_id not in self._speaker_map:
            self._speaker_map[pyannote_id] = f"Speaker {self._next_speaker_id}"
            self._next_speaker_id += 1
        return self._speaker_map[pyannote_id]

    def clear(self) -> None:
        self._audio = np.array([], dtype=np.float32)
        self._audio_offset = 0.0
        self._pending.clear()
        self._speaker_map.clear()
        self._next_speaker_id = 1
