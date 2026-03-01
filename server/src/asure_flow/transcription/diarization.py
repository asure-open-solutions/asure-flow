"""Speaker diarization engine — wraps pyannote.audio for speaker identification."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from asure_flow.config import settings

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class DiarSegment:
    start: float
    end: float
    speaker: str  # pyannote's raw label, e.g. "SPEAKER_00"


class DiarizationEngine:
    """Manages the pyannote diarization pipeline."""

    def __init__(self) -> None:
        self._pipeline = None
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    async def load(self) -> None:
        """Load the diarization pipeline. Requires a HuggingFace token."""
        if not settings.hf_diarization_token:
            logger.info("Diarization unavailable: no HuggingFace token configured")
            return

        loop = asyncio.get_event_loop()
        try:
            self._pipeline = await loop.run_in_executor(None, self._load_pipeline)
            self._available = True
            logger.info("Diarization pipeline loaded (device: %s)", settings.diarization_device or "auto")
        except ImportError:
            logger.info("pyannote.audio not installed — diarization unavailable")
        except Exception:
            logger.warning("Failed to load diarization pipeline", exc_info=True)

    def _load_pipeline(self):
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=settings.hf_diarization_token,
        )

        device = settings.diarization_device
        if not device:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        if device == "cuda":
            import torch
            pipeline.to(torch.device("cuda"))

        return pipeline

    async def diarize(self, audio: np.ndarray) -> list[DiarSegment]:
        """Run diarization on a float32 mono audio buffer.

        Returns a list of segments with pyannote speaker labels.
        """
        if not self._available or self._pipeline is None:
            return []

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._diarize_sync, audio)

    def _diarize_sync(self, audio: np.ndarray) -> list[DiarSegment]:
        import torch

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        input_data = {"waveform": waveform, "sample_rate": SAMPLE_RATE}

        diarization = self._pipeline(input_data)

        return [
            DiarSegment(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]


diarization_engine = DiarizationEngine()
