from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class SegmenterConfig:
    sample_rate: int = 16000
    pre_roll_s: float = 0.30
    end_silence_s: float = 0.65
    max_utterance_s: float = 12.0

    # Adaptive RMS trigger
    start_noise_rms: float = 0.004
    noise_alpha: float = 0.08
    min_trigger_rms: float = 0.008
    trigger_multiplier: float = 3.2


class EnergyUtteranceSegmenter:
    """
    Lightweight energy-based utterance segmentation.

    Feed it fixed-size float32 chunks (e.g. 1024 samples @ 16kHz). It groups chunks into an
    "utterance" and returns the buffered chunks once enough trailing silence has been observed.
    """

    def __init__(self, cfg: SegmenterConfig):
        self.cfg = cfg
        self.sample_rate = int(cfg.sample_rate)
        self.end_silence_samples = int(self.sample_rate * float(cfg.end_silence_s))
        self.max_utterance_samples = int(self.sample_rate * float(cfg.max_utterance_s))
        self.pre_roll_samples = int(self.sample_rate * float(cfg.pre_roll_s))

        self.noise_rms = float(cfg.start_noise_rms)
        self._pre_roll: deque[np.ndarray] = deque()
        self._pre_roll_len = 0

        self._speaking = False
        self._silence_samples = 0
        self._utt_parts: list[np.ndarray] = []
        self._utt_len = 0

    @staticmethod
    def _rms(x: np.ndarray) -> float:
        x = x.astype(np.float32, copy=False)
        return float(np.sqrt(np.mean(x * x))) if x.size else 0.0

    def _push_pre_roll(self, chunk: np.ndarray) -> None:
        self._pre_roll.append(chunk)
        self._pre_roll_len += int(len(chunk))
        while self._pre_roll_len > self.pre_roll_samples and self._pre_roll:
            dropped = self._pre_roll.popleft()
            self._pre_roll_len -= int(len(dropped))

    def push(self, chunk: np.ndarray) -> list[np.ndarray] | None:
        """
        Push one chunk. Returns a completed utterance (list of chunks) or None.
        May also return an utterance when max_utterance_s is reached.
        """
        chunk = np.nan_to_num(chunk.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        if chunk.size == 0:
            return None

        self._push_pre_roll(chunk)

        level = self._rms(chunk)
        trigger = max(float(self.cfg.min_trigger_rms), float(self.noise_rms) * float(self.cfg.trigger_multiplier))
        is_speech = level >= trigger

        if not self._speaking:
            # Update noise floor during non-speech.
            a = float(self.cfg.noise_alpha)
            self.noise_rms = (1.0 - a) * float(self.noise_rms) + a * level

        if is_speech:
            if not self._speaking:
                self._speaking = True
                self._silence_samples = 0
                self._utt_parts = list(self._pre_roll)
                self._utt_len = int(sum(len(p) for p in self._utt_parts))
                if (not self._utt_parts) or (self._utt_parts[-1] is not chunk):
                    self._utt_parts.append(chunk)
                    self._utt_len += int(len(chunk))
            else:
                self._utt_parts.append(chunk)
                self._utt_len += int(len(chunk))
            self._silence_samples = 0

            if self._utt_len >= self.max_utterance_samples:
                parts = self._utt_parts
                self._speaking = False
                self._silence_samples = 0
                self._utt_parts = []
                self._utt_len = 0
                return parts

            return None

        # silence chunk
        if self._speaking:
            self._silence_samples += int(len(chunk))
            if self._silence_samples >= self.end_silence_samples:
                parts = self._utt_parts
                self._speaking = False
                self._silence_samples = 0
                self._utt_parts = []
                self._utt_len = 0
                return parts

        return None

    def flush(self) -> list[np.ndarray] | None:
        if not self._speaking or not self._utt_parts:
            return None
        parts = self._utt_parts
        self._speaking = False
        self._silence_samples = 0
        self._utt_parts = []
        self._utt_len = 0
        return parts
