import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeechPreprocessConfig:
    enabled: bool = True
    vad_enabled: bool = True
    vad_threshold: float = 0.5
    vad_neg_threshold: float = 0.35
    vad_min_speech_duration_ms: int = 200
    vad_max_speech_duration_s: float = 30.0
    vad_min_silence_duration_ms: int = 250
    vad_speech_pad_ms: int = 120
    vad_concat_silence_ms: int = 60

    denoise_enabled: bool = False
    denoise_strength: float = 0.8
    denoise_floor: float = 0.06
    denoise_n_fft: int = 512
    denoise_hop: int = 128


def _clamp_audio(audio: np.ndarray) -> np.ndarray:
    audio = np.nan_to_num(audio.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(audio, -1.0, 1.0, out=audio)


def _concat_segments(audio: np.ndarray, segments: Iterable[tuple[int, int]], gap_samples: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    silence = np.zeros((max(0, int(gap_samples)),), dtype=np.float32)
    for start, end in segments:
        start_i = int(max(0, start))
        end_i = int(min(len(audio), end))
        if end_i <= start_i:
            continue
        if parts and len(silence):
            parts.append(silence)
        parts.append(audio[start_i:end_i])
    if not parts:
        return np.array([], dtype=np.float32)
    return np.concatenate(parts).astype(np.float32, copy=False)


def extract_speech(audio: np.ndarray, sample_rate: int, cfg: SpeechPreprocessConfig) -> np.ndarray:
    audio = _clamp_audio(audio)
    if not cfg.enabled or not cfg.vad_enabled:
        return audio

    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception as e:
        logger.warning(f"VAD unavailable; skipping speech trimming: {e}")
        return audio

    try:
        vad_options = VadOptions(
            threshold=float(cfg.vad_threshold),
            neg_threshold=float(cfg.vad_neg_threshold),
            min_speech_duration_ms=int(cfg.vad_min_speech_duration_ms),
            max_speech_duration_s=float(cfg.vad_max_speech_duration_s),
            min_silence_duration_ms=int(cfg.vad_min_silence_duration_ms),
            speech_pad_ms=int(cfg.vad_speech_pad_ms),
        )
        stamps = get_speech_timestamps(audio, vad_options=vad_options, sampling_rate=int(sample_rate))
        gap_samples = int(sample_rate * (float(cfg.vad_concat_silence_ms) / 1000.0))
        return _concat_segments(audio, [(s["start"], s["end"]) for s in stamps], gap_samples=gap_samples)
    except Exception as e:
        logger.warning(f"VAD failed; skipping speech trimming: {e}")
        return audio


def _frame_signal(x: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if len(x) < frame_size:
        pad = np.zeros((frame_size - len(x),), dtype=np.float32)
        x = np.concatenate([x.astype(np.float32, copy=False), pad])

    n_frames = 1 + (len(x) - frame_size) // hop
    shape = (n_frames, frame_size)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()


def _istft(frames: np.ndarray, n_fft: int, hop: int, length: int) -> np.ndarray:
    window = np.hanning(n_fft).astype(np.float32)
    out_len = hop * (frames.shape[0] - 1) + n_fft
    y = np.zeros((out_len,), dtype=np.float32)
    wsum = np.zeros((out_len,), dtype=np.float32)

    for i in range(frames.shape[0]):
        start = i * hop
        y[start : start + n_fft] += frames[i] * window
        wsum[start : start + n_fft] += window * window

    y = np.divide(y, wsum, out=np.zeros_like(y), where=wsum > 1e-8)
    return y[:length].astype(np.float32, copy=False)


def spectral_denoise(
    audio: np.ndarray,
    sample_rate: int,
    cfg: SpeechPreprocessConfig,
    noise_reference: np.ndarray | None = None,
) -> np.ndarray:
    audio = _clamp_audio(audio)
    if not cfg.enabled or not cfg.denoise_enabled:
        return audio

    n_fft = int(cfg.denoise_n_fft)
    hop = int(cfg.denoise_hop)
    strength = float(cfg.denoise_strength)
    floor = float(cfg.denoise_floor)

    if n_fft <= 0 or hop <= 0 or hop > n_fft:
        return audio

    window = np.hanning(n_fft).astype(np.float32)

    x = audio.astype(np.float32, copy=False)
    frames = _frame_signal(x, n_fft, hop) * window[None, :]
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec).astype(np.float32, copy=False)

    if noise_reference is not None and len(noise_reference):
        nframes = _frame_signal(_clamp_audio(noise_reference), n_fft, hop) * window[None, :]
        nmag = np.abs(np.fft.rfft(nframes, axis=1)).astype(np.float32, copy=False)
        noise_mag = np.median(nmag, axis=0)
    else:
        # Fallback: estimate a conservative noise profile from the quietest frames.
        frame_energy = np.mean(mag, axis=1)
        if len(frame_energy) >= 8:
            idx = np.argsort(frame_energy)[: max(1, len(frame_energy) // 8)]
            noise_mag = np.median(mag[idx], axis=0)
        else:
            noise_mag = np.median(mag, axis=0)

    noise_mag = noise_mag.astype(np.float32, copy=False)
    phase = spec / (mag + 1e-8)
    clean_mag = np.maximum(mag - (strength * noise_mag[None, :]), mag * floor)
    spec_clean = clean_mag * phase
    frames_clean = np.fft.irfft(spec_clean, n=n_fft, axis=1).astype(np.float32, copy=False)
    y = _istft(frames_clean, n_fft=n_fft, hop=hop, length=len(x))
    return _clamp_audio(y)


def preprocess_audio(audio: np.ndarray, sample_rate: int, cfg: SpeechPreprocessConfig) -> np.ndarray:
    audio = _clamp_audio(audio)
    if not cfg.enabled:
        return audio

    # Optional: compute a noise reference from non-speech areas in the current buffer.
    noise_ref = None
    if cfg.vad_enabled and cfg.denoise_enabled:
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            vad_options = VadOptions(
                threshold=float(cfg.vad_threshold),
                neg_threshold=float(cfg.vad_neg_threshold),
                min_speech_duration_ms=int(cfg.vad_min_speech_duration_ms),
                max_speech_duration_s=float(cfg.vad_max_speech_duration_s),
                min_silence_duration_ms=int(cfg.vad_min_silence_duration_ms),
                speech_pad_ms=int(cfg.vad_speech_pad_ms),
            )
            stamps = get_speech_timestamps(audio, vad_options=vad_options, sampling_rate=int(sample_rate))
            if stamps:
                mask = np.ones((len(audio),), dtype=bool)
                for s in stamps:
                    mask[int(s["start"]) : int(s["end"])] = False
                noise_ref = audio[mask]
        except Exception:
            noise_ref = None

    audio = spectral_denoise(audio, sample_rate=sample_rate, cfg=cfg, noise_reference=noise_ref)
    audio = extract_speech(audio, sample_rate=sample_rate, cfg=cfg)
    return _clamp_audio(audio)


def _wav_read(path: str) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(path, "rb") as wf:
        nchannels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sampwidth == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth * 8}-bit")

    if nchannels > 1:
        x = x.reshape(-1, nchannels).mean(axis=1)
    return _clamp_audio(x), int(sample_rate)


def _wav_write(path: str, audio: np.ndarray, sample_rate: int) -> None:
    import wave

    audio = _clamp_audio(audio)
    pcm16 = (audio * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Speech-only preprocessing (VAD + optional denoise).")
    parser.add_argument("input_wav", help="Input WAV (PCM 16/32-bit).")
    parser.add_argument("output_wav", help="Output WAV (mono PCM 16-bit).")
    parser.add_argument("--disable-vad", action="store_true", help="Disable VAD speech trimming.")
    parser.add_argument("--enable-denoise", action="store_true", help="Enable spectral denoise.")
    parser.add_argument("--vad-threshold", type=float, default=0.5)
    parser.add_argument("--denoise-strength", type=float, default=0.8)
    args = parser.parse_args()

    inp, sr = _wav_read(args.input_wav)
    cfg = SpeechPreprocessConfig(
        enabled=True,
        vad_enabled=not args.disable_vad,
        vad_threshold=args.vad_threshold,
        denoise_enabled=bool(args.enable_denoise),
        denoise_strength=args.denoise_strength,
    )
    out = preprocess_audio(inp, sample_rate=sr, cfg=cfg)
    _wav_write(args.output_wav, out, sample_rate=sr)
    print(f"Wrote {args.output_wav} ({len(out)/sr:.2f}s @ {sr}Hz)")
