import numpy as np


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + (hz / 700.0))


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _frame_signal(x: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if len(x) < frame_size:
        pad = np.zeros((frame_size - len(x),), dtype=np.float32)
        x = np.concatenate([x, pad])

    n_frames = 1 + (len(x) - frame_size) // hop
    shape = (n_frames, frame_size)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()


def _mel_filterbank(*, sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    sr = int(sample_rate)
    n_fft = int(n_fft)
    n_mels = int(n_mels)
    fmin = float(max(0.0, fmin))
    fmax = float(min(float(sr) / 2.0, fmax))
    if n_mels <= 0 or n_fft <= 0 or sr <= 0 or fmax <= fmin:
        return np.zeros((0, 0), dtype=np.float32)

    n_freqs = n_fft // 2 + 1
    mmin = _hz_to_mel(np.array([fmin], dtype=np.float32))[0]
    mmax = _hz_to_mel(np.array([fmax], dtype=np.float32))[0]
    m_pts = np.linspace(mmin, mmax, num=(n_mels + 2), dtype=np.float32)
    hz_pts = _mel_to_hz(m_pts)
    bins = np.floor((n_fft + 1) * hz_pts / float(sr)).astype(np.int32)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        left = int(bins[m])
        center = int(bins[m + 1])
        right = int(bins[m + 2])
        if right <= left or center <= left or right <= center:
            continue

        if center > left:
            up = np.linspace(0.0, 1.0, num=(center - left), endpoint=False, dtype=np.float32)
            fb[m, left:center] = up
        if right > center:
            down = np.linspace(1.0, 0.0, num=(right - center), endpoint=False, dtype=np.float32)
            fb[m, center:right] = np.maximum(fb[m, center:right], down)

    # Normalize area to 1 (helps energy scale stability).
    denom = np.sum(fb, axis=1, keepdims=True) + 1e-8
    fb = fb / denom
    return fb


def _dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    n_mfcc = int(n_mfcc)
    n_mels = int(n_mels)
    if n_mfcc <= 0 or n_mels <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    n = np.arange(n_mels, dtype=np.float32)
    k = np.arange(n_mfcc, dtype=np.float32)[:, None]
    basis = np.cos((np.pi / float(n_mels)) * (n + 0.5) * k).astype(np.float32)
    return basis


def compute_voiceprint(audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
    """
    Best-effort session-local speaker fingerprint from raw audio.
    Uses MFCC-ish statistics + simple spectral features; no external deps.
    Returns a unit-norm vector suitable for cosine similarity.
    """
    if audio is None:
        return None
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim != 1:
        x = x.reshape(-1).astype(np.float32, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = np.clip(x, -1.0, 1.0, out=x)

    sr = int(sample_rate)
    if sr <= 0:
        return None

    # Too little speech -> unstable fingerprints.
    if x.size < int(sr * 0.35):
        return None

    # Pre-emphasis to highlight formants a bit.
    x = x.copy()
    x[1:] = x[1:] - 0.97 * x[:-1]

    frame_len = max(160, int(0.025 * sr))
    hop = max(80, int(0.010 * sr))
    frames = _frame_signal(x, frame_len, hop)
    if frames.size == 0:
        return None

    window = np.hanning(frame_len).astype(np.float32)
    frames *= window[None, :]

    n_fft = 1
    while n_fft < frame_len:
        n_fft <<= 1
    n_fft = max(256, min(2048, n_fft))

    spec = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spec) ** 2).astype(np.float32, copy=False)

    # Frame energy + basic silence rejection.
    frame_energy = np.mean(power, axis=1).astype(np.float32, copy=False)
    if not np.isfinite(frame_energy).any():
        return None
    thr = np.percentile(frame_energy, 40.0) * 0.6
    keep = frame_energy > max(1e-8, float(thr))
    if keep.sum() < 6:
        keep[:] = True

    power = power[keep]
    frames_kept = frames[keep]
    frame_energy = frame_energy[keep]

    n_mels = 40
    n_mfcc = 13
    fb = _mel_filterbank(sample_rate=sr, n_fft=n_fft, n_mels=n_mels, fmin=80.0, fmax=min(7600.0, sr / 2.0))
    if fb.size == 0:
        return None

    mel = power @ fb.T
    mel = np.log(mel + 1e-8).astype(np.float32, copy=False)
    dct = _dct_basis(n_mfcc, n_mels)
    mfcc = (mel @ dct.T).astype(np.float32, copy=False)

    # Simple spectral features (centroid + rolloff) from linear power.
    freqs = np.linspace(0.0, sr / 2.0, num=(n_fft // 2 + 1), dtype=np.float32)
    power_sum = np.sum(power, axis=1) + 1e-8
    centroid = (power @ freqs) / power_sum
    cumsum = np.cumsum(power, axis=1)
    roll_target = (power_sum * 0.85)[:, None]
    roll_idx = np.argmax(cumsum >= roll_target, axis=1)
    rolloff = freqs[roll_idx]

    # Zero crossing rate (from time-domain frames).
    signs = np.sign(frames_kept)
    signs[signs == 0] = 1.0
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1).astype(np.float32, copy=False)

    def _stats(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(v, dtype=np.float32)
        return np.mean(v, axis=0).astype(np.float32, copy=False), np.std(v, axis=0).astype(np.float32, copy=False)

    mfcc_mean, mfcc_std = _stats(mfcc)
    energy_mean, energy_std = _stats(frame_energy)
    centroid_mean, centroid_std = _stats(centroid)
    roll_mean, roll_std = _stats(rolloff)
    zcr_mean, zcr_std = _stats(zcr)

    vec = np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
            np.array([energy_mean, energy_std, centroid_mean, centroid_std, roll_mean, roll_std, zcr_mean, zcr_std], dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    norm = float(np.linalg.norm(vec))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return (vec / norm).astype(np.float32, copy=False)


class SpeakerDiarizer:
    def __init__(self, *, similarity_threshold: float = 0.84, max_speakers: int = 8, centroid_alpha: float = 0.25):
        self.similarity_threshold = float(similarity_threshold)
        self.max_speakers = int(max_speakers)
        self.centroid_alpha = float(centroid_alpha)
        self._speakers: list[dict] = []

    def reset(self) -> None:
        self._speakers = []

    @property
    def speaker_count(self) -> int:
        return len(self._speakers)

    def assign(self, voiceprint: np.ndarray | None) -> tuple[str | None, str | None]:
        if voiceprint is None:
            return None, None
        vp = np.asarray(voiceprint, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vp))
        if not np.isfinite(norm) or norm <= 1e-8:
            return None, None
        vp = vp / norm

        best_i = -1
        best_sim = -1.0
        for i, s in enumerate(self._speakers):
            c = s["centroid"]
            sim = float(np.dot(vp, c))
            if sim > best_sim:
                best_sim = sim
                best_i = i

        if best_i >= 0 and best_sim >= self.similarity_threshold:
            s = self._speakers[best_i]
            alpha = float(np.clip(self.centroid_alpha, 0.05, 0.75))
            new_c = (1.0 - alpha) * s["centroid"] + alpha * vp
            new_c_norm = float(np.linalg.norm(new_c)) or 1.0
            s["centroid"] = (new_c / new_c_norm).astype(np.float32, copy=False)
            s["count"] = int(s.get("count", 0)) + 1
            return str(s["id"]), str(s["label"])

        if len(self._speakers) >= self.max_speakers:
            return None, None

        new_idx = len(self._speakers) + 1
        speaker_id = f"spk{new_idx}"
        label = f"Speaker {new_idx}"
        self._speakers.append({"id": speaker_id, "label": label, "centroid": vp.astype(np.float32, copy=False), "count": 1})
        return speaker_id, label

