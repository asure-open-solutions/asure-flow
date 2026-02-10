from faster_whisper import WhisperModel
import numpy as np
import threading
import logging
from dataclasses import dataclass
import os
import ctypes
from pathlib import Path

logger = logging.getLogger(__name__)

from backend.speech_processing import SpeechPreprocessConfig, preprocess_audio
from backend.diarization import compute_voiceprint

_CUDA_DLL_DIR_HANDLES: list[object] = []


def _looks_like_missing_cuda_runtime(exc: BaseException) -> bool:
    msg = str(exc or "").casefold()
    if not msg:
        return False

    # Common Windows CUDA runtime failures when ctranslate2 is built for CUDA but the
    # CUDA runtime isn't installed / not on PATH.
    if ".dll" in msg and ("not found" in msg or "cannot be loaded" in msg or "could not locate" in msg):
        if any(tok in msg for tok in ("cublas", "cudart", "cufft", "curand", "cusolver", "cusparse", "cudnn")):
            return True
        if msg.startswith("library ") and " is not found" in msg:
            return True

    return False


def _candidate_cuda_bin_dirs() -> list[Path]:
    dirs: list[Path] = []

    # CUDA toolkit env vars often exist after install.
    for k, v in os.environ.items():
        if not isinstance(k, str) or not k.upper().startswith("CUDA_PATH"):
            continue
        if not v:
            continue
        root = Path(v)
        for p in (root / "bin", root / "bin" / "x64"):
            if p.is_dir():
                dirs.append(p)

    # Common default install location.
    base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
    if base.is_dir():
        for d in sorted(base.glob("v*"), reverse=True):
            for p in (d / "bin", d / "bin" / "x64"):
                if p.is_dir():
                    dirs.append(p)

    out: list[Path] = []
    seen = set()
    for d in dirs:
        key = str(d).casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out


def _ensure_windows_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return
    if not hasattr(os, "add_dll_directory"):
        return

    # Keep handles alive for the process lifetime so directories remain active.
    for d in _candidate_cuda_bin_dirs():
        try:
            handle = os.add_dll_directory(str(d))
            _CUDA_DLL_DIR_HANDLES.append(handle)
        except Exception:
            continue


def _windows_cuda_dlls_available() -> bool:
    if os.name != "nt":
        return True

    # Make CUDA bins visible to dynamic loader in this process.
    _ensure_windows_cuda_dll_dirs()

    # For current faster-whisper/ctranslate2 wheels on Windows, runtime typically needs:
    # - cuBLAS 12: cublas64_12.dll
    # - cuDNN 9:   cudnn_ops64_9.dll (or cudnn64_9.dll on some layouts)
    required_cublas = ("cublas64_12.dll",)
    required_cudnn = ("cudnn_ops64_9.dll", "cudnn64_9.dll")

    def _has_any(names: tuple[str, ...]) -> bool:
        for dll in names:
            try:
                ctypes.WinDLL(dll)
                return True
            except OSError:
                continue
        return False

    has_cublas = _has_any(required_cublas)
    has_cudnn = _has_any(required_cudnn)
    if has_cublas and has_cudnn:
        return True

    # Fallback: direct load from known CUDA install paths.
    for d in _candidate_cuda_bin_dirs():
        for dll in (*required_cublas, *required_cudnn):
            p = d / dll
            if not p.is_file():
                continue
            try:
                ctypes.WinDLL(str(p))
            except OSError:
                continue

    # Re-check after direct loads.
    has_cublas = False
    has_cudnn = False
    for dll in required_cublas:
        try:
            ctypes.WinDLL(dll)
            has_cublas = True
            break
        except OSError:
            continue
    for dll in required_cudnn:
        try:
            ctypes.WinDLL(dll)
            has_cudnn = True
            break
        except OSError:
            continue

    return has_cublas and has_cudnn


@dataclass(frozen=True)
class TranscriptionChunk:
    text: str
    voiceprint: np.ndarray | None = None

class TranscriptionEngine:
    def __init__(
        self,
        model_size="tiny",
        device="cpu",
        compute_type: str | None = None,
        *,
        sample_rate: int = 16000,
        preprocess: SpeechPreprocessConfig | None = None,
        whisper_vad_filter: bool = True,
        chunk_duration_s: float = 2.4,
        chunk_overlap_s: float = 0.24,
        beam_size: int | None = None,
        condition_on_previous_text: bool = False,
    ):
        self._lock = threading.Lock()
        self.model_size = str(model_size or "tiny").strip() or "tiny"

        device = (str(device or "cpu").strip().lower() or "cpu")
        if device in ("gpu", "cuda"):
            device = "cuda"
        if device not in ("cpu", "cuda"):
            device = "cpu"

        if device == "cuda":
            try:
                if not _windows_cuda_dlls_available():
                    raise RuntimeError("Missing CUDA runtime DLLs (e.g. cublas64_12.dll / cudnn_ops64_9.dll)")

                import ctranslate2

                get_count = getattr(ctranslate2, "get_cuda_device_count", None)
                if callable(get_count) and int(get_count() or 0) <= 0:
                    raise RuntimeError("No CUDA devices detected")
            except Exception as e:
                logger.warning(f"CUDA not available ({e}); using CPU for transcription.")
                device = "cpu"

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        compute_type = str(compute_type or "").strip().lower() or ("float16" if device == "cuda" else "int8")

        self.device = device
        self.compute_type = compute_type
        self._load_model(device=self.device, compute_type=self.compute_type)
        self.sample_rate = int(sample_rate)
        self.preprocess = preprocess or SpeechPreprocessConfig()
        self.whisper_vad_filter = bool(whisper_vad_filter)
        self.chunk_duration_s = float(max(1.0, min(8.0, float(chunk_duration_s or 2.4))))
        self.chunk_overlap_s = float(max(0.0, min(1.2, float(chunk_overlap_s or 0.24))))
        self.condition_on_previous_text = bool(condition_on_previous_text)
        if beam_size is None:
            # Favor latency for small/real-time models.
            self.beam_size = 1 if self.model_size in ("tiny", "base", "small") else 3
        else:
            self.beam_size = int(max(1, min(8, int(beam_size))))
        # If preprocess VAD trims everything but the buffer still has meaningful energy,
        # retry once using the raw buffer + Whisper VAD to avoid dropped phrases.
        self.vad_rescue_min_rms = 0.004
        self.vad_rescue_min_peak = 0.020
        self.language_lock_min_probability = 0.80
        self.detected_language: str | None = None
        logger.info("Whisper model loaded.")

    def _load_model(self, *, device: str, compute_type: str) -> None:
        logger.info(f"Loading Whisper model: {self.model_size} on {device} ({compute_type})...")
        try:
            self.model = WhisperModel(self.model_size, device=device, compute_type=compute_type)
            self.device = device
            self.compute_type = compute_type
        except Exception as e:
            if device != "cpu":
                logger.warning(f"Failed to load Whisper on {device}; falling back to CPU: {e}")
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
                self.device = "cpu"
                self.compute_type = "int8"
            else:
                raise

    @staticmethod
    def _audio_energy(audio: np.ndarray) -> tuple[float, float]:
        if audio is None:
            return 0.0, 0.0
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0, 0.0
        peak = float(np.max(np.abs(arr)))
        rms = float(np.sqrt(np.mean(arr * arr)))
        return rms, peak

    @staticmethod
    def _normalize_transcript_text(text: str) -> str:
        import re

        t = " ".join((text or "").split()).strip()
        if not t:
            return ""

        # Common STT stutters.
        # Examples: "100% percent" -> "100%", "100 percent %" -> "100%"
        t = re.sub(r"\b(\d+(?:\.\d+)?)\s*%\s*percent\b", r"\1%", t, flags=re.I)
        t = re.sub(r"\b(\d+(?:\.\d+)?)\s*percent\s*%\b", r"\1%", t, flags=re.I)

        # Collapse repeated short phrases/words commonly produced around chunk seams.
        phrase_pat = re.compile(
            r"\b([a-z][a-z0-9']{2,}\s+[a-z][a-z0-9']{2,})\b(?:[\s,.;:!?-]+\1\b)+",
            flags=re.I,
        )
        word_pat = re.compile(
            r"\b([a-z][a-z0-9']{2,})\b(?:[\s,.;:!?-]+\1\b)+",
            flags=re.I,
        )

        prev = None
        while prev != t:
            prev = t
            t = phrase_pat.sub(lambda m: m.group(1), t)
            t = word_pat.sub(lambda m: m.group(1), t)

        # Collapse short prefix fragments followed by a longer completion.
        # Examples: "wor worse", "clean cleaner", "env environmental".
        def _collapse_prefix_stutter(m):
            a = m.group(1)
            b = m.group(2)
            a_cf = a.casefold()
            b_cf = b.casefold()
            # Keep conservative to avoid changing real word pairs.
            if len(a_cf) < 3 or len(a_cf) > 6:
                return m.group(0)
            if not b_cf.startswith(a_cf):
                return m.group(0)
            if len(b_cf) - len(a_cf) > 7:
                return m.group(0)
            return b

        prev = None
        while prev != t:
            prev = t
            t = re.sub(
                r"\b([a-z][a-z0-9']{2,5})\b[\s,.;:!?-]+([a-z][a-z0-9']{2,})\b",
                _collapse_prefix_stutter,
                t,
                flags=re.I,
            )

        # Drop chopped stem carryover after sentence seams.
        # Example: "overloads. loads the grid" -> "overloads the grid"
        def _drop_chopped_stem_after_seam(m):
            full = m.group(1)
            stem = m.group(3)
            full_cf = full.casefold()
            stem_cf = stem.casefold()
            if full_cf == stem_cf:
                return m.group(0)
            if len(stem_cf) < 4:
                return m.group(0)
            if full_cf.endswith(stem_cf):
                return full
            return m.group(0)

        prev = None
        while prev != t:
            prev = t
            t = re.sub(
                r"\b([a-z][a-z0-9']{4,})\b([.?!])\s+([a-z][a-z0-9']{3,})\b",
                _drop_chopped_stem_after_seam,
                t,
                flags=re.I,
            )

        t = re.sub(r"\s+([,.;:!?])", r"\1", t)
        return t.strip()

    @staticmethod
    def _trim_leading_overlap(previous_text: str, current_text: str) -> str:
        import re

        a = " ".join((previous_text or "").split()).strip()
        b = " ".join((current_text or "").split()).strip()
        if not a or not b:
            return b

        a_tokens = re.findall(r"[a-z0-9']+", a.casefold(), flags=re.I)
        b_tokens = re.findall(r"[a-z0-9']+", b.casefold(), flags=re.I)
        if not a_tokens or not b_tokens:
            return b

        max_overlap = min(12, len(a_tokens), len(b_tokens))
        overlap = 0
        for n in range(max_overlap, 1, -1):
            if a_tokens[-n:] == b_tokens[:n]:
                overlap = n
                break

        def _trim_prefix_tokens(text: str, token_count: int) -> str:
            if token_count <= 0:
                return text
            matches = list(re.finditer(r"[a-z0-9']+", text, flags=re.I))
            if token_count > len(matches):
                return text
            cut = matches[token_count - 1].end()
            return re.sub(r"^[\s,.;:!?-]+", "", text[cut:])

        if overlap > 0:
            b = _trim_prefix_tokens(b, overlap).strip()
            return b

        tail = a_tokens[-1]
        head = b_tokens[0]
        if tail == head and len(tail) >= 3:
            return _trim_prefix_tokens(b, 1).strip()

        # Handle chopped stem seam: "overloads. loads the grid".
        if len(head) >= 4 and tail.endswith(head) and tail != head:
            return _trim_prefix_tokens(b, 1).strip()

        return b

    @classmethod
    def _merge_segment_texts(cls, segments: list[str]) -> str:
        merged = ""
        for raw in segments:
            seg = cls._normalize_transcript_text(raw)
            if not seg:
                continue
            if merged:
                seg = cls._trim_leading_overlap(merged, seg)
                seg = cls._normalize_transcript_text(seg)
                if not seg:
                    continue
                merged = f"{merged} {seg}".strip()
                merged = cls._normalize_transcript_text(merged)
            else:
                merged = seg
        return cls._normalize_transcript_text(merged)

    def _transcribe_with_fallback(self, audio: np.ndarray, *, force_whisper_vad: bool | None = None):
        if force_whisper_vad is None:
            use_whisper_vad = bool(self.whisper_vad_filter)
            if self.preprocess is not None and bool(getattr(self.preprocess, "enabled", False)) and bool(getattr(self.preprocess, "vad_enabled", False)):
                # Avoid paying VAD cost twice when preprocess already trimmed speech.
                use_whisper_vad = False
        else:
            use_whisper_vad = bool(force_whisper_vad)

        def _run_transcribe(input_audio: np.ndarray):
            kwargs = {
                "beam_size": self.beam_size,
                "vad_filter": use_whisper_vad,
                "condition_on_previous_text": bool(getattr(self, "condition_on_previous_text", False)),
                # Mild anti-repeat defaults to reduce seam loops without hurting latency too much.
                "repetition_penalty": 1.04,
                "no_repeat_ngram_size": 3,
                # Favor deterministic decoding and lower overhead for live transcription.
                "temperature": 0.0,
                "best_of": 1,
                "patience": 1.0,
                "without_timestamps": True,
            }
            if self.detected_language:
                kwargs["language"] = self.detected_language
            try:
                segments, info = self.model.transcribe(input_audio, **kwargs)
            except TypeError:
                # Older faster-whisper builds may not expose some kwargs.
                kwargs.pop("condition_on_previous_text", None)
                kwargs.pop("repetition_penalty", None)
                kwargs.pop("no_repeat_ngram_size", None)
                kwargs.pop("temperature", None)
                kwargs.pop("best_of", None)
                kwargs.pop("patience", None)
                kwargs.pop("without_timestamps", None)
                segments, info = self.model.transcribe(input_audio, **kwargs)
            seg_list = list(segments)
            if self.detected_language is None:
                lang = getattr(info, "language", None)
                try:
                    prob = float(getattr(info, "language_probability", 0.0) or 0.0)
                except Exception:
                    prob = 0.0
                if lang and prob >= float(self.language_lock_min_probability):
                    self.detected_language = str(lang).strip().lower() or None
                    if self.detected_language:
                        logger.info(
                            "Transcription: locked language to '%s' (p=%.2f)",
                            self.detected_language,
                            prob,
                        )
            return seg_list, info

        with self._lock:
            try:
                return _run_transcribe(audio)
            except RuntimeError as e:
                if self.device == "cuda" and _looks_like_missing_cuda_runtime(e):
                    logger.warning(f"CUDA runtime libraries are missing ({e}). Falling back to CPU transcription.")
                    self._load_model(device="cpu", compute_type="int8")
                    return _run_transcribe(audio)
                raise

    def transcribe_stream(self, audio_generator, *, return_voiceprint: bool = False):
        """
        Consumes an audio generator (yielding float32 numpy arrays) and performs transcription.
        This is a simplified approach. faster-whisper usually expects a complete buffer or file.
        For real-time, we need to buffer audio until we have enough to transcribe, or use a VAD 
        to chop it up.
        
        For this MVP, we will accumulate a fixed buffer (e.g., 3-5 seconds) and transcribe it.
        A more advanced version would use Silero VAD to detect end of speech.
        """
        buffer_parts: list[np.ndarray] = []
        buffer_samples = 0
        BUFFER_THRESHOLD = int(self.sample_rate * float(self.chunk_duration_s))
        OVERLAP_SAMPLES = int(self.sample_rate * float(self.chunk_overlap_s))
        last_emitted_text = ""

        chunks_received = 0
        for chunk in audio_generator:
            if chunk is None:
                continue
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if chunk.size == 0:
                continue
            chunks_received += 1
            if chunks_received == 1:
                logger.info(f"Transcription: first audio chunk received ({len(chunk)} samples)")
            buffer_parts.append(chunk)
            buffer_samples += int(chunk.size)
            
            if buffer_samples >= BUFFER_THRESHOLD:
                buffer = np.concatenate(buffer_parts).astype(np.float32, copy=False)
                logger.info(f"Transcription: buffer full ({len(buffer)} samples), running whisper...")
                audio = preprocess_audio(buffer, sample_rate=self.sample_rate, cfg=self.preprocess)
                rescued_from_empty_preprocess = False
                if len(audio) == 0:
                    rms, peak = self._audio_energy(buffer)
                    should_rescue = (
                        bool(self.preprocess is not None)
                        and bool(getattr(self.preprocess, "enabled", False))
                        and bool(getattr(self.preprocess, "vad_enabled", False))
                        and (rms >= float(self.vad_rescue_min_rms) or peak >= float(self.vad_rescue_min_peak))
                    )
                    if should_rescue:
                        logger.info(
                            "Transcription: preprocess removed all audio; retrying raw buffer with Whisper VAD "
                            f"(rms={rms:.4f}, peak={peak:.4f})"
                        )
                        audio = buffer
                        rescued_from_empty_preprocess = True
                    else:
                        logger.info("Transcription: no speech detected in buffer (preprocess removed all audio)")
                        # Keep overlap even when no speech is detected to avoid losing context.
                        if OVERLAP_SAMPLES > 0 and len(buffer) > OVERLAP_SAMPLES:
                            tail = buffer[-OVERLAP_SAMPLES:].copy()
                            buffer_parts = [tail]
                            buffer_samples = int(tail.size)
                        else:
                            buffer_parts = []
                            buffer_samples = 0
                        continue

                # Transcribe the current buffer (with CUDA->CPU fallback if needed).
                segments, info = self._transcribe_with_fallback(
                    audio,
                    force_whisper_vad=(True if rescued_from_empty_preprocess else None),
                )

                segment_texts = [str(getattr(segment, "text", "") or "") for segment in segments]
                normalized = self._merge_segment_texts(segment_texts)
                if normalized:
                    if normalized and last_emitted_text:
                        normalized = self._trim_leading_overlap(last_emitted_text, normalized)
                        normalized = self._normalize_transcript_text(normalized)
                    if not normalized:
                        # Nothing new after overlap trimming; continue accumulating stream.
                        if OVERLAP_SAMPLES > 0 and len(buffer) > OVERLAP_SAMPLES:
                            tail = buffer[-OVERLAP_SAMPLES:].copy()
                            buffer_parts = [tail]
                            buffer_samples = int(tail.size)
                        else:
                            buffer_parts = []
                            buffer_samples = 0
                        continue

                    logger.info(f"Transcription: got text: {normalized[:80]}...")
                    last_emitted_text = normalized
                    if return_voiceprint:
                        vp = None
                        try:
                            vp = compute_voiceprint(audio, sample_rate=self.sample_rate)
                        except Exception:
                            vp = None
                        yield TranscriptionChunk(text=normalized, voiceprint=vp)
                    else:
                        yield normalized
                else:
                    logger.info("Transcription: no speech detected in buffer (VAD filtered or silence)")
                
                # Keep overlap instead of clearing to reduce chopped words.
                if OVERLAP_SAMPLES > 0 and len(buffer) > OVERLAP_SAMPLES:
                    tail = buffer[-OVERLAP_SAMPLES:].copy()
                    buffer_parts = [tail]
                    buffer_samples = int(tail.size)
                else:
                    buffer_parts = []
                    buffer_samples = 0
