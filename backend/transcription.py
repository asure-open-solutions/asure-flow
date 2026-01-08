from faster_whisper import WhisperModel
import numpy as np
import threading
import logging
from dataclasses import dataclass
import os
import ctypes

logger = logging.getLogger(__name__)

from backend.speech_processing import SpeechPreprocessConfig, preprocess_audio
from backend.diarization import compute_voiceprint


def _looks_like_missing_cuda_runtime(exc: BaseException) -> bool:
    msg = str(exc or "").casefold()
    if not msg:
        return False

    # Common Windows CUDA runtime failures when ctranslate2 is built for CUDA but the
    # CUDA runtime isn't installed / not on PATH.
    if ".dll" in msg and ("not found" in msg or "cannot be loaded" in msg):
        if any(tok in msg for tok in ("cublas", "cudart", "cufft", "curand", "cusolver", "cusparse")):
            return True
        if msg.startswith("library ") and " is not found" in msg:
            return True

    return False


def _windows_cuda_dlls_available() -> bool:
    if os.name != "nt":
        return True

    # Most common dependency for CUDA-enabled ctranslate2/faster-whisper wheels on Windows.
    candidates = ("cublas64_12.dll", "cublas64_11.dll")
    for dll in candidates:
        try:
            ctypes.WinDLL(dll)
            return True
        except OSError:
            continue

    return False


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
                    raise RuntimeError("Missing CUDA runtime DLLs (e.g. cublas64_12.dll)")

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

    def _transcribe_with_fallback(self, audio: np.ndarray):
        with self._lock:
            try:
                segments, info = self.model.transcribe(audio, beam_size=5, vad_filter=self.whisper_vad_filter)
                return list(segments), info
            except RuntimeError as e:
                if self.device == "cuda" and _looks_like_missing_cuda_runtime(e):
                    logger.warning(
                        "CUDA runtime libraries are missing (e.g. cublas64_12.dll). Falling back to CPU transcription."
                    )
                    self._load_model(device="cpu", compute_type="int8")
                    segments, info = self.model.transcribe(audio, beam_size=5, vad_filter=self.whisper_vad_filter)
                    return list(segments), info
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
        buffer = np.array([], dtype=np.float32)
        # Short buffers cause mid-sentence fragmentation.
        # Keep latency reasonable while improving continuity.
        CHUNK_DURATION_S = 4.0
        BUFFER_THRESHOLD = int(self.sample_rate * CHUNK_DURATION_S)

        # Keep a small overlap to reduce chopped words at chunk boundaries.
        OVERLAP_S = 0.35
        OVERLAP_SAMPLES = int(self.sample_rate * OVERLAP_S)

        chunks_received = 0
        for chunk in audio_generator:
            chunks_received += 1
            if chunks_received == 1:
                logger.info(f"Transcription: first audio chunk received ({len(chunk)} samples)")
            buffer = np.concatenate((buffer, chunk))
            
            if len(buffer) >= BUFFER_THRESHOLD:
                logger.info(f"Transcription: buffer full ({len(buffer)} samples), running whisper...")
                audio = preprocess_audio(buffer, sample_rate=self.sample_rate, cfg=self.preprocess)
                if len(audio) == 0:
                    logger.info("Transcription: no speech detected in buffer (preprocess removed all audio)")
                    # Keep overlap even when no speech is detected to avoid losing context.
                    if OVERLAP_SAMPLES > 0 and len(buffer) > OVERLAP_SAMPLES:
                        buffer = buffer[-OVERLAP_SAMPLES:]
                    else:
                        buffer = np.array([], dtype=np.float32)
                    continue

                # Transcribe the current buffer (with CUDA->CPU fallback if needed).
                segments, info = self._transcribe_with_fallback(audio)
                
                text_accumulated = ""
                for segment in segments:
                    text_accumulated += segment.text + " "
                
                if text_accumulated.strip():
                    logger.info(f"Transcription: got text: {text_accumulated.strip()[:80]}...")
                    if return_voiceprint:
                        vp = None
                        try:
                            vp = compute_voiceprint(audio, sample_rate=self.sample_rate)
                        except Exception:
                            vp = None
                        yield TranscriptionChunk(text=text_accumulated.strip(), voiceprint=vp)
                    else:
                        yield text_accumulated.strip()
                else:
                    logger.info("Transcription: no speech detected in buffer (VAD filtered or silence)")
                
                # Keep overlap instead of clearing to reduce chopped words.
                if OVERLAP_SAMPLES > 0 and len(buffer) > OVERLAP_SAMPLES:
                    buffer = buffer[-OVERLAP_SAMPLES:]
                else:
                    buffer = np.array([], dtype=np.float32)
