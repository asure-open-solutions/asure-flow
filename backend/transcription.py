from faster_whisper import WhisperModel
import numpy as np
import threading
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from backend.speech_processing import SpeechPreprocessConfig, preprocess_audio
from backend.diarization import compute_voiceprint

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
        device = (str(device or "cpu").strip().lower() or "cpu")
        if device in ("gpu", "cuda"):
            device = "cuda"
        if device not in ("cpu", "cuda"):
            device = "cpu"

        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        compute_type = str(compute_type or "").strip().lower() or ("float16" if device == "cuda" else "int8")

        logger.info(f"Loading Whisper model: {model_size} on {device} ({compute_type})...")
        try:
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            if device != "cpu":
                logger.warning(f"Failed to load Whisper on {device}; falling back to CPU: {e}")
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
            else:
                raise
        self._lock = threading.Lock()
        self.sample_rate = int(sample_rate)
        self.preprocess = preprocess or SpeechPreprocessConfig()
        self.whisper_vad_filter = bool(whisper_vad_filter)
        logger.info("Whisper model loaded.")

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
                # Transcribe the current buffer
                with self._lock:
                    segments, info = self.model.transcribe(audio, beam_size=5, vad_filter=self.whisper_vad_filter)
                
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
