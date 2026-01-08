from backend.transcription import TranscriptionEngine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Test")

try:
    logger.info("Testing Whisper Model Loading...")
    engine = TranscriptionEngine(model_size="tiny", device="cpu", compute_type="int8")
    logger.info("Model loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
