"""Asuré Flow server — FastAPI entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from asure_flow.agent.router import init_router
from asure_flow.api.routes import router as api_router
from asure_flow.audio.manager import audio_capture_manager
from asure_flow.config import settings
from asure_flow.profile import profile
from asure_flow.transcription.engine import whisper_engine
from asure_flow.ws.audio import router as audio_ws_router
from asure_flow.ws.session import router as session_ws_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Asuré Flow server …")

    # Load whisper model
    await whisper_engine.load()

    # Pre-warm VAD model (used for silence-aware flush gating)
    from faster_whisper.vad import get_vad_model
    get_vad_model()
    logger.info("VAD model loaded")

    # Initialise LLM router
    init_router()

    # Load embedding engine for semantic search
    try:
        from asure_flow.search.embeddings import embedding_engine
        await embedding_engine.load()
    except Exception:
        logger.info("Embedding engine unavailable — search will use substring matching")

    # Load diarization engine if enabled
    if profile.diarization_enabled:
        try:
            from asure_flow.transcription.diarization import diarization_engine
            await diarization_engine.load()
        except Exception:
            logger.info("Diarization engine unavailable")

    # Start server-side audio capture if configured.
    # Mic uses server capture only when audio_capture_source == "server".
    # System audio always uses server-side loopback when system_device_id is set.
    def _parse_device_id(value: str | None) -> int | None:
        if not value:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            logger.warning("Invalid device ID %r — ignoring", value)
            return None

    mic_device = (
        _parse_device_id(settings.mic_device_id)
        if settings.audio_capture_source == "server"
        else None
    )
    system_device = _parse_device_id(settings.system_device_id)
    if mic_device is not None or system_device is not None:
        try:
            await audio_capture_manager.start(
                mic_device_id=mic_device,
                system_device_id=system_device,
            )
        except Exception:
            logger.warning("Server-side audio capture failed to start", exc_info=True)

    logger.info("Server ready on %s:%d", settings.host, settings.port)
    yield

    audio_capture_manager.stop()
    logger.info("Shutting down …")


app = FastAPI(
    title="Asuré Flow",
    description="Real-time conversation assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow all origins for local-first usage; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(api_router)
app.include_router(audio_ws_router)
app.include_router(session_ws_router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "asure_flow.main:app",
        host=settings.host,
        port=settings.port,
        ws_max_size=1048576,
    )
