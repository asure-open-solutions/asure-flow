"""Regression tests for the interview-critical reliability fixes."""

import asyncio
import threading
import time

import numpy as np
import pytest
from unittest.mock import MagicMock

from asure_flow.api.routes import UpdateProfileRequest
from asure_flow.agent.context import build_context, SUMMARY_TOKEN_BUDGET
from asure_flow.agent.loop import run_agent
from asure_flow.agent.router import build_router
from asure_flow.config import ProviderEntry, settings
from asure_flow.sessions.models import Session
from asure_flow.transcription.engine import AudioBuffer, WhisperEngine
from asure_flow.ws.audio import _resolve_capture_mode
from asure_flow.ws.session import (
    TriggerSignals,
    _compute_trigger_delay,
    _is_user_speaker,
    _persist_tool_result,
)


def test_profile_execution_settings_are_not_dropped():
    request = UpdateProfileRequest(
        agent_mode="specialists",
        parallel_tools=True,
        ai_response_profile="realtime",
    )
    changes = request.model_dump(exclude_none=True)
    assert changes["agent_mode"] == "specialists"
    assert changes["parallel_tools"] is True
    assert changes["ai_response_profile"] == "realtime"


def test_remote_client_forces_client_capture(monkeypatch):
    monkeypatch.setattr(settings, "audio_capture_source", "server")
    monkeypatch.setattr(settings, "system_device_id", "4")
    assert _resolve_capture_mode("auto") == (True, True)
    assert _resolve_capture_mode("client") == (False, False)


def test_response_profiles_bound_interactive_throttle():
    signals = TriggerSignals(
        word_count=6,
        is_trivial=False,
        has_question=True,
        speaker_changed=True,
        other_to_user=False,
        user_just_spoke=False,
        seconds_since_last_fire=0.1,
    )
    assert _compute_trigger_delay(signals, "realtime") <= 0.25
    assert _compute_trigger_delay(signals, "balanced") <= 1.0
    assert _compute_trigger_delay(signals, "quality") > 2.0


def test_rolling_summary_is_capped():
    session = Session()
    session.add_transcript("User", "Recent context")
    context = build_context(session, "x" * 20_000)
    assert len(context) < SUMMARY_TOKEN_BUDGET * 4 + 200


def test_router_exposes_quality_and_realtime_models(monkeypatch):
    monkeypatch.setattr(settings, "providers", [
        ProviderEntry(
            id="test",
            name="Test",
            litellm_prefix="openai",
            model="quality-model",
            realtime_model="fast-model",
            api_key="test-key",
        ),
    ])
    router = build_router()
    assert router is not None
    routes = {
        item["model_name"]: item["litellm_params"]["model"]
        for item in router.model_list
    }
    assert routes["assistant"] == "openai/quality-model"
    assert routes["assistant_realtime"] == "openai/fast-model"


def test_hosted_provider_without_key_is_unavailable(monkeypatch):
    monkeypatch.setattr(settings, "providers", [
        ProviderEntry(
            id="openrouter",
            name="OpenRouter",
            litellm_prefix="openrouter",
            model="some-model",
            api_base="https://openrouter.ai/api/v1",
        ),
    ])
    assert build_router() is None


@pytest.mark.asyncio
async def test_plain_model_answer_becomes_suggestion():
    class PlainTextRouter:
        async def acompletion(self, **_kwargs):
            async def chunks():
                choice = MagicMock()
                choice.delta.content = "Use a concise STAR example."
                choice.delta.tool_calls = None
                choice.finish_reason = "stop"
                chunk = MagicMock()
                chunk.choices = [choice]
                chunk.usage = None
                yield chunk
            return chunks()

    events = [
        event async for event in run_agent(
            router=PlainTextRouter(),
            transcript_text="[Interviewer]: Tell me about a challenge.",
            fact_checking=False,
            suggestions=True,
            notes=False,
            search_transcript=False,
            search_sessions=False,
            web_search=False,
            format_code=False,
            max_iterations=1,
            fallback_suggestion=True,
        )
    ]
    suggestion = next(e for e in events if e["type"] == "tool_result")
    assert suggestion["name"] == "suggest_response"
    assert "STAR" in suggestion["result"]["suggestion"]


def test_vad_rejects_single_noise_spike(monkeypatch):
    import asure_flow.transcription.engine as engine_module

    class FakeVad:
        def __call__(self, audio):
            windows = max(8, len(audio) // 512)
            probs = np.full(windows, 0.05, dtype=np.float32)
            probs[2] = 0.95  # a click/bump, not sustained speech
            return probs

    monkeypatch.setattr(engine_module, "_vad_model", FakeVad())
    buffer = AudioBuffer(WhisperEngine(), speaker_label="User")
    buffer.add_audio(np.zeros(16_000, dtype=np.int16).tobytes())
    assert buffer._check_vad_state() is False
    assert buffer._has_speech is False


def test_fact_check_result_keeps_triggering_transcript_id():
    session = Session()
    target = session.add_transcript("Third Party", "The claim to verify")
    later = session.add_transcript("User", "A later line")
    event = {
        "name": "fact_check",
        "result": {
            "claims": [{
                "claim": "The claim to verify",
                "verdict": "uncertain",
                "reasoning": "More evidence is required",
            }],
        },
    }

    _persist_tool_result(session, target.id, event)

    assert event["transcript_id"] == target.id
    assert len(session.transcript[0].fact_checks) == 1
    assert session.transcript[1].id == later.id
    assert session.transcript[1].fact_checks == []


def test_rerun_recognizes_renamed_user():
    session = Session()
    session.rename_speaker("User", "Henri")
    assert _is_user_speaker(session, "Henri") is True
    assert _is_user_speaker(session, "Third Party") is False


@pytest.mark.asyncio
async def test_whisper_inference_is_serialized(monkeypatch):
    engine = WhisperEngine()
    engine._model = object()
    guard = threading.Lock()
    active = 0
    max_active = 0

    def fake_transcribe(_audio, _prompt):
        nonlocal active, max_active
        with guard:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.03)
        with guard:
            active -= 1
        return []

    monkeypatch.setattr(engine, "_transcribe_sync", fake_transcribe)
    audio = np.zeros(320, dtype=np.float32)
    await asyncio.gather(engine.transcribe(audio), engine.transcribe(audio))

    assert max_active == 1
