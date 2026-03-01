"""Tests for AI feature schema filtering and passthrough execution."""

import json
import pytest

from asure_flow.agent.features import (
    get_features,
    is_passthrough,
    execute_feature,
    ALL_FEATURES,
)


class TestGetFeatures:
    def test_all_enabled(self):
        features = get_features(
            fact_checking=True,
            suggestions=True,
            notes=True,
            format_code=True,
            deep_think=True,
        )
        names = [f["function"]["name"] for f in features]
        assert "fact_check" in names
        assert "suggest_response" in names
        assert "extract_notes" in names
        assert "format_code" in names
        assert "deep_think" in names

    def test_all_disabled(self):
        features = get_features(
            fact_checking=False,
            suggestions=False,
            notes=False,
            format_code=False,
            deep_think=False,
        )
        assert features == []

    def test_selective_toggles(self):
        features = get_features(
            fact_checking=True,
            suggestions=False,
            notes=True,
            format_code=False,
            deep_think=False,
        )
        names = [f["function"]["name"] for f in features]
        assert "fact_check" in names
        assert "extract_notes" in names
        assert "suggest_response" not in names
        assert "format_code" not in names
        assert "deep_think" not in names

    def test_deep_think_default_off(self):
        features = get_features()
        names = [f["function"]["name"] for f in features]
        assert "deep_think" not in names

    def test_returns_valid_schemas(self):
        features = get_features()
        for f in features:
            assert f["type"] == "function"
            assert "name" in f["function"]
            assert "parameters" in f["function"]
            assert "description" in f["function"]


class TestIsPassthrough:
    def test_known_passthrough(self):
        assert is_passthrough("fact_check")
        assert is_passthrough("suggest_response")
        assert is_passthrough("extract_notes")
        assert is_passthrough("format_code")
        assert is_passthrough("deep_think")

    def test_not_passthrough(self):
        assert not is_passthrough("search_transcript")
        assert not is_passthrough("web_search")
        assert not is_passthrough("unknown_tool")


class TestExecuteFeature:
    @pytest.mark.anyio
    async def test_echoes_arguments(self):
        args = {"claims": [{"claim": "test", "verdict": "supported", "reasoning": "ok"}]}
        result = await execute_feature("fact_check", args)
        parsed = json.loads(result)
        assert parsed == args

    @pytest.mark.anyio
    async def test_extract_notes_schema(self):
        """Verify extract_notes accepts structured action items."""
        args = {
            "action_items": [
                {"content": "Send report", "owner": "Alice", "due_date": "2026-03-15"},
                {"content": "Review PR"},
            ],
            "decisions": ["Use React"],
            "key_facts": ["Budget is 50k"],
            "risks": ["Tight deadline"],
        }
        result = await execute_feature("extract_notes", args)
        parsed = json.loads(result)
        assert parsed["action_items"][0]["owner"] == "Alice"
        assert parsed["action_items"][1].get("owner") is None


class TestExtractNotesSchema:
    def test_action_items_are_objects(self):
        """Verify the extract_notes schema uses structured action items."""
        from asure_flow.agent.features import FEATURE_EXTRACT_NOTES

        items_schema = FEATURE_EXTRACT_NOTES["function"]["parameters"]["properties"]["action_items"]["items"]
        assert items_schema["type"] == "object"
        assert "content" in items_schema["properties"]
        assert "owner" in items_schema["properties"]
        assert "due_date" in items_schema["properties"]
        assert items_schema["required"] == ["content"]
