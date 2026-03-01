"""Tests for AI tool execution (search, web, schemas)."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from asure_flow.agent.tools import (
    get_tools,
    get_all_schemas,
    execute_tool,
    _score_credibility,
)


class TestGetTools:
    def test_all_enabled(self):
        tools = get_tools(search_transcript=True, search_sessions=True, web_search=True)
        names = [t["function"]["name"] for t in tools]
        assert "search_transcript" in names
        assert "search_sessions" in names
        assert "web_search" in names

    def test_all_disabled(self):
        tools = get_tools(search_transcript=False, search_sessions=False, web_search=False)
        assert tools == []

    def test_selective(self):
        tools = get_tools(search_transcript=True, search_sessions=False, web_search=True)
        names = [t["function"]["name"] for t in tools]
        assert "search_transcript" in names
        assert "web_search" in names
        assert "search_sessions" not in names


class TestGetAllSchemas:
    def test_combines_features_and_tools(self):
        schemas = get_all_schemas(
            fact_checking=True,
            suggestions=True,
            notes=True,
            format_code=False,
            search_transcript=True,
            search_sessions=False,
            web_search=True,
        )
        names = [s["function"]["name"] for s in schemas]
        # Features
        assert "fact_check" in names
        assert "suggest_response" in names
        assert "extract_notes" in names
        # Tools
        assert "search_transcript" in names
        assert "web_search" in names
        # Disabled
        assert "format_code" not in names
        assert "search_sessions" not in names


class TestSearchTranscript:
    @pytest.fixture
    def mock_session(self):
        """Create a mock session with transcript entries."""
        entry1 = MagicMock()
        entry1.id = "e1"
        entry1.speaker = "User"
        entry1.text = "Let's discuss the quarterly budget"
        entry1.timestamp = datetime(2026, 3, 1, 10, 0, 0)

        entry2 = MagicMock()
        entry2.id = "e2"
        entry2.speaker = "Speaker 1"
        entry2.text = "The budget is set at fifty thousand dollars"
        entry2.timestamp = datetime(2026, 3, 1, 10, 1, 0)

        entry3 = MagicMock()
        entry3.id = "e3"
        entry3.speaker = "User"
        entry3.text = "What about the timeline for the project?"
        entry3.timestamp = datetime(2026, 3, 1, 10, 2, 0)

        session = MagicMock()
        session.id = "test-session"
        session.transcript = [entry1, entry2, entry3]
        return session

    @pytest.mark.anyio
    async def test_substring_search(self, mock_session):
        result = await execute_tool(
            "search_transcript",
            {"query": "budget"},
            session=mock_session,
        )
        data = json.loads(result)
        assert data["total_matches"] >= 1
        assert any("budget" in r["text"].lower() for r in data["results"])

    @pytest.mark.anyio
    async def test_speaker_filter(self, mock_session):
        result = await execute_tool(
            "search_transcript",
            {"query": "budget", "speaker": "Speaker 1"},
            session=mock_session,
        )
        data = json.loads(result)
        for r in data["results"]:
            assert r["speaker"] == "Speaker 1"

    @pytest.mark.anyio
    async def test_no_session(self):
        result = await execute_tool(
            "search_transcript",
            {"query": "anything"},
            session=None,
        )
        data = json.loads(result)
        assert data["results"] == []

    @pytest.mark.anyio
    async def test_empty_query(self, mock_session):
        result = await execute_tool(
            "search_transcript",
            {"query": ""},
            session=mock_session,
        )
        data = json.loads(result)
        assert data["results"] == []

    @pytest.mark.anyio
    async def test_no_match(self, mock_session):
        result = await execute_tool(
            "search_transcript",
            {"query": "xyznonexistent"},
            session=mock_session,
        )
        data = json.loads(result)
        assert data["total_matches"] == 0


class TestExecuteToolDispatch:
    @pytest.mark.anyio
    async def test_passthrough_feature(self):
        """Passthrough features should echo their arguments."""
        result = await execute_tool(
            "fact_check",
            {"claims": [{"claim": "x", "verdict": "supported", "reasoning": "y"}]},
        )
        data = json.loads(result)
        assert data["claims"][0]["claim"] == "x"

    @pytest.mark.anyio
    async def test_unknown_tool(self):
        """Unknown tools should return the arguments as fallback."""
        result = await execute_tool(
            "unknown_future_tool",
            {"foo": "bar"},
        )
        data = json.loads(result)
        assert data["foo"] == "bar"

    @pytest.mark.anyio
    async def test_web_search_no_deps(self):
        """Web search should gracefully handle missing duckduckgo-search."""
        result = await execute_tool("web_search", {"query": "test"})
        data = json.loads(result)
        assert "results" in data


class TestToolSchemas:
    def test_all_have_required_fields(self):
        tools = get_tools()
        for tool in tools:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params

    def test_web_search_mentions_citations(self):
        """Web search description should instruct LLM to cite sources."""
        tools = get_tools()
        ws_tool = next(t for t in tools if t["function"]["name"] == "web_search")
        desc = ws_tool["function"]["description"]
        assert "cite" in desc.lower() or "source" in desc.lower()
