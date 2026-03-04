"""AI tool definitions — active tools that execute server-side logic."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from asure_flow.agent.features import get_features, is_passthrough, execute_feature

if TYPE_CHECKING:
    from asure_flow.sessions.models import Session

logger = logging.getLogger(__name__)

# ── Tool schemas (OpenAI function-calling format) ──
# These tools perform actual server-side computation and return real results
# to the LLM for further reasoning.

TOOL_SEARCH_TRANSCRIPT = {
    "type": "function",
    "function": {
        "name": "search_transcript",
        "description": (
            "Search through the current session's transcript for specific "
            "information, quotes, or topics mentioned in the conversation. "
            "Supports semantic search when embeddings are available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords, phrases, or topics to find in the transcript",
                },
                "speaker": {
                    "type": "string",
                    "description": "Filter results to a specific speaker (e.g., 'User', 'Speaker 1')",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

TOOL_SEARCH_SESSIONS = {
    "type": "function",
    "function": {
        "name": "search_sessions",
        "description": (
            "Search across all past session transcripts for information, "
            "previously discussed topics, or recurring themes. "
            "Supports semantic search when embeddings are available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to find across all sessions",
                },
                "speaker": {
                    "type": "string",
                    "description": "Filter results to a specific speaker",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5)",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

TOOL_WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the web for current information to verify facts, find "
            "statistics, or research topics discussed in the conversation. "
            "Always cite sources with their URLs when presenting web results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The web search query",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def get_tools(
    search_transcript: bool = True,
    search_sessions: bool = True,
    web_search: bool = True,
) -> list[dict]:
    """Return the active tool list filtered by toggles."""
    tools: list[dict] = []
    if search_transcript:
        tools.append(TOOL_SEARCH_TRANSCRIPT)
    if search_sessions:
        tools.append(TOOL_SEARCH_SESSIONS)
    if web_search:
        tools.append(TOOL_WEB_SEARCH)
    return tools


def get_all_schemas(
    fact_checking: bool = True,
    suggestions: bool = True,
    notes: bool = True,
    format_code: bool = True,
    search_transcript: bool = True,
    search_sessions: bool = True,
    web_search: bool = True,
    deep_think: bool = False,
) -> list[dict]:
    """Return combined feature + tool schemas for the agentic loop."""
    return get_features(
        fact_checking=fact_checking,
        suggestions=suggestions,
        notes=notes,
        format_code=format_code,
        deep_think=deep_think,
    ) + get_tools(
        search_transcript=search_transcript,
        search_sessions=search_sessions,
        web_search=web_search,
    )


# ── Tool execution ──


def _ts(entry) -> str:
    """Format a transcript entry's timestamp as an ISO string."""
    return entry.timestamp.isoformat() if hasattr(entry.timestamp, "isoformat") else str(entry.timestamp)


async def _ensure_index(session_id: str, transcript: list) -> None:
    """Lazily backfill embeddings for entries not yet indexed."""
    from asure_flow.search.embeddings import embedding_engine
    from asure_flow.search.index import get_index

    if not embedding_engine.available:
        return

    idx = get_index(session_id)
    missing = [e for e in transcript if not idx.has_entry(e.id)]
    if not missing:
        return

    texts = [e.text for e in missing]
    embeddings = await embedding_engine.embed(texts)
    for entry, emb in zip(missing, embeddings):
        idx.add(entry.id, emb)
    idx.save()


async def _execute_search_transcript(arguments: dict, session: Session | None) -> str:
    """Search the current session's transcript — semantic if available, else substring."""
    query = arguments.get("query", "").strip()
    speaker_filter = arguments.get("speaker")
    if not session or not query:
        return json.dumps({"results": [], "message": "No transcript available"})

    from asure_flow.search.embeddings import embedding_engine
    from asure_flow.search.index import get_index

    results = []
    search_type = "substring"

    if embedding_engine.available and session.transcript:
        await _ensure_index(session.id, session.transcript)
        idx = get_index(session.id)

        if idx.count > 0:
            search_type = "semantic"
            query_emb = await embedding_engine.embed_single(query)
            matches = idx.search(query_emb, top_k=20)
            entry_map = {e.id: e for e in session.transcript}

            for entry_id, score in matches:
                entry = entry_map.get(entry_id)
                if not entry:
                    continue
                if speaker_filter and entry.speaker.lower() != speaker_filter.lower():
                    continue
                results.append({
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "timestamp": _ts(entry),
                    "relevance": round(score, 3),
                })

    # Fallback to substring
    if not results and search_type == "substring":
        q = query.lower()
        for entry in session.transcript:
            if speaker_filter and entry.speaker.lower() != speaker_filter.lower():
                continue
            if q in entry.text.lower():
                results.append({
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "timestamp": _ts(entry),
                })

    return json.dumps({
        "results": results[:20],
        "total_matches": len(results),
        "search_type": search_type,
    })


async def _execute_search_sessions(arguments: dict) -> str:
    """Search across all past sessions — semantic if available, else substring."""
    from asure_flow.sessions.manager import session_manager
    from asure_flow.search.embeddings import embedding_engine
    from asure_flow.search.index import get_index

    query = arguments.get("query", "").strip()
    speaker_filter = arguments.get("speaker")
    max_results = arguments.get("max_results", 5)
    if not query:
        return json.dumps({"results": [], "message": "No query provided"})

    results = []
    search_type = "substring"

    sessions = session_manager.list_sessions()[:50]

    if embedding_engine.available:
        search_type = "semantic"
        query_emb = await embedding_engine.embed_single(query)

        # Collect scored results across sessions
        scored: list[tuple[float, dict]] = []
        for summary in sessions:
            session_obj = session_manager.get(summary.id)
            if not session_obj or not session_obj.transcript:
                continue

            await _ensure_index(session_obj.id, session_obj.transcript)
            idx = get_index(session_obj.id)
            if idx.count == 0:
                continue

            matches = idx.search(query_emb, top_k=max_results)
            entry_map = {e.id: e for e in session_obj.transcript}

            for entry_id, score in matches:
                entry = entry_map.get(entry_id)
                if not entry:
                    continue
                if speaker_filter and entry.speaker.lower() != speaker_filter.lower():
                    continue
                scored.append((score, {
                    "session_name": session_obj.name,
                    "session_id": session_obj.id,
                    "speaker": entry.speaker,
                    "text": entry.text,
                    "timestamp": _ts(entry),
                    "relevance": round(score, 3),
                }))

        # Sort by relevance and take top results
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [r for _, r in scored[:max_results]]
    else:
        # Substring fallback
        q = query.lower()
        for summary in sessions:
            session_obj = session_manager.get(summary.id)
            if not session_obj:
                continue
            for entry in session_obj.transcript:
                if speaker_filter and entry.speaker.lower() != speaker_filter.lower():
                    continue
                if q in entry.text.lower():
                    results.append({
                        "session_name": session_obj.name,
                        "session_id": session_obj.id,
                        "speaker": entry.speaker,
                        "text": entry.text,
                        "timestamp": _ts(entry),
                    })
                    if len(results) >= max_results:
                        break
            if len(results) >= max_results:
                break

    # Entity-enhanced matching: also search entity names across sessions
    query_lower = query.lower()
    entity_matches: list[dict] = []
    for summary in sessions:
        session_obj = session_manager.get(summary.id)
        if not session_obj or not session_obj.entities:
            continue
        for person in session_obj.entities.people:
            if query_lower in person.name.lower():
                entity_matches.append({
                    "session_name": session_obj.name,
                    "session_id": session_obj.id,
                    "match_type": "entity_person",
                    "text": f"Person: {person.name}" + (f" ({person.role})" if person.role else ""),
                    "speaker": "",
                    "timestamp": "",
                })
        for project in session_obj.entities.projects:
            if query_lower in project.name.lower():
                entity_matches.append({
                    "session_name": session_obj.name,
                    "session_id": session_obj.id,
                    "match_type": "entity_project",
                    "text": f"Project: {project.name}" + (f" — {project.description}" if project.description else ""),
                    "speaker": "",
                    "timestamp": "",
                })
        for decision in session_obj.entities.decisions:
            if query_lower in decision.summary.lower():
                entity_matches.append({
                    "session_name": session_obj.name,
                    "session_id": session_obj.id,
                    "match_type": "entity_decision",
                    "text": f"Decision: {decision.summary}",
                    "speaker": "",
                    "timestamp": "",
                })

    # Merge entity matches (deduplicate by session_id + text)
    seen = {(r.get("session_id", ""), r.get("text", "")) for r in results}
    for em in entity_matches:
        key = (em["session_id"], em["text"])
        if key not in seen:
            results.append(em)
            seen.add(key)

    return json.dumps({
        "results": results[:max_results],
        "total_matches": len(results),
        "search_type": search_type,
    })


# ── Web credibility scoring ──

_HIGH_CREDIBILITY_DOMAINS = frozenset({
    "wikipedia.org", "nature.com", "science.org",
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
    "pubmed.ncbi.nlm.nih.gov", "arxiv.org", "scholar.google.com",
    "who.int", "cdc.gov",
})

_LOW_CREDIBILITY_DOMAINS = frozenset({
    "reddit.com", "quora.com", "yahoo.com",
})


def _score_credibility(url: str) -> dict[str, object]:
    """Return credibility tier and score based on domain heuristics."""
    from urllib.parse import urlparse

    try:
        domain = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return {"tier": "medium", "score": 0.5}

    tld = domain.rsplit(".", 1)[-1] if "." in domain else ""

    if tld in ("gov", "edu") or any(domain == d or domain.endswith("." + d) for d in _HIGH_CREDIBILITY_DOMAINS):
        return {"tier": "high", "score": 0.9}
    if any(domain == d or domain.endswith("." + d) for d in _LOW_CREDIBILITY_DOMAINS):
        return {"tier": "low", "score": 0.3}
    return {"tier": "medium", "score": 0.6}


async def _execute_web_search(arguments: dict) -> str:
    """Search the web using DuckDuckGo with credibility scoring."""
    query = arguments.get("query", "")
    if not query:
        return json.dumps({"results": [], "message": "No query provided"})

    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(ddgs.text(query, max_results=5))

        results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
                "credibility": _score_credibility(r.get("href", "")),
            }
            for r in raw_results
        ]
        return json.dumps({"results": results})
    except ImportError:
        logger.warning("duckduckgo-search not installed — web search unavailable")
        return json.dumps({"results": [], "message": "Web search unavailable (duckduckgo-search not installed)"})
    except Exception as e:
        logger.warning("Web search failed: %s", e)
        return json.dumps({"results": [], "message": f"Web search failed: {e}"})


async def execute_tool(name: str, arguments: dict, session: Session | None = None) -> str:
    """
    Execute any tool call (feature or active tool) and return a JSON string.

    Dispatches to the appropriate handler based on the tool name.
    """
    if is_passthrough(name):
        return execute_feature(name, arguments)
    elif name == "search_transcript":
        return await _execute_search_transcript(arguments, session)
    elif name == "search_sessions":
        return await _execute_search_sessions(arguments)
    elif name == "web_search":
        return await _execute_web_search(arguments)
    else:
        # Unknown tool — passthrough as fallback
        logger.warning("Unknown tool called: %s", name)
        return json.dumps(arguments)
