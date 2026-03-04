"""Parallel specialist agents — run focused micro-agents concurrently."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncGenerator

from asure_flow.agent.features import (
    FEATURE_DEEP_THINK,
    FEATURE_EXTRACT_NOTES,
    FEATURE_FACT_CHECK,
    FEATURE_FORMAT_CODE,
    FEATURE_SUGGEST_RESPONSE,
)
from asure_flow.agent.loop import run_agent
from asure_flow.agent.presets import build_specialist_prompt
from asure_flow.agent.tools import TOOL_SEARCH_SESSIONS, TOOL_SEARCH_TRANSCRIPT, TOOL_WEB_SEARCH

if TYPE_CHECKING:
    from litellm import Router
    from asure_flow.sessions.models import Session

logger = logging.getLogger(__name__)


@dataclass
class Specialist:
    """Definition of a specialist micro-agent."""

    name: str
    tools: list[dict] = field(default_factory=list)
    max_iterations: int = 2


def get_enabled_specialists(
    *,
    fact_checking: bool = True,
    suggestions: bool = True,
    notes: bool = True,
    search_transcript: bool = True,
    search_sessions: bool = True,
    web_search: bool = True,
    format_code: bool = True,
    deep_think_mode: str = "off",
) -> list[Specialist]:
    """Return specialist agents based on active feature toggles."""
    specialists: list[Specialist] = []
    deep_think_tool = [FEATURE_DEEP_THINK] if deep_think_mode != "off" else []

    if fact_checking:
        tools = [FEATURE_FACT_CHECK] + deep_think_tool
        if web_search:
            tools.append(TOOL_WEB_SEARCH)
        specialists.append(Specialist(name="fact_checker", tools=tools))

    if notes:
        specialists.append(Specialist(
            name="note_taker",
            tools=[FEATURE_EXTRACT_NOTES] + deep_think_tool,
        ))

    if suggestions:
        specialists.append(Specialist(
            name="suggester",
            tools=[FEATURE_SUGGEST_RESPONSE] + deep_think_tool,
        ))

    if search_transcript or search_sessions or (web_search and not fact_checking):
        tools = deep_think_tool[:]
        if search_transcript:
            tools.append(TOOL_SEARCH_TRANSCRIPT)
        if search_sessions:
            tools.append(TOOL_SEARCH_SESSIONS)
        if web_search:
            tools.append(TOOL_WEB_SEARCH)
        if tools:
            specialists.append(Specialist(name="researcher", tools=tools, max_iterations=3))

    if format_code:
        specialists.append(Specialist(
            name="code_analyst",
            tools=[FEATURE_FORMAT_CODE] + deep_think_tool,
        ))

    return specialists


async def run_specialists(
    router: Router,
    specialists: list[Specialist],
    transcript_text: str,
    conversation_context: str = "",
    session_context: str = "",
    prior_outputs: str = "",
    preset_id: str = "general",
    deep_think_mode: str = "off",
    session: Session | None = None,
    parallel_tools: bool = False,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run multiple specialist agents in parallel and merge their event streams.

    Each specialist gets a focused system prompt and limited tool set.
    Events are tagged with a `specialist` field for client-side attribution.
    """
    if not specialists:
        yield {"type": "done", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        return

    # Collect events from each specialist into queues
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    active_count = len(specialists)

    async def _run_one(spec: Specialist) -> None:
        """Run a single specialist and push events to the shared queue."""
        system_prompt = build_specialist_prompt(spec.name, preset_id, deep_think_mode)
        try:
            async for event in run_agent(
                router=router,
                transcript_text=transcript_text,
                conversation_context=conversation_context,
                session_context=session_context,
                prior_outputs=prior_outputs,
                system_prompt=system_prompt,
                session=session,
                max_iterations=spec.max_iterations,
                parallel_tools=parallel_tools,
                tools_override=spec.tools,
            ):
                tagged = {**event, "specialist": spec.name}
                await queue.put(tagged)
        except asyncio.CancelledError:
            await queue.put({"type": "done", "reason": "preempted", "specialist": spec.name,
                             "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
        except Exception as e:
            logger.exception("Specialist %s failed", spec.name)
            await queue.put({"type": "error", "message": str(e), "specialist": spec.name})
        finally:
            await queue.put(None)  # sentinel

    # Start all specialists concurrently
    tasks = [asyncio.create_task(_run_one(spec)) for spec in specialists]

    # Merge event streams
    total_prompt_tokens = 0
    total_completion_tokens = 0
    finished = 0

    try:
        while finished < active_count:
            event = await queue.get()
            if event is None:
                finished += 1
                continue

            # Accumulate token usage from done events
            if event.get("type") == "done":
                usage = event.get("usage", {})
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                # Don't yield individual done events — we yield one combined done at the end
                continue

            yield event

    except asyncio.CancelledError:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        yield {"type": "done", "reason": "preempted", "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        }}
        return

    yield {"type": "done", "usage": {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }}
