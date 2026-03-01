"""Agentic loop — streams LLM completions, executes tool calls, iterates."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

from litellm import Router

from asure_flow.agent.tools import execute_tool, get_all_schemas

if TYPE_CHECKING:
    from asure_flow.sessions.models import Session

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5


async def run_agent(
    router: Router,
    transcript_text: str,
    conversation_context: str = "",
    session_context: str = "",
    fact_checking: bool = True,
    suggestions: bool = True,
    notes: bool = True,
    search_transcript: bool = True,
    search_sessions: bool = True,
    web_search: bool = True,
    format_code: bool = True,
    deep_think: bool = False,
    system_prompt: str | None = None,
    session: Session | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the agentic loop on a transcript segment.

    Yields events:
      - {"type": "content_delta", "text": str}
      - {"type": "tool_call", "name": str, "arguments": dict}
      - {"type": "tool_result", "name": str, "result": dict}
      - {"type": "done"}
      - {"type": "error", "message": str}
    """
    tools = get_all_schemas(
        fact_checking=fact_checking,
        suggestions=suggestions,
        notes=notes,
        format_code=format_code,
        search_transcript=search_transcript,
        search_sessions=search_sessions,
        web_search=web_search,
        deep_think=deep_think,
    )

    if not tools:
        # No features or tools enabled — nothing to do
        yield {"type": "done"}
        return

    if not system_prompt:
        # Fallback: should not happen as ws/session.py always resolves it
        system_prompt = "You are Asuré Flow, an AI conversation assistant. Be concise and actionable."

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]

    # Build the user message with optional session context
    user_parts: list[str] = []

    if session_context:
        user_parts.append(f"Session briefing:\n{session_context}")

    if conversation_context:
        user_parts.append(f"Conversation so far:\n\n{conversation_context}")

    user_parts.append(f"New transcript segment to process:\n\n{transcript_text}")

    messages.append({
        "role": "user",
        "content": "\n\n---\n\n".join(user_parts),
    })

    for iteration in range(MAX_ITERATIONS):
        try:
            full_content = ""
            tool_calls_acc: dict[int, dict] = {}
            finish_reason = None

            response = await router.acompletion(
                model="assistant",
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                stream=True,
                temperature=0.7,
            )

            async for chunk in response:
                choice = chunk.choices[0]
                delta = choice.delta
                finish_reason = choice.finish_reason

                if delta.content:
                    full_content += delta.content
                    yield {"type": "content_delta", "text": delta.content}

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc_delta.id:
                            tool_calls_acc[idx]["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_calls_acc[idx]["name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments

            # No tool calls — agent is done
            if finish_reason != "tool_calls" or not tool_calls_acc:
                yield {"type": "done"}
                return

            # Build assistant message with tool calls
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": full_content or None,
                "tool_calls": [],
            }
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                assistant_msg["tool_calls"].append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })
            messages.append(assistant_msg)

            # Execute each tool
            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    logger.warning("Malformed tool arguments for %s: %s", func_name, tc["function"]["arguments"])
                    error_msg = json.dumps({"error": f"Invalid JSON in arguments for {func_name}. Please retry with valid JSON."})
                    yield {"type": "tool_call", "name": func_name, "arguments": {}}
                    yield {"type": "tool_result", "name": func_name, "result": {"error": "Invalid JSON arguments"}}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": error_msg,
                    })
                    continue

                yield {"type": "tool_call", "name": func_name, "arguments": func_args}

                result_str = await execute_tool(func_name, func_args, session=session)
                result_data = json.loads(result_str)

                yield {"type": "tool_result", "name": func_name, "result": result_data}

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_str,
                })

        except Exception as e:
            logger.exception("Agent loop error at iteration %d", iteration)
            yield {"type": "error", "message": str(e)}
            return

    yield {"type": "done"}
