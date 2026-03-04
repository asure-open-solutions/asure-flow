"""Agentic loop — streams LLM completions, executes tool calls, iterates."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, AsyncGenerator

from litellm import Router

from asure_flow.agent.context import cap_text, SESSION_BRIEFING_TOKEN_BUDGET, TRANSCRIPT_SEGMENT_TOKEN_BUDGET
from asure_flow.agent.features import validate_tool_args
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
    prior_outputs: str = "",
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
    max_iterations: int = MAX_ITERATIONS,
    parallel_tools: bool = False,
    tools_override: list[dict] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run the agentic loop on a transcript segment.

    Yields events:
      - {"type": "content_delta", "text": str}
      - {"type": "tool_call", "name": str, "arguments": dict}
      - {"type": "tool_result", "name": str, "result": dict}
      - {"type": "done", "usage": {"prompt_tokens": int, "completion_tokens": int}}
      - {"type": "error", "message": str}
    """
    tools = tools_override if tools_override is not None else get_all_schemas(
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
        yield {"type": "done", "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
        return

    if not system_prompt:
        system_prompt = "You are Asuré Flow, an AI conversation assistant. Be concise and actionable."

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
    ]

    user_parts: list[str] = []

    if session_context:
        user_parts.append(f"Session briefing:\n{cap_text(session_context, SESSION_BRIEFING_TOKEN_BUDGET)}")

    if conversation_context:
        user_parts.append(f"Conversation so far:\n\n{conversation_context}")

    if prior_outputs:
        user_parts.append(prior_outputs)

    user_parts.append(
        f"New transcript segment to process:\n\n{cap_text(transcript_text, TRANSCRIPT_SEGMENT_TOKEN_BUDGET)}"
    )

    messages.append({
        "role": "user",
        "content": "\n\n---\n\n".join(user_parts),
    })

    # Accumulate token usage across iterations
    total_prompt_tokens = 0
    total_completion_tokens = 0

    for iteration in range(max_iterations):
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

                # Extract token usage from final chunk
                if hasattr(chunk, "usage") and chunk.usage:
                    total_prompt_tokens += getattr(chunk.usage, "prompt_tokens", 0) or 0
                    total_completion_tokens += getattr(chunk.usage, "completion_tokens", 0) or 0

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
                yield {"type": "done", "usage": {
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": total_completion_tokens,
                }}
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

            # Parse and validate all tool calls first
            parsed_calls: list[tuple[dict, str, dict | None, str | None]] = []
            for tc in assistant_msg["tool_calls"]:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    logger.warning("Malformed tool arguments for %s: %s", func_name, tc["function"]["arguments"])
                    parsed_calls.append((tc, func_name, None, f"Invalid JSON in arguments for {func_name}. Please retry with valid JSON."))
                    continue

                # Structured output validation
                validation_error = validate_tool_args(func_name, func_args)
                if validation_error:
                    parsed_calls.append((tc, func_name, func_args, validation_error))
                    continue

                parsed_calls.append((tc, func_name, func_args, None))

            # Execute tools — parallel or sequential
            if parallel_tools and len(parsed_calls) > 1:
                async for event in _execute_tools_parallel(parsed_calls, messages, session):
                    yield event
            else:
                for tc, func_name, func_args, error in parsed_calls:
                    if error or func_args is None:
                        error_msg = json.dumps({"error": error or "Invalid arguments"})
                        yield {"type": "tool_call", "name": func_name, "arguments": func_args or {}}
                        yield {"type": "tool_result", "name": func_name, "result": {"error": error or "Invalid arguments"}}
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": error_msg})
                        continue

                    yield {"type": "tool_call", "name": func_name, "arguments": func_args}

                    result_str = await execute_tool(func_name, func_args, session=session)
                    try:
                        result_data = json.loads(result_str)
                    except json.JSONDecodeError:
                        logger.warning("Tool %s returned invalid JSON: %s", func_name, result_str[:200])
                        result_data = {"error": "Invalid tool response"}
                        result_str = json.dumps(result_data)

                    yield {"type": "tool_result", "name": func_name, "result": result_data}
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})

        except asyncio.CancelledError:
            yield {"type": "done", "reason": "preempted", "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
            }}
            return

        except json.JSONDecodeError as e:
            logger.warning("Malformed LLM response at iteration %d: %s", iteration, e)
            yield {"type": "error", "message": "Malformed AI response, retrying..."}
            continue

        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
                yield {"type": "error", "message": "Rate limited — will retry shortly"}
                return
            elif "context_length" in error_str or "too many tokens" in error_str or "maximum context" in error_str:
                yield {"type": "error", "message": "Context too long for AI model"}
                return
            else:
                logger.exception("Agent loop error at iteration %d", iteration)
                yield {"type": "error", "message": str(e)}
                return

    yield {"type": "done", "usage": {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
    }}


async def _execute_tools_parallel(
    parsed_calls: list[tuple[dict, str, dict | None, str | None]],
    messages: list[dict],
    session: Any,
) -> AsyncGenerator[dict[str, Any], None]:
    """Execute multiple tool calls concurrently and yield events."""
    # Yield all tool_call events upfront
    for tc, func_name, func_args, error in parsed_calls:
        if error or func_args is None:
            yield {"type": "tool_call", "name": func_name, "arguments": func_args or {}}
        else:
            yield {"type": "tool_call", "name": func_name, "arguments": func_args}

    # Execute valid tools in parallel, errors immediately
    async def _safe_execute(func_name: str, func_args: dict) -> tuple[str, str]:
        try:
            result_str = await execute_tool(func_name, func_args, session=session)
            return func_name, result_str
        except Exception as e:
            logger.warning("Parallel tool %s failed: %s", func_name, e)
            return func_name, json.dumps({"error": str(e)})

    tasks = []
    task_indices = []
    for i, (tc, func_name, func_args, error) in enumerate(parsed_calls):
        if error or func_args is None:
            # Handle errors immediately
            error_msg = json.dumps({"error": error or "Invalid arguments"})
            yield {"type": "tool_result", "name": func_name, "result": {"error": error or "Invalid arguments"}}
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": error_msg})
        else:
            tasks.append(_safe_execute(func_name, func_args))
            task_indices.append(i)

    if tasks:
        results = await asyncio.gather(*tasks)
        for idx, (func_name, result_str) in zip(task_indices, results):
            tc = parsed_calls[idx][0]
            try:
                result_data = json.loads(result_str)
            except json.JSONDecodeError:
                logger.warning("Parallel tool %s returned invalid JSON", func_name)
                result_data = {"error": "Invalid tool response"}
                result_str = json.dumps(result_data)
            yield {"type": "tool_result", "name": func_name, "result": result_data}
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_str})
