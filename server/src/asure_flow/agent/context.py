"""Context window management — token-based windowing with rolling summary.

Provides intelligent context for the agentic loop by:
1. Keeping recent transcript entries verbatim (up to a token budget)
2. Periodically summarizing older entries into a condensed summary
3. Combining both for maximum context within token limits
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm import Router
    from asure_flow.sessions.models import Session

logger = logging.getLogger(__name__)

# Approximate tokens per character (conservative estimate for English text)
_CHARS_PER_TOKEN = 4

# Default token budgets
CONTEXT_TOKEN_BUDGET = 6000  # tokens for the context window
SUMMARY_TOKEN_BUDGET = 1500  # tokens reserved for the rolling summary
RECENT_TOKEN_BUDGET = CONTEXT_TOKEN_BUDGET - SUMMARY_TOKEN_BUDGET  # ~4500 for verbatim entries

# How many new entries to accumulate before triggering a summary refresh
SUMMARY_REFRESH_INTERVAL = 15


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _format_entry(speaker: str, text: str) -> str:
    return f"[{speaker}]: {text}"


def build_context(session: Session, rolling_summary: str | None = None) -> str:
    """
    Build the context string for the AI agent.

    Strategy:
    - Work backwards from the most recent entry
    - Add entries verbatim until we hit the recent token budget
    - Prepend the rolling summary (if available) for older context
    """
    if not session.transcript:
        return ""

    # Collect recent entries within token budget (newest first)
    recent_lines: list[str] = []
    tokens_used = 0

    for entry in reversed(session.transcript):
        line = _format_entry(entry.speaker, entry.text)
        line_tokens = _estimate_tokens(line)
        if tokens_used + line_tokens > RECENT_TOKEN_BUDGET:
            break
        recent_lines.append(line)
        tokens_used += line_tokens

    # Reverse back to chronological order
    recent_lines.reverse()

    parts: list[str] = []

    if rolling_summary:
        parts.append(f"[Summary of earlier conversation]\n{rolling_summary}\n")

    if recent_lines:
        parts.append("\n".join(recent_lines))

    return "\n\n---\n\n".join(parts) if len(parts) > 1 else (parts[0] if parts else "")


def needs_summary_refresh(
    session: Session,
    last_summarized_index: int,
) -> bool:
    """Check if we should generate a new rolling summary."""
    total = len(session.transcript)
    unsummarized = total - last_summarized_index
    return unsummarized >= SUMMARY_REFRESH_INTERVAL


async def generate_summary(
    router: Router,
    session: Session,
    existing_summary: str | None,
    start_index: int,
    end_index: int,
) -> str:
    """
    Generate or update the rolling summary using the LLM.

    Summarises transcript entries from start_index to end_index,
    incorporating the existing summary if present.
    """
    entries = session.transcript[start_index:end_index]
    if not entries:
        return existing_summary or ""

    transcript_text = "\n".join(
        _format_entry(e.speaker, e.text) for e in entries
    )

    prompt_parts = []
    if existing_summary:
        prompt_parts.append(
            f"Here is the existing summary of the conversation so far:\n\n{existing_summary}\n\n"
            f"Here are new transcript segments to incorporate:\n\n{transcript_text}"
        )
    else:
        prompt_parts.append(
            f"Here is a transcript of a conversation:\n\n{transcript_text}"
        )

    prompt_parts.append(
        "\n\nProvide a concise summary (max 300 words) capturing:\n"
        "- Key topics discussed\n"
        "- Important decisions or conclusions\n"
        "- Notable claims or facts mentioned\n"
        "- Any unresolved questions or action items\n"
        "Write in present tense. Be factual and concise."
    )

    try:
        response = await router.acompletion(
            model="assistant",
            messages=[
                {"role": "system", "content": "You are a concise conversation summarizer."},
                {"role": "user", "content": "".join(prompt_parts)},
            ],
            temperature=0.3,
            max_tokens=500,
        )
        summary = response.choices[0].message.content or ""
        logger.info(
            "Generated rolling summary (entries %d-%d, %d chars)",
            start_index, end_index, len(summary),
        )
        return summary.strip()
    except Exception:
        logger.exception("Failed to generate rolling summary")
        return existing_summary or ""
