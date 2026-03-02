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
CONTEXT_TOKEN_BUDGET = 6000  # tokens for the conversation context window
SUMMARY_TOKEN_BUDGET = 1500  # tokens reserved for the rolling summary
RECENT_TOKEN_BUDGET = CONTEXT_TOKEN_BUDGET - SUMMARY_TOKEN_BUDGET  # ~4500 for verbatim entries
PRIOR_OUTPUTS_TOKEN_BUDGET = 1500  # tokens for prior AI outputs (dedup context)
SESSION_BRIEFING_TOKEN_BUDGET = 500  # tokens for user-provided session briefing
TRANSCRIPT_SEGMENT_TOKEN_BUDGET = 2000  # tokens for the new transcript segment

# How many new entries to accumulate before triggering a summary refresh
SUMMARY_REFRESH_INTERVAL = 15


def _estimate_tokens(text: str) -> int:
    """Rough token estimate from character count."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _format_entry(speaker: str, text: str) -> str:
    return f"[{speaker}]: {text}"


def _truncate(text: str, max_chars: int = 200) -> str:
    """Truncate text to max_chars, adding ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"


def cap_text(text: str, token_budget: int) -> str:
    """Truncate text to fit within a token budget."""
    max_chars = token_budget * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"


def build_prior_outputs(session: Session, token_budget: int = PRIOR_OUTPUTS_TOKEN_BUDGET) -> str:
    """
    Build a compact summary of what the AI has already output this session.

    Token-budgeted: stops adding items once the budget is reached.
    Individual items are truncated to keep entries compact — the LLM only
    needs enough to recognise "I already covered this topic".
    """
    header = "YOUR PRIOR OUTPUTS (do NOT repeat these):\n\n"
    tokens_used = _estimate_tokens(header)
    sections: list[str] = []

    # --- Suggestions (most recent, reversed for recency priority) ---
    if session.suggestions:
        section_lines = ["Previous suggestions you already made:"]
        section_tokens = _estimate_tokens(section_lines[0])
        for s in reversed(session.suggestions):
            line = f"- {_truncate(s.text, 150)}"
            if s.responding_to:
                line += f" (re: {_truncate(s.responding_to, 60)})"
            line_tokens = _estimate_tokens(line)
            if tokens_used + section_tokens + line_tokens > token_budget:
                break
            section_lines.append(line)
            section_tokens += line_tokens
        if len(section_lines) > 1:
            tokens_used += section_tokens
            sections.append("\n".join(section_lines))

    # --- Notes (most recent) ---
    if session.notes:
        section_lines = ["Notes already extracted:"]
        section_tokens = _estimate_tokens(section_lines[0])
        for n in reversed(session.notes):
            line = f"- [{n.type.value}] {_truncate(n.content, 150)}"
            line_tokens = _estimate_tokens(line)
            if tokens_used + section_tokens + line_tokens > token_budget:
                break
            section_lines.append(line)
            section_tokens += line_tokens
        if len(section_lines) > 1:
            tokens_used += section_tokens
            sections.append("\n".join(section_lines))

    # --- Fact-checks (from recent transcript entries) ---
    checks_lines = ["Claims already fact-checked:"]
    checks_tokens = _estimate_tokens(checks_lines[0])
    for entry in reversed(session.transcript):
        if tokens_used + checks_tokens >= token_budget:
            break
        for fc in entry.fact_checks:
            line = f"- \"{_truncate(fc.claim, 100)}\" → {fc.verdict}"
            line_tokens = _estimate_tokens(line)
            if tokens_used + checks_tokens + line_tokens > token_budget:
                break
            checks_lines.append(line)
            checks_tokens += line_tokens
    if len(checks_lines) > 1:
        tokens_used += checks_tokens
        sections.append("\n".join(checks_lines))

    if not sections:
        return ""

    return header + "\n\n".join(sections)


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
