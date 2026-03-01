"""Follow-up draft generation from session content."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asure_flow.sessions.models import Session

logger = logging.getLogger(__name__)

_FORMAT_INSTRUCTIONS = {
    "email": (
        "Generate a professional follow-up email summarising this meeting. "
        "Include a clear subject line, greeting, summary of key discussion points, "
        "action items with owners and due dates (if known), decisions made, "
        "and a closing."
    ),
    "message": (
        "Generate a concise follow-up message (Slack/Teams style) summarising "
        "this meeting. Use bullet points for action items and decisions. "
        "Keep it brief but comprehensive."
    ),
    "summary": (
        "Generate a structured meeting summary with sections for: "
        "Overview, Key Discussion Points, Decisions Made, Action Items "
        "(with owner and due date if known), and Next Steps."
    ),
}


async def generate_followup(
    router, session: Session, format: str = "email",
) -> dict[str, str]:
    """Generate a follow-up draft summarising the session."""
    instruction = _FORMAT_INSTRUCTIONS.get(format, _FORMAT_INSTRUCTIONS["email"])

    # Build context from session
    parts: list[str] = []
    parts.append(f"Meeting: {session.name}")

    if session.participants:
        names = [p.display_name for p in session.participants]
        parts.append(f"Participants: {', '.join(names)}")

    # Include recent transcript (up to 100 entries)
    transcript_text = session.get_context(last_n=100)
    if transcript_text:
        parts.append(f"\nTranscript:\n{transcript_text}")

    # Include extracted notes
    action_items = [n for n in session.notes if n.type.value == "action_item"]
    decisions = [n for n in session.notes if n.type.value == "decision"]

    if action_items:
        items_text = "\n".join(
            f"- {n.content}" + (f" (Owner: {n.owner})" if n.owner else "")
            + (f" (Due: {n.due_date})" if n.due_date else "")
            for n in action_items
        )
        parts.append(f"\nAction Items:\n{items_text}")

    if decisions:
        dec_text = "\n".join(f"- {n.content}" for n in decisions)
        parts.append(f"\nDecisions:\n{dec_text}")

    context = "\n".join(parts)

    # Use function calling for structured output
    tool = {
        "type": "function",
        "function": {
            "name": "draft_followup",
            "description": "Generate the follow-up draft",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {
                        "type": "string",
                        "description": "Subject line or title",
                    },
                    "body": {
                        "type": "string",
                        "description": "The full follow-up text",
                    },
                },
                "required": ["subject", "body"],
            },
        },
    }

    try:
        response = await router.acompletion(
            model="assistant",
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": context},
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "draft_followup"}},
            temperature=0.4,
            max_tokens=2000,
        )

        tc = response.choices[0].message.tool_calls
        if tc:
            data = json.loads(tc[0].function.arguments)
            return {
                "subject": data.get("subject", session.name),
                "body": data.get("body", ""),
                "format": format,
            }
    except Exception:
        logger.warning("Failed to generate follow-up", exc_info=True)

    return {"subject": session.name, "body": "Failed to generate follow-up.", "format": format}
