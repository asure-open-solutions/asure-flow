"""Topic extraction from session transcripts."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asure_flow.sessions.models import Session

logger = logging.getLogger(__name__)


async def extract_topics(router, session: Session) -> list[str]:
    """Use the LLM to extract 3-8 topic tags from a session transcript."""
    if not session.transcript:
        return []

    text = session.get_context(last_n=50)

    try:
        response = await router.acompletion(
            model="assistant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract concise topic tags from conversations. "
                        "Return ONLY a JSON array of 3-8 short topic strings. "
                        "Example: [\"project planning\", \"budget review\", \"Q3 targets\"]"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Extract topics from this conversation:\n\n{text}",
                },
            ],
            temperature=0.3,
            max_tokens=200,
        )

        content = response.choices[0].message.content
        if not content:
            return []
        raw = content.strip()
        # Handle cases where the LLM wraps in markdown code block
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        topics = json.loads(raw)
        if isinstance(topics, list):
            return [str(t).strip() for t in topics[:8] if t]
    except Exception:
        logger.warning("Failed to extract topics", exc_info=True)
    return []
