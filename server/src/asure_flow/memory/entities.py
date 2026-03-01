"""Structured entity extraction from session transcripts."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asure_flow.sessions.models import Session

from asure_flow.sessions.models import (
    PersonEntity,
    ProjectEntity,
    DecisionEntity,
    SessionEntities,
)

logger = logging.getLogger(__name__)

_ENTITY_TOOL = {
    "type": "function",
    "function": {
        "name": "record_entities",
        "description": "Record structured entities extracted from the conversation",
        "parameters": {
            "type": "object",
            "properties": {
                "people": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                },
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                },
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "participants": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["summary"],
                    },
                },
            },
            "required": ["people", "projects", "decisions"],
        },
    },
}


async def extract_entities(router, session: Session) -> SessionEntities:
    """Use the LLM to extract structured entities from a session."""
    text = session.get_context(last_n=100)
    if not text:
        return SessionEntities()

    try:
        response = await router.acompletion(
            model="assistant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract all people, projects, and decisions mentioned in "
                        "the conversation. Use the record_entities tool."
                    ),
                },
                {"role": "user", "content": text},
            ],
            tools=[_ENTITY_TOOL],
            tool_choice={"type": "function", "function": {"name": "record_entities"}},
            temperature=0.3,
        )

        tc = response.choices[0].message.tool_calls
        if tc:
            data = json.loads(tc[0].function.arguments)
            return SessionEntities(
                people=[PersonEntity(**p) for p in data.get("people", [])],
                projects=[ProjectEntity(**p) for p in data.get("projects", [])],
                decisions=[DecisionEntity(**d) for d in data.get("decisions", [])],
            )
    except Exception:
        logger.warning("Failed to extract entities", exc_info=True)
    return SessionEntities()
