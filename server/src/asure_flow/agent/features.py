"""AI feature schemas — passthrough tools where the LLM generates structured output."""

from __future__ import annotations

import json

# ── Feature schemas (OpenAI function-calling format) ──
# These are "passthrough" — the LLM fills in the structured output via function
# calling, and execute_feature() simply echoes the arguments back as JSON.

FEATURE_FACT_CHECK = {
    "type": "function",
    "function": {
        "name": "fact_check",
        "description": (
            "Analyse factual claims made during the conversation. "
            "For each claim, determine whether it is supported, contradicted, or uncertain "
            "based on your knowledge. Provide brief reasoning. "
            "If a logical fallacy is present, identify it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claims": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "claim": {
                                "type": "string",
                                "description": "The factual claim extracted from the conversation",
                            },
                            "verdict": {
                                "type": "string",
                                "enum": ["supported", "contradicted", "uncertain"],
                                "description": "Whether the claim is supported, contradicted, or uncertain",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why this verdict was given",
                            },
                            "fallacy": {
                                "type": "string",
                                "description": "If applicable, the logical fallacy identified (e.g., ad hominem, straw man, appeal to authority)",
                            },
                        },
                        "required": ["claim", "verdict", "reasoning"],
                        "additionalProperties": False,
                    },
                    "description": "List of fact-checked claims",
                },
            },
            "required": ["claims"],
            "additionalProperties": False,
        },
    },
}

FEATURE_SUGGEST_RESPONSE = {
    "type": "function",
    "function": {
        "name": "suggest_response",
        "description": (
            "Generate a practical, fact-grounded reply suggestion for the user "
            "based on the current conversation context. The suggestion should be "
            "something the user can say or adapt in their ongoing conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "suggestion": {
                    "type": "string",
                    "description": "The suggested response text the user could use",
                },
                "tone": {
                    "type": "string",
                    "enum": ["professional", "casual", "empathetic", "assertive", "diplomatic"],
                    "description": "The tone of the suggested response",
                },
                "key_points": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key points addressed in the suggestion",
                },
            },
            "required": ["suggestion", "tone", "key_points"],
            "additionalProperties": False,
        },
    },
}

FEATURE_EXTRACT_NOTES = {
    "type": "function",
    "function": {
        "name": "extract_notes",
        "description": (
            "Extract and organise key information from the conversation into "
            "structured rolling notes. Identify action items, decisions, key facts, "
            "and risks."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action_items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The action item",
                            },
                            "owner": {
                                "type": "string",
                                "description": "Person responsible (from session participants if known)",
                            },
                            "due_date": {
                                "type": "string",
                                "description": "Due date if mentioned (ISO format, e.g. 2026-03-15)",
                            },
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                    "description": "Tasks or follow-ups that need to be done, with optional owner and due date",
                },
                "decisions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Decisions that were made during the conversation",
                },
                "key_facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Important facts or information mentioned",
                },
                "risks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Potential risks or concerns raised",
                },
            },
            "required": ["action_items", "decisions", "key_facts", "risks"],
            "additionalProperties": False,
        },
    },
}

FEATURE_FORMAT_CODE = {
    "type": "function",
    "function": {
        "name": "format_code",
        "description": (
            "Detect, format, and analyse code mentioned in the conversation. "
            "Identify the programming language, provide clean formatting, and "
            "analyse the code for correctness, complexity, and potential issues. "
            "Useful during coding interviews or technical discussions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code snippet to format and analyse",
                },
                "language": {
                    "type": "string",
                    "description": "The programming language (auto-detect if not specified)",
                },
                "analysis": {
                    "type": "string",
                    "description": "Analysis of the code: correctness, complexity, potential issues, improvements",
                },
            },
            "required": ["code", "analysis"],
            "additionalProperties": False,
        },
    },
}

FEATURE_DEEP_THINK = {
    "type": "function",
    "function": {
        "name": "deep_think",
        "description": (
            "Think step-by-step about a complex or nuanced topic before responding. "
            "Use this when the situation requires careful analysis, when claims are "
            "ambiguous, or when the best response isn't immediately obvious."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Step-by-step reasoning about the topic",
                },
                "conclusion": {
                    "type": "string",
                    "description": "The conclusion or decision reached after reasoning",
                },
            },
            "required": ["reasoning", "conclusion"],
            "additionalProperties": False,
        },
    },
}

# All passthrough feature schemas
ALL_FEATURES = [
    FEATURE_FACT_CHECK,
    FEATURE_SUGGEST_RESPONSE,
    FEATURE_EXTRACT_NOTES,
    FEATURE_FORMAT_CODE,
    FEATURE_DEEP_THINK,
]

# Names of passthrough features (for dispatch in execute)
_PASSTHROUGH_NAMES = frozenset(f["function"]["name"] for f in ALL_FEATURES)


def get_features(
    fact_checking: bool = True,
    suggestions: bool = True,
    notes: bool = True,
    format_code: bool = True,
    deep_think: bool = False,
) -> list[dict]:
    """Return the feature (passthrough tool) list filtered by toggles."""
    features: list[dict] = []
    if fact_checking:
        features.append(FEATURE_FACT_CHECK)
    if suggestions:
        features.append(FEATURE_SUGGEST_RESPONSE)
    if notes:
        features.append(FEATURE_EXTRACT_NOTES)
    if format_code:
        features.append(FEATURE_FORMAT_CODE)
    if deep_think:
        features.append(FEATURE_DEEP_THINK)
    return features


def is_passthrough(name: str) -> bool:
    """Check if a tool name is a passthrough feature."""
    return name in _PASSTHROUGH_NAMES


async def execute_feature(name: str, arguments: dict) -> str:
    """Execute a passthrough feature — just echo the LLM's structured output."""
    return json.dumps(arguments)
