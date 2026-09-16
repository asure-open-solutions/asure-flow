"""Session data models."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex[:12]


class SessionStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    ENDED = "ended"


class FactCheck(BaseModel):
    claim: str
    verdict: str  # supported, contradicted, uncertain
    reasoning: str
    fallacy: Optional[str] = None  # logical fallacy if identified


class TranscriptEntry(BaseModel):
    id: str = Field(default_factory=_new_id)
    timestamp: datetime = Field(default_factory=_utcnow)
    speaker: str
    text: str
    fact_checks: list[FactCheck] = Field(default_factory=list)
    audio_start: Optional[float] = None  # Audio offset (seconds) for diarization
    audio_end: Optional[float] = None


class NoteType(str, Enum):
    ACTION_ITEM = "action_item"
    DECISION = "decision"
    KEY_FACT = "key_fact"
    RISK = "risk"


class NoteEntry(BaseModel):
    id: str = Field(default_factory=_new_id)
    type: NoteType
    content: str
    timestamp: datetime = Field(default_factory=_utcnow)
    owner: Optional[str] = None  # Participant display_name (for action items)
    due_date: Optional[str] = None  # ISO date string (for action items)
    completed: bool = False


class SuggestionEntry(BaseModel):
    id: str = Field(default_factory=_new_id)
    text: str
    responding_to: str = ""
    timestamp: datetime = Field(default_factory=_utcnow)


class Participant(BaseModel):
    """A named participant in a session."""

    speaker_label: str  # Diarization label, e.g. "Speaker 1"
    display_name: str  # User-assigned name, e.g. "Alice"
    role: Optional[str] = None  # e.g. "Project Manager"
    notes: Optional[str] = None  # Free-form notes about this person


class PersonEntity(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    role: Optional[str] = None
    mentioned_in: list[str] = Field(default_factory=list)


class ProjectEntity(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str
    description: Optional[str] = None
    mentioned_in: list[str] = Field(default_factory=list)


class DecisionEntity(BaseModel):
    id: str = Field(default_factory=_new_id)
    summary: str
    date: Optional[str] = None
    participants: list[str] = Field(default_factory=list)
    mentioned_in: list[str] = Field(default_factory=list)


class SessionEntities(BaseModel):
    """Container for all extracted entities in a session."""

    people: list[PersonEntity] = Field(default_factory=list)
    projects: list[ProjectEntity] = Field(default_factory=list)
    decisions: list[DecisionEntity] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Accumulated token usage for a session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class SessionSettings(BaseModel):
    """Per-session overrides. Only non-null fields override globals."""

    fact_checking: Optional[bool] = None
    suggestions: Optional[bool] = None
    notes: Optional[bool] = None
    search_transcript: Optional[bool] = None
    search_sessions: Optional[bool] = None
    web_search: Optional[bool] = None
    format_code: Optional[bool] = None
    deep_think: Optional[str] = None  # "off" | "auto" | "always"
    agent_mode: Optional[str] = None  # "unified" | "specialists"
    parallel_tools: Optional[bool] = None
    diarization: Optional[bool] = None
    pii_redaction: Optional[bool] = None
    privacy_mode: Optional[bool] = None


class Session(BaseModel):
    id: str = Field(default_factory=_new_id)
    name: str = "Untitled Session"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    status: SessionStatus = SessionStatus.ACTIVE
    context: str = ""  # User-provided session briefing for AI
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    notes: list[NoteEntry] = Field(default_factory=list)
    suggestions: list[SuggestionEntry] = Field(default_factory=list)
    participants: list[Participant] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    entities: SessionEntities = Field(default_factory=SessionEntities)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    settings: Optional[SessionSettings] = None

    def add_transcript(
        self,
        speaker: str,
        text: str,
        audio_start: float | None = None,
        audio_end: float | None = None,
        entry_id: str | None = None,
    ) -> TranscriptEntry:
        entry = TranscriptEntry(
            **({"id": entry_id} if entry_id else {}),
            speaker=speaker, text=text, audio_start=audio_start, audio_end=audio_end,
        )
        self.transcript.append(entry)
        self.updated_at = _utcnow()
        return entry

    def add_fact_checks(self, transcript_id: str, checks: list[FactCheck]) -> None:
        for entry in self.transcript:
            if entry.id == transcript_id:
                entry.fact_checks.extend(checks)
                self.updated_at = _utcnow()
                return

    def add_notes(self, notes: list[NoteEntry]) -> None:
        self.notes.extend(notes)
        self.updated_at = _utcnow()

    def add_suggestion(self, text: str, responding_to: str = "") -> SuggestionEntry:
        entry = SuggestionEntry(text=text, responding_to=responding_to)
        self.suggestions.append(entry)
        self.updated_at = _utcnow()
        return entry

    def get_context(self, last_n: int = 20) -> str:
        """Get recent transcript as a plain-text string for AI context."""
        recent = self.transcript[-last_n:]
        lines = [f"[{self.get_display_name(e.speaker)}]: {e.text}" for e in recent]
        return "\n".join(lines)

    def rename_speaker(
        self, speaker_label: str, display_name: str, role: str | None = None,
    ) -> Participant:
        """Rename a speaker label and update all matching transcript entries."""
        # Find or create participant entry
        participant: Participant | None = None
        for p in self.participants:
            if p.speaker_label == speaker_label:
                participant = p
                break
        if participant:
            participant.display_name = display_name
            if role is not None:
                participant.role = role
        else:
            participant = Participant(
                speaker_label=speaker_label, display_name=display_name, role=role,
            )
            self.participants.append(participant)
        # Update all transcript entries with the old label
        for entry in self.transcript:
            if entry.speaker == speaker_label:
                entry.speaker = display_name
        self.updated_at = _utcnow()
        return participant

    def delete_transcript_entry(self, entry_id: str) -> bool:
        """Remove a transcript entry by ID. Returns True if found and removed."""
        for i, entry in enumerate(self.transcript):
            if entry.id == entry_id:
                self.transcript.pop(i)
                self.updated_at = _utcnow()
                return True
        return False

    def edit_transcript_entry(self, entry_id: str, new_text: str) -> TranscriptEntry | None:
        """Update the text of a transcript entry. Returns the updated entry or None."""
        for entry in self.transcript:
            if entry.id == entry_id:
                entry.text = new_text
                self.updated_at = _utcnow()
                return entry
        return None

    def get_display_name(self, speaker_label: str) -> str:
        """Resolve a speaker label to its display name, if one exists."""
        for p in self.participants:
            if p.speaker_label == speaker_label:
                return p.display_name
        return speaker_label


class SessionSummary(BaseModel):
    """Lightweight session info for list endpoints."""

    id: str
    name: str
    created_at: datetime
    updated_at: datetime
    status: SessionStatus
    transcript_count: int
    notes_count: int
    topics: list[str] = Field(default_factory=list)
