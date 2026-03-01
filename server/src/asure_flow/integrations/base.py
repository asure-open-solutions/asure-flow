"""Base class for external integrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from asure_flow.sessions.models import Session, NoteEntry


class Integration(ABC):
    """Base class for external service integrations."""

    id: str
    name: str
    description: str

    @abstractmethod
    async def on_session_end(self, session: Session) -> dict[str, Any]:
        """Called when a session ends. Return status dict."""
        ...

    @abstractmethod
    async def on_note_added(self, session: Session, note: NoteEntry) -> dict[str, Any]:
        """Called when a new note is extracted."""
        ...

    @abstractmethod
    def configure(self, config: dict[str, Any]) -> None:
        """Set integration-specific configuration."""
        ...
