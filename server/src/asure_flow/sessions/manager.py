"""Session manager — CRUD operations with local JSON storage."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

from asure_flow.config import settings
from pydantic import ValidationError

from asure_flow.sessions.models import (
    Session,
    SessionStatus,
    SessionSummary,
)

logger = logging.getLogger(__name__)

_VALID_SESSION_ID = re.compile(r"^[a-f0-9]{12}$")


class SessionManager:
    def __init__(self) -> None:
        self._base_dir = Path(settings.session_dir).expanduser()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, Session] = {}
        self._deleted: set[str] = set()
        # Serialises _save across the autosave executor thread and event-loop saves.
        self._save_lock = threading.Lock()

    @staticmethod
    def _is_valid_id(session_id: str) -> bool:
        return bool(_VALID_SESSION_ID.match(session_id))

    def _session_path(self, session_id: str) -> Path:
        if not self._is_valid_id(session_id):
            raise ValueError(f"Invalid session ID: {session_id!r}")
        path = (self._base_dir / f"{session_id}.json").resolve()
        if not path.is_relative_to(self._base_dir.resolve()):
            raise ValueError(f"Session path escapes base directory: {session_id!r}")
        return path

    def create(self, name: str = "Untitled Session") -> Session:
        session = Session(name=name)
        self._active[session.id] = session
        self._save(session)
        logger.info("Created session: %s (%s)", session.id, name)
        return session

    def get(self, session_id: str) -> Optional[Session]:
        if not self._is_valid_id(session_id):
            return None
        if session_id in self._deleted:
            return None
        # Check in-memory cache first
        if session_id in self._active:
            return self._active[session_id]
        # Try loading from disk
        return self._load(session_id)

    def save(self, session: Session) -> None:
        if session.id in self._deleted:
            return
        self._active[session.id] = session
        self._save(session)

    async def save_async(self, session: Session) -> None:
        """Non-blocking save — offloads serialisation + I/O to a thread."""
        if session.id in self._deleted:
            return
        self._active[session.id] = session
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._save, session)

    def list_sessions(self) -> list[SessionSummary]:
        summaries: list[SessionSummary] = []
        for path in sorted(self._base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                summaries.append(SessionSummary(
                    id=data["id"],
                    name=data["name"],
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                    status=data["status"],
                    transcript_count=len(data.get("transcript", [])),
                    notes_count=len(data.get("notes", [])),
                    topics=data.get("topics", []),
                ))
            except (json.JSONDecodeError, KeyError, OSError, ValidationError):
                logger.warning("Failed to read session file: %s", path, exc_info=True)
        return summaries

    def delete(self, session_id: str) -> bool:
        if not self._is_valid_id(session_id):
            return False
        self._active.pop(session_id, None)
        self._deleted.add(session_id)
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            logger.info("Deleted session: %s", session_id)
            return True
        return False

    def end_session(self, session_id: str) -> Optional[Session]:
        session = self.get(session_id)
        if session:
            session.status = SessionStatus.ENDED
            self.save(session)
        return session

    def _save(self, session: Session) -> None:
        path = self._session_path(session.id)
        with self._save_lock:
            # Re-check inside the lock: a delete may have landed between the
            # caller's guard and acquiring the lock — don't resurrect the file.
            if session.id in self._deleted:
                return
            data = session.model_dump_json(indent=2)
            # Atomic write: serialise to a temp file in the same dir, then replace.
            # A crash mid-write leaves the previous file intact, not a truncated one.
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)

    def _load(self, session_id: str) -> Optional[Session]:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            session = Session.model_validate_json(path.read_text(encoding="utf-8"))
            self._active[session.id] = session
            return session
        except Exception:
            logger.exception("Failed to load session: %s", session_id)
            return None


# Singleton
session_manager = SessionManager()
