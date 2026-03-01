"""Webhook integration — posts session events to a configured URL."""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from asure_flow.integrations.base import Integration

if TYPE_CHECKING:
    from asure_flow.sessions.models import Session, NoteEntry

logger = logging.getLogger(__name__)


class WebhookIntegration(Integration):
    id = "webhook"
    name = "Webhook"
    description = "Send session events to a webhook URL"

    def __init__(self) -> None:
        self.url: str | None = None
        self.secret: str | None = None

    def configure(self, config: dict[str, Any]) -> None:
        self.url = config.get("url")
        self.secret = config.get("secret")

    async def on_session_end(self, session: Session) -> dict[str, Any]:
        if not self.url:
            return {"status": "skipped", "reason": "no URL configured"}
        try:
            import httpx

            headers: dict[str, str] = {}
            if self.secret:
                headers["X-Webhook-Secret"] = self.secret
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self.url,
                    json={
                        "event": "session_end",
                        "session": session.model_dump(mode="json"),
                    },
                    headers=headers,
                )
                return {"status": "sent", "response_code": resp.status_code}
        except ImportError:
            logger.warning("httpx not installed — webhook unavailable")
            return {"status": "error", "reason": "httpx not installed"}
        except Exception as e:
            logger.warning("Webhook failed: %s", e)
            return {"status": "error", "reason": str(e)}

    async def on_note_added(self, session: Session, note: NoteEntry) -> dict[str, Any]:
        if not self.url:
            return {"status": "skipped"}
        try:
            import httpx

            headers: dict[str, str] = {}
            if self.secret:
                headers["X-Webhook-Secret"] = self.secret
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(
                    self.url,
                    json={
                        "event": "note_added",
                        "session_id": session.id,
                        "note": note.model_dump(mode="json"),
                    },
                    headers=headers,
                )
                return {"status": "sent", "response_code": resp.status_code}
        except Exception as e:
            logger.warning("Webhook failed: %s", e)
            return {"status": "error", "reason": str(e)}
