"""Tests for REST API endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from asure_flow.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.mark.anyio
async def test_health(client: AsyncClient):
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


@pytest.mark.anyio
async def test_create_and_list_sessions(client: AsyncClient):
    # Create
    resp = await client.post("/api/sessions", json={"name": "Test"})
    assert resp.status_code == 200
    session = resp.json()
    assert session["name"] == "Test"
    session_id = session["id"]

    # List
    resp = await client.get("/api/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert any(s["id"] == session_id for s in sessions)

    # Get
    resp = await client.get(f"/api/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == session_id

    # Delete
    resp = await client.delete(f"/api/sessions/{session_id}")
    assert resp.status_code == 200


@pytest.mark.anyio
async def test_get_missing_session(client: AsyncClient):
    resp = await client.get("/api/sessions/nonexistent")
    assert resp.status_code == 404
