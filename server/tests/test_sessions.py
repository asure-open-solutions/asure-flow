"""Tests for session management."""

import pytest
from asure_flow.sessions.models import Session, NoteEntry, NoteType, FactCheck


def test_create_session():
    session = Session(name="Test Session")
    assert session.name == "Test Session"
    assert session.status == "active"
    assert len(session.transcript) == 0
    assert len(session.notes) == 0


def test_add_transcript():
    session = Session()
    entry = session.add_transcript("User", "Hello world")
    assert entry.speaker == "User"
    assert entry.text == "Hello world"
    assert len(session.transcript) == 1


def test_add_fact_checks():
    session = Session()
    entry = session.add_transcript("User", "The earth is round")
    checks = [FactCheck(claim="The earth is round", verdict="supported", reasoning="Scientific consensus")]
    session.add_fact_checks(entry.id, checks)
    assert len(session.transcript[0].fact_checks) == 1
    assert session.transcript[0].fact_checks[0].verdict == "supported"


def test_add_notes():
    session = Session()
    notes = [
        NoteEntry(type=NoteType.ACTION_ITEM, content="Follow up on X"),
        NoteEntry(type=NoteType.DECISION, content="Use FastAPI"),
    ]
    session.add_notes(notes)
    assert len(session.notes) == 2


def test_get_context():
    session = Session()
    session.add_transcript("User", "Hello")
    session.add_transcript("Third Party", "Hi there")
    context = session.get_context()
    assert "[User]: Hello" in context
    assert "[Third Party]: Hi there" in context
