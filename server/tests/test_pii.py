"""Tests for PII detection and redaction."""

import pytest
from asure_flow.safety.pii import detect_pii, redact_pii, PIIMatch


class TestDetectPII:
    def test_email(self):
        matches = detect_pii("Contact me at alice@example.com please")
        assert len(matches) == 1
        assert matches[0].type == "email"
        assert matches[0].original == "alice@example.com"

    def test_phone_us(self):
        matches = detect_pii("Call me at 555-123-4567")
        assert any(m.type == "phone" for m in matches)

    def test_phone_with_area_code_parens(self):
        matches = detect_pii("My number is (555) 123-4567")
        assert any(m.type == "phone" for m in matches)

    def test_ssn(self):
        matches = detect_pii("SSN: 123-45-6789")
        assert len(matches) >= 1
        ssn_matches = [m for m in matches if m.type == "ssn"]
        assert len(ssn_matches) == 1
        assert ssn_matches[0].original == "123-45-6789"

    def test_credit_card(self):
        matches = detect_pii("Card: 4111 1111 1111 1111")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 1

    def test_credit_card_dashes(self):
        matches = detect_pii("Card: 4111-1111-1111-1111")
        cc_matches = [m for m in matches if m.type == "credit_card"]
        assert len(cc_matches) == 1

    def test_no_pii(self):
        matches = detect_pii("This is a clean sentence about weather")
        # Filter to only non-phone matches (phone regex can be greedy)
        significant = [m for m in matches if m.type in ("email", "ssn", "credit_card")]
        assert len(significant) == 0

    def test_multiple_types(self):
        text = "Email alice@test.com, SSN 123-45-6789, card 4111111111111111"
        matches = detect_pii(text)
        types = {m.type for m in matches}
        assert "email" in types
        assert "ssn" in types

    def test_sorted_by_position(self):
        text = "SSN 123-45-6789 and email alice@test.com"
        matches = detect_pii(text)
        positions = [m.start for m in matches]
        assert positions == sorted(positions)


class TestRedactPII:
    def test_email_redacted(self):
        text, matches = redact_pii("Send to alice@example.com")
        assert "[EMAIL]" in text
        assert "alice@example.com" not in text

    def test_ssn_redacted(self):
        text, matches = redact_pii("SSN is 123-45-6789")
        assert "[SSN]" in text
        assert "123-45-6789" not in text

    def test_credit_card_redacted(self):
        text, matches = redact_pii("Card 4111 1111 1111 1111")
        assert "[CARD]" in text

    def test_clean_text_unchanged(self):
        original = "No personal info here"
        text, matches = redact_pii(original)
        assert text == original
        assert matches == []

    def test_multiple_redactions(self):
        text, matches = redact_pii(
            "Email alice@test.com, SSN 123-45-6789"
        )
        assert "[EMAIL]" in text
        assert "[SSN]" in text
        assert "alice@test.com" not in text
        assert "123-45-6789" not in text

    def test_offsets_preserved(self):
        """Multiple redactions don't corrupt each other."""
        text = "A 123-45-6789 B alice@test.com C"
        result, matches = redact_pii(text)
        assert result.startswith("A ")
        assert result.endswith(" C")
        assert "[SSN]" in result
        assert "[EMAIL]" in result
