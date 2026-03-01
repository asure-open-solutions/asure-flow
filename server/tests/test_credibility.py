"""Tests for web source credibility scoring."""

from asure_flow.agent.tools import _score_credibility


class TestCredibilityScoring:
    def test_gov_domain(self):
        result = _score_credibility("https://www.cdc.gov/health/tips")
        assert result["tier"] == "high"
        assert result["score"] >= 0.8

    def test_edu_domain(self):
        result = _score_credibility("https://cs.stanford.edu/papers/ml.pdf")
        assert result["tier"] == "high"

    def test_wikipedia(self):
        result = _score_credibility("https://en.wikipedia.org/wiki/Python")
        assert result["tier"] == "high"

    def test_arxiv(self):
        result = _score_credibility("https://arxiv.org/abs/2401.12345")
        assert result["tier"] == "high"

    def test_nature(self):
        result = _score_credibility("https://www.nature.com/articles/s41586-024")
        assert result["tier"] == "high"

    def test_reddit_low(self):
        result = _score_credibility("https://www.reddit.com/r/science/")
        assert result["tier"] == "low"
        assert result["score"] <= 0.4

    def test_quora_low(self):
        result = _score_credibility("https://www.quora.com/What-is-AI")
        assert result["tier"] == "low"

    def test_medium_domain(self):
        result = _score_credibility("https://blog.example.com/article")
        assert result["tier"] == "medium"
        assert 0.4 < result["score"] < 0.8

    def test_nytimes_high(self):
        result = _score_credibility("https://www.nytimes.com/2024/01/01/article")
        assert result["tier"] == "high"

    def test_bbc_high(self):
        result = _score_credibility("https://www.bbc.com/news/world")
        assert result["tier"] == "high"

    def test_empty_url(self):
        result = _score_credibility("")
        assert result["tier"] == "medium"

    def test_invalid_url(self):
        result = _score_credibility("not-a-url")
        assert result["tier"] == "medium"
