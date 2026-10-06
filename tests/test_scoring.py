"""Tests for scoring logic."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from openai import AsyncAzureOpenAI

from callscore.models import (
    Scorecard,
    ScorecardCriterion,
    ScorecardExample,
)
from callscore.scoring import _build_prompt, _score_criterion, score_transcript


@pytest.fixture
def mock_openai_response():
    """Mock OpenAI response."""
    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = json.dumps(
        {
            "score": 1.0,
            "confidence": 0.9,
            "rationale": "Test rationale",
        }
    )
    return mock_response


def test_build_prompt():
    """Test prompt building for a criterion."""
    criterion = ScorecardCriterion(
        id="C01",
        name="test_criterion",
        description="Test description",
        weight=0.5,
        scoring_type="binary",
        rubric="Test rubric",
        examples=[
            ScorecardExample(
                transcript_snippet="Good example",
                score=1.0,
                reason="Pass",
            ),
            ScorecardExample(
                transcript_snippet="Bad example",
                score=0.0,
                reason="Fail",
            ),
        ],
    )

    transcript = "Test transcript text"
    messages = _build_prompt(transcript, criterion)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "test_criterion" in messages[0]["content"]
    assert "Test description" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert transcript in messages[1]["content"]


@pytest.mark.asyncio
async def test_score_criterion_success(mock_openai_response):
    """Test successful criterion scoring."""
    criterion = ScorecardCriterion(
        id="C01",
        name="test",
        description="Test",
        weight=0.5,
        scoring_type="binary",
        rubric="Test",
        examples=[],
    )

    with patch("callscore.scoring.get_client") as mock_get_client:
        mock_client = AsyncMock(spec=AsyncAzureOpenAI)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await _score_criterion("Test transcript", criterion, "CALL-001")

        assert result.criterion_id == "C01"
        assert result.score == 1.0
        assert result.confidence == 0.9
        assert result.rationale == "Test rationale"
        assert result.raw_llm_response is not None


@pytest.mark.asyncio
async def test_score_criterion_invalid_json():
    """Test criterion scoring with invalid JSON response."""
    criterion = ScorecardCriterion(
        id="C01",
        name="test",
        description="Test",
        weight=0.5,
        scoring_type="binary",
        rubric="Test",
        examples=[],
    )

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = "not valid json"

    with patch("callscore.scoring.get_client") as mock_get_client:
        mock_client = AsyncMock(spec=AsyncAzureOpenAI)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await _score_criterion("Test transcript", criterion, "CALL-001")

        assert result.criterion_id == "C01"
        assert result.score is None
        assert result.confidence == 0.0
        assert "Failed to parse" in result.rationale


@pytest.mark.asyncio
async def test_score_criterion_score_out_of_range():
    """Test criterion scoring with score out of range is set to null."""
    criterion = ScorecardCriterion(
        id="C01",
        name="test",
        description="Test",
        weight=0.5,
        scoring_type="binary",
        rubric="Test",
        examples=[],
    )

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock()]
    mock_response.choices[0].message.content = json.dumps(
        {
            "score": 1.5,
            "confidence": 0.9,
            "rationale": "Test",
        }
    )

    with patch("callscore.scoring.get_client") as mock_get_client:
        mock_client = AsyncMock(spec=AsyncAzureOpenAI)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        result = await _score_criterion("Test transcript", criterion, "CALL-001")

        assert result.score is None


@pytest.mark.asyncio
async def test_score_transcript_concurrent(mock_openai_response):
    """Test concurrent scoring of multiple criteria."""
    scorecard = Scorecard(
        scorecard_id="SC-TEST",
        version="1.0",
        criteria=[
            ScorecardCriterion(
                id="C01",
                name="test1",
                description="Test",
                weight=0.5,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            ),
            ScorecardCriterion(
                id="C02",
                name="test2",
                description="Test",
                weight=0.5,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            ),
        ],
    )

    with patch("callscore.scoring.get_client") as mock_get_client:
        mock_client = AsyncMock(spec=AsyncAzureOpenAI)
        mock_client.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_get_client.return_value = mock_client

        result = await score_transcript("Test transcript", scorecard, "CALL-001")

        assert result.call_id == "CALL-001"
        assert len(result.criteria_scores) == 2
        assert result.weighted_total == 1.0


@pytest.mark.asyncio
async def test_score_transcript_weighted_total_excludes_nulls():
    """Test weighted total calculation excludes null scores."""
    scorecard = Scorecard(
        scorecard_id="SC-TEST",
        version="1.0",
        criteria=[
            ScorecardCriterion(
                id="C01",
                name="test1",
                description="Test",
                weight=0.5,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            ),
            ScorecardCriterion(
                id="C02",
                name="test2",
                description="Test",
                weight=0.5,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            ),
        ],
    )

    # Mock one success, one failure
    async def mock_create(*args, **kwargs):
        mock_resp = AsyncMock()
        mock_resp.choices = [AsyncMock()]
        # First call returns valid, second returns invalid
        if kwargs["messages"][1]["content"].endswith("test1"):
            mock_resp.choices[0].message.content = json.dumps(
                {"score": 1.0, "confidence": 0.9, "rationale": "Good"}
            )
        else:
            mock_resp.choices[0].message.content = json.dumps(
                {"score": 0.5, "confidence": 0.9, "rationale": "Okay"}
            )
        return mock_resp

    with patch("callscore.scoring.get_client") as mock_get_client:
        mock_client = AsyncMock(spec=AsyncAzureOpenAI)
        mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
        mock_get_client.return_value = mock_client

        result = await score_transcript("Test transcript", scorecard, "CALL-001")

        # Both scored, so weighted total should be average
        assert result.weighted_total == 0.75
