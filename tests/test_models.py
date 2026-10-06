"""Tests for Pydantic models."""

import pytest
from pydantic import ValidationError

from callscore.models import (
    CriterionScore,
    Scorecard,
    ScorecardCriterion,
    ScorecardExample,
    ScoreRequest,
    ScoreResponse,
    ScoringResult,
    ValidationResult,
)


def test_scorecard_example_valid():
    """Test valid ScorecardExample."""
    example = ScorecardExample(
        transcript_snippet="Test transcript",
        score=1.0,
        reason="Good",
    )
    assert example.transcript_snippet == "Test transcript"
    assert example.score == 1.0
    assert example.reason == "Good"


def test_scorecard_example_score_out_of_range():
    """Test ScorecardExample rejects score out of range."""
    with pytest.raises(ValidationError):
        ScorecardExample(
            transcript_snippet="Test",
            score=1.5,
            reason="Bad",
        )


def test_scorecard_criterion_valid():
    """Test valid ScorecardCriterion."""
    criterion = ScorecardCriterion(
        id="C01",
        name="test",
        description="Test criterion",
        weight=0.5,
        scoring_type="binary",
        rubric="Test rubric",
        examples=[],
    )
    assert criterion.id == "C01"
    assert criterion.weight == 0.5


def test_scorecard_criterion_weight_out_of_range():
    """Test ScorecardCriterion rejects weight out of range."""
    with pytest.raises(ValidationError):
        ScorecardCriterion(
            id="C01",
            name="test",
            description="Test",
            weight=1.5,
            scoring_type="binary",
            rubric="Test",
            examples=[],
        )


def test_scorecard_weights_sum_to_one():
    """Test Scorecard validates weights sum to 1.0."""
    with pytest.raises(ValidationError, match="weights must sum to 1.0"):
        Scorecard(
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
                    weight=0.3,
                    scoring_type="binary",
                    rubric="Test",
                    examples=[],
                ),
            ],
        )


def test_scorecard_weights_sum_to_one_valid():
    """Test Scorecard accepts weights that sum to 1.0."""
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
    assert scorecard.scorecard_id == "SC-TEST"


def test_criterion_score_null_score():
    """Test CriterionScore allows null score."""
    score = CriterionScore(
        criterion_id="C01",
        score=None,
        confidence=0.5,
        rationale="Failed to score",
    )
    assert score.score is None


def test_criterion_score_out_of_range():
    """Test CriterionScore rejects score out of range."""
    with pytest.raises(ValidationError):
        CriterionScore(
            criterion_id="C01",
            score=1.5,
            confidence=0.5,
            rationale="Test",
        )


def test_scoring_result_valid():
    """Test valid ScoringResult."""
    result = ScoringResult(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=1.0,
                confidence=0.9,
                rationale="Good",
            )
        ],
        weighted_total=1.0,
    )
    assert result.call_id == "TEST-001"
    assert result.weighted_total == 1.0


def test_score_request_defaults():
    """Test ScoreRequest has correct defaults."""
    request = ScoreRequest(
        call_id="TEST-001",
        transcript="Test transcript",
    )
    assert request.scorecard_id == "SC-DEBT-V1"
    assert request.scorecard_version == "1.0"


def test_score_response_valid():
    """Test valid ScoreResponse."""
    response = ScoreResponse(
        call_id="TEST-001",
        weighted_total=0.8,
        criteria_scores=[],
        valid=True,
    )
    assert response.valid is True
    assert response.errors == []


def test_validation_result_valid():
    """Test valid ValidationResult."""
    result = ValidationResult(valid=True, errors=[], warnings=[])
    assert result.valid is True
    assert result.errors == []
    assert result.warnings == []
