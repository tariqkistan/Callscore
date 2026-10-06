"""Tests for validation logic."""


import pytest

from callscore.models import CriterionScore, Scorecard, ScorecardCriterion, ScoringResult
from callscore.validation import validate_scoring_result


@pytest.fixture
def sample_scorecard():
    """Sample scorecard for testing."""
    return Scorecard(
        scorecard_id="SC-TEST",
        version="1.0",
        criteria=[
            ScorecardCriterion(
                id="C01",
                name="test1",
                description="Test 1",
                weight=0.5,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            ),
            ScorecardCriterion(
                id="C02",
                name="test2",
                description="Test 2",
                weight=0.5,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            ),
        ],
    )


def test_validate_missing_criteria(sample_scorecard):
    """Test validation error for missing criteria."""
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
        weighted_total=0.5,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    assert not validation.valid
    assert any("Missing criteria" in err for err in validation.errors)
    assert "C02" in validation.errors[0]


def test_validate_score_out_of_range(sample_scorecard):
    """Test validation error for score out of range."""
    # Use model_construct to bypass Pydantic validation for this test
    result = ScoringResult.model_construct(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore.model_construct(
                criterion_id="C01",
                score=1.5,
                confidence=0.9,
                rationale="Bad",
            ),
            CriterionScore.model_construct(
                criterion_id="C02",
                score=1.0,
                confidence=0.9,
                rationale="Good",
            ),
        ],
        weighted_total=1.0,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    assert not validation.valid
    assert any("out of range" in err for err in validation.errors)


def test_validate_empty_rationale(sample_scorecard):
    """Test validation error for empty rationale."""
    result = ScoringResult(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=1.0,
                confidence=0.9,
                rationale="   ",
            ),
            CriterionScore(
                criterion_id="C02",
                score=1.0,
                confidence=0.9,
                rationale="Good",
            ),
        ],
        weighted_total=1.0,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    assert not validation.valid
    assert any("empty rationale" in err for err in validation.errors)


def test_validate_weighted_total_out_of_range(sample_scorecard):
    """Test validation error for weighted total out of range."""
    # Use model_construct to bypass Pydantic validation for this test
    result = ScoringResult.model_construct(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=1.0,
                confidence=0.9,
                rationale="Good",
            ),
            CriterionScore(
                criterion_id="C02",
                score=1.0,
                confidence=0.9,
                rationale="Good",
            ),
        ],
        weighted_total=1.5,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    assert not validation.valid
    assert any("Weighted total out of range" in err for err in validation.errors)


def test_validate_low_confidence_warning(sample_scorecard):
    """Test validation warning for low confidence."""
    result = ScoringResult(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=1.0,
                confidence=0.5,
                rationale="Good",
            ),
            CriterionScore(
                criterion_id="C02",
                score=1.0,
                confidence=0.9,
                rationale="Good",
            ),
        ],
        weighted_total=1.0,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    assert validation.valid
    assert any("low confidence" in warn for warn in validation.warnings)
    assert "C01" in validation.warnings[0]


def test_validate_low_total_warning(sample_scorecard):
    """Test validation warning for low weighted total."""
    result = ScoringResult(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=0.2,
                confidence=0.9,
                rationale="Poor",
            ),
            CriterionScore(
                criterion_id="C02",
                score=0.2,
                confidence=0.9,
                rationale="Poor",
            ),
        ],
        weighted_total=0.2,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    assert validation.valid
    assert any("Low weighted total" in warn for warn in validation.warnings)


def test_validate_overconfidence_warning(sample_scorecard):
    """Test validation warning for all confidences > 0.98."""
    result = ScoringResult(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=1.0,
                confidence=0.99,
                rationale="Good",
            ),
            CriterionScore(
                criterion_id="C02",
                score=1.0,
                confidence=0.99,
                rationale="Good",
            ),
        ],
        weighted_total=1.0,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    assert validation.valid
    assert any("overconfidence" in warn for warn in validation.warnings)


def test_validate_valid_result(sample_scorecard):
    """Test validation passes for a valid result."""
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
            ),
            CriterionScore(
                criterion_id="C02",
                score=1.0,
                confidence=0.9,
                rationale="Good",
            ),
        ],
        weighted_total=1.0,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    assert validation.valid
    assert len(validation.errors) == 0


def test_validate_null_score_no_error(sample_scorecard):
    """Test null scores don't cause errors (only warnings for low confidence)."""
    result = ScoringResult(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=None,
                confidence=0.0,
                rationale="Failed",
            ),
            CriterionScore(
                criterion_id="C02",
                score=1.0,
                confidence=0.9,
                rationale="Good",
            ),
        ],
        weighted_total=1.0,
    )

    validation = validate_scoring_result(result, sample_scorecard)

    # Null score is valid (just low confidence warning)
    assert validation.valid
    assert any("low confidence" in warn for warn in validation.warnings)
