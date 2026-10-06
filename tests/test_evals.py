"""Evaluation tests with real API calls.

These tests require real Azure OpenAI credentials and are skipped by default.
Run with: pytest tests/test_evals.py -m eval
"""

import os

import pytest

from callscore.models import Scorecard
from callscore.scoring import score_transcript
from callscore.validation import validate_scoring_result


@pytest.mark.eval
@pytest.mark.asyncio
async def test_eval_fx001(fixture_loader):
    """Test FX-001 fixture with real API - happy path."""
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        pytest.skip("No Azure OpenAI API key configured")

    fixture = fixture_loader("FX-001")

    # Load scorecard
    import json
    from pathlib import Path

    scorecard_path = Path(__file__).parent.parent / "scorecards" / "SC-DEBT-V1.json"
    with open(scorecard_path) as f:
        scorecard_data = json.load(f)
    scorecard = Scorecard(**scorecard_data)

    # Score the transcript
    result = await score_transcript(
        transcript=fixture["transcript"],
        scorecard=scorecard,
        call_id=fixture["call_id"],
    )

    # Validate
    validation = validate_scoring_result(result, scorecard)
    assert validation.valid, f"Validation failed: {validation.errors}"

    # Check expected scores within tolerance
    expected = fixture["expected_scores"]
    for criterion_score in result.criteria_scores:
        expected_score = expected.get(criterion_score.criterion_id)
        if expected_score is not None and criterion_score.score is not None:
            assert (
                abs(criterion_score.score - expected_score) <= 0.3
            ), f"Criterion {criterion_score.criterion_id}: expected {expected_score}, got {criterion_score.score}"

    # Check weighted total meets minimum
    assert (
        result.weighted_total >= fixture["expected_weighted_total_min"]
    ), f"Weighted total {result.weighted_total} below minimum {fixture['expected_weighted_total_min']}"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_eval_fx002(fixture_loader):
    """Test FX-002 fixture with real API - missing greeting."""
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        pytest.skip("No Azure OpenAI API key configured")

    fixture = fixture_loader("FX-002")

    # Load scorecard
    import json
    from pathlib import Path

    scorecard_path = Path(__file__).parent.parent / "scorecards" / "SC-DEBT-V1.json"
    with open(scorecard_path) as f:
        scorecard_data = json.load(f)
    scorecard = Scorecard(**scorecard_data)

    # Score the transcript
    result = await score_transcript(
        transcript=fixture["transcript"],
        scorecard=scorecard,
        call_id=fixture["call_id"],
    )

    # Validate
    validation = validate_scoring_result(result, scorecard)
    assert validation.valid, f"Validation failed: {validation.errors}"

    # Check C01 (greeting) is low
    c01_score = next(
        (cs.score for cs in result.criteria_scores if cs.criterion_id == "C01"), None
    )
    assert c01_score is not None, "C01 score is null"
    assert c01_score <= 0.3, f"C01 (greeting) should be low, got {c01_score}"

    # Check expected scores within tolerance
    expected = fixture["expected_scores"]
    for criterion_score in result.criteria_scores:
        expected_score = expected.get(criterion_score.criterion_id)
        if expected_score is not None and criterion_score.score is not None:
            assert (
                abs(criterion_score.score - expected_score) <= 0.3
            ), f"Criterion {criterion_score.criterion_id}: expected {expected_score}, got {criterion_score.score}"


@pytest.mark.eval
@pytest.mark.asyncio
async def test_eval_fx003(fixture_loader):
    """Test FX-003 fixture with real API - threatening tone."""
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        pytest.skip("No Azure OpenAI API key configured")

    fixture = fixture_loader("FX-003")

    # Load scorecard
    import json
    from pathlib import Path

    scorecard_path = Path(__file__).parent.parent / "scorecards" / "SC-DEBT-V1.json"
    with open(scorecard_path) as f:
        scorecard_data = json.load(f)
    scorecard = Scorecard(**scorecard_data)

    # Score the transcript
    result = await score_transcript(
        transcript=fixture["transcript"],
        scorecard=scorecard,
        call_id=fixture["call_id"],
    )

    # Validate
    validation = validate_scoring_result(result, scorecard)
    assert validation.valid, f"Validation failed: {validation.errors}"

    # Check C04 (empathy) is low
    c04_score = next(
        (cs.score for cs in result.criteria_scores if cs.criterion_id == "C04"), None
    )
    assert c04_score is not None, "C04 score is null"
    assert c04_score <= 0.3, f"C04 (empathy) should be low, got {c04_score}"

    # Check expected scores within tolerance
    expected = fixture["expected_scores"]
    for criterion_score in result.criteria_scores:
        expected_score = expected.get(criterion_score.criterion_id)
        if expected_score is not None and criterion_score.score is not None:
            assert (
                abs(criterion_score.score - expected_score) <= 0.3
            ), f"Criterion {criterion_score.criterion_id}: expected {expected_score}, got {criterion_score.score}"
