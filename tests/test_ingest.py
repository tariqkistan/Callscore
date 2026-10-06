"""Tests for database ingestion logic."""


import pytest
from sqlalchemy import select

from callscore.ingest import CallScore, ingest_scores
from callscore.models import (
    CriterionScore,
    Scorecard,
    ScorecardCriterion,
    ScoringResult,
    ValidationResult,
)


@pytest.mark.asyncio
async def test_ingest_upsert_idempotency(db_session):
    """Test that ingesting the same call twice results in one row per criterion."""
    from callscore.ingest import get_engine

    # Ensure engine is initialized with test database
    get_engine()

    _scorecard = Scorecard(
        scorecard_id="SC-TEST",
        version="1.0",
        criteria=[
            ScorecardCriterion(
                id="C01",
                name="test",
                description="Test",
                weight=1.0,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            )
        ],
    )

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
                raw_llm_response='{"score": 1.0}',
            )
        ],
        weighted_total=1.0,
    )

    validation = ValidationResult(valid=True, errors=[], warnings=[])

    # Ingest first time
    await ingest_scores(result, validation, session=db_session)

    # Check one row exists
    stmt = select(CallScore).where(CallScore.call_id == "TEST-001")
    rows = await db_session.execute(stmt)
    first_count = len(rows.all())
    assert first_count == 1

    # Ingest second time (same call)
    await ingest_scores(result, validation, session=db_session)

    # Check still only one row (UPSERT)
    stmt = select(CallScore).where(CallScore.call_id == "TEST-001")
    rows = await db_session.execute(stmt)
    second_count = len(rows.all())
    assert second_count == 1


@pytest.mark.asyncio
async def test_ingest_flagged_low_confidence(db_session):
    """Test that low confidence scores are flagged."""
    from callscore.ingest import get_engine

    get_engine()

    _scorecard = Scorecard(
        scorecard_id="SC-TEST",
        version="1.0",
        criteria=[
            ScorecardCriterion(
                id="C01",
                name="test",
                description="Test",
                weight=1.0,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            )
        ],
    )

    result = ScoringResult(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=1.0,
                confidence=0.5,  # Low confidence
                rationale="Good",
            )
        ],
        weighted_total=1.0,
    )

    validation = ValidationResult(valid=True, errors=[], warnings=[])

    await ingest_scores(result, validation, session=db_session)

    # Check that flagged is True
    stmt = select(CallScore).where(CallScore.call_id == "TEST-001")
    rows = await db_session.execute(stmt)
    row = rows.scalar_one()
    assert row.flagged is True


@pytest.mark.asyncio
async def test_ingest_not_flagged_high_confidence(db_session):
    """Test that high confidence scores are not flagged."""
    from callscore.ingest import get_engine

    get_engine()

    _scorecard = Scorecard(
        scorecard_id="SC-TEST",
        version="1.0",
        criteria=[
            ScorecardCriterion(
                id="C01",
                name="test",
                description="Test",
                weight=1.0,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            )
        ],
    )

    result = ScoringResult(
        call_id="TEST-001",
        scorecard_id="SC-TEST",
        scorecard_version="1.0",
        criteria_scores=[
            CriterionScore(
                criterion_id="C01",
                score=1.0,
                confidence=0.9,  # High confidence
                rationale="Good",
            )
        ],
        weighted_total=1.0,
    )

    validation = ValidationResult(valid=True, errors=[], warnings=[])

    await ingest_scores(result, validation, session=db_session)

    # Check that flagged is False
    stmt = select(CallScore).where(CallScore.call_id == "TEST-001")
    rows = await db_session.execute(stmt)
    row = rows.scalar_one()
    assert row.flagged is False


@pytest.mark.asyncio
async def test_ingest_raw_response_stored(db_session):
    """Test that raw LLM response is stored."""
    from callscore.ingest import get_engine

    get_engine()

    _scorecard = Scorecard(
        scorecard_id="SC-TEST",
        version="1.0",
        criteria=[
            ScorecardCriterion(
                id="C01",
                name="test",
                description="Test",
                weight=1.0,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            )
        ],
    )

    raw_response = '{"score": 1.0, "confidence": 0.9, "rationale": "Test"}'
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
                raw_llm_response=raw_response,
            )
        ],
        weighted_total=1.0,
    )

    validation = ValidationResult(valid=True, errors=[], warnings=[])

    await ingest_scores(result, validation, session=db_session)

    # Check that raw response is stored
    stmt = select(CallScore).where(CallScore.call_id == "TEST-001")
    rows = await db_session.execute(stmt)
    row = rows.scalar_one()
    assert row.raw_llm_response == raw_response


@pytest.mark.asyncio
async def test_ingest_skip_on_validation_error(db_session):
    """Test that ingestion is skipped when validation fails."""
    _scorecard = Scorecard(
        scorecard_id="SC-TEST",
        version="1.0",
        criteria=[
            ScorecardCriterion(
                id="C01",
                name="test",
                description="Test",
                weight=1.0,
                scoring_type="binary",
                rubric="Test",
                examples=[],
            )
        ],
    )

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

    validation = ValidationResult(
        valid=False, errors=["Test error"], warnings=[]
    )

    await ingest_scores(result, validation)

    # Check that no rows were inserted
    stmt = select(CallScore).where(CallScore.call_id == "TEST-001")
    rows = await db_session.execute(stmt)
    assert rows.scalar_one_or_none() is None


@pytest.mark.asyncio
async def test_ingest_multiple_criteria(db_session):
    """Test ingestion of multiple criteria for a single call."""
    from callscore.ingest import get_engine

    get_engine()

    _scorecard = Scorecard(
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
                score=0.5,
                confidence=0.8,
                rationale="Okay",
            ),
        ],
        weighted_total=0.75,
    )

    validation = ValidationResult(valid=True, errors=[], warnings=[])

    await ingest_scores(result, validation, session=db_session)

    # Check that two rows exist (one per criterion)
    stmt = select(CallScore).where(CallScore.call_id == "TEST-001")
    rows = await db_session.execute(stmt)
    assert len(rows.all()) == 2

    # Check that scores are correct
    stmt = select(CallScore).where(
        CallScore.call_id == "TEST-001", CallScore.criterion_id == "C01"
    )
    rows = await db_session.execute(stmt)
    row = rows.scalar_one()
    assert row.score == 1.0

    stmt = select(CallScore).where(
        CallScore.call_id == "TEST-001", CallScore.criterion_id == "C02"
    )
    rows = await db_session.execute(stmt)
    row = rows.scalar_one()
    assert row.score == 0.5
