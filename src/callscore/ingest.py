"""Database ingestion for scoring results."""

import os

from sqlalchemy import Boolean, Float, String, UniqueConstraint, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from callscore.logger import get_logger
from callscore.models import ScoringResult, ValidationResult

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""


class CallScore(Base):
    """Model for call score records."""

    __tablename__ = "call_scores"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(String(255), index=True)
    scorecard_id: Mapped[str] = mapped_column(String(100))
    scorecard_version: Mapped[str] = mapped_column(String(50))
    criterion_id: Mapped[str] = mapped_column(String(50))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(String(2000))
    weighted_total: Mapped[float] = mapped_column(Float)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    raw_llm_response: Mapped[str | None] = mapped_column(String(5000), nullable=True)
    scored_at: Mapped[str] = mapped_column(String(50))

    __table_args__ = (
        # Unique constraint for idempotency
        UniqueConstraint(
            "call_id", "scorecard_id", "scorecard_version", "criterion_id", name="uq_call_score"
        ),
    )


# Module-level engine
_engine = None
_async_session_maker = None


def get_engine():
    """Get or create the async engine."""
    global _engine, _async_session_maker
    if _engine is None:
        database_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./call_scores.db")

        # Only set pool args for non-SQLite databases
        if not database_url.startswith("sqlite"):
            _engine = create_async_engine(
                database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )
        else:
            _engine = create_async_engine(database_url)

        _async_session_maker = async_sessionmaker(_engine, expire_on_commit=False)

    return _engine


async def create_tables() -> None:
    """Create database tables."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")


async def ingest_scores(
    result: ScoringResult, validation: ValidationResult, session: AsyncSession | None = None
) -> None:
    """Upsert scoring results to the database.

    Only ingests if validation passed. Uses INSERT OR REPLACE for idempotency.

    Args:
        result: The scoring result to ingest
        validation: The validation result
        session: Optional async session for testing. If None, creates a new session.
    """
    if not validation.valid:
        logger.warning(
            "Skipping ingestion due to validation errors",
            call_id=result.call_id,
            errors=validation.errors,
        )
        return

    # Ensure engine is initialized
    get_engine()

    if session is None:
        async with _async_session_maker() as session:
            await _do_ingest(result, session)
    else:
        await _do_ingest(result, session)


async def _do_ingest(result: ScoringResult, session: AsyncSession) -> None:
    """Perform the actual ingestion."""
    for criterion_score in result.criteria_scores:
        # Flag if confidence is low
        flagged = criterion_score.confidence < 0.65

        # Check if record exists
        stmt = select(CallScore).where(
            CallScore.call_id == result.call_id,
            CallScore.scorecard_id == result.scorecard_id,
            CallScore.scorecard_version == result.scorecard_version,
            CallScore.criterion_id == criterion_score.criterion_id,
        )
        existing = await session.execute(stmt)
        existing_record = existing.scalar_one_or_none()

        if existing_record:
            # Update existing record
            existing_record.score = criterion_score.score
            existing_record.confidence = criterion_score.confidence
            existing_record.rationale = criterion_score.rationale
            existing_record.weighted_total = result.weighted_total
            existing_record.flagged = flagged
            existing_record.raw_llm_response = criterion_score.raw_llm_response
            existing_record.scored_at = result.scored_at.isoformat()
        else:
            # Insert new record
            record = CallScore(
                call_id=result.call_id,
                scorecard_id=result.scorecard_id,
                scorecard_version=result.scorecard_version,
                criterion_id=criterion_score.criterion_id,
                score=criterion_score.score,
                confidence=criterion_score.confidence,
                rationale=criterion_score.rationale,
                weighted_total=result.weighted_total,
                flagged=flagged,
                raw_llm_response=criterion_score.raw_llm_response,
                scored_at=result.scored_at.isoformat(),
            )
            session.add(record)

    await session.commit()

    logger.info(
        "Scores ingested",
        call_id=result.call_id,
        num_criteria=len(result.criteria_scores),
    )
