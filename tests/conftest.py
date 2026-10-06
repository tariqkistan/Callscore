"""Pytest configuration and shared fixtures."""

import json
import os
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from callscore.ingest import Base
from callscore.models import Scorecard

# In-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def sample_scorecard() -> Scorecard:
    """Sample scorecard for testing."""
    return Scorecard(
        scorecard_id="SC-TEST-V1",
        version="1.0",
        criteria=[
            {
                "id": "C01",
                "name": "test_criterion",
                "description": "A test criterion",
                "weight": 1.0,
                "scoring_type": "binary",
                "rubric": "1.0 if pass, 0.0 if fail",
                "examples": [
                    {
                        "transcript_snippet": "Good example",
                        "score": 1.0,
                        "reason": "Passes",
                    },
                    {
                        "transcript_snippet": "Bad example",
                        "score": 0.0,
                        "reason": "Fails",
                    },
                ],
            }
        ],
    )


@pytest.fixture
async def in_memory_db():
    """Create an in-memory SQLite database for testing."""
    # Override the DATABASE_URL for testing
    os.environ["DATABASE_URL"] = TEST_DATABASE_URL

    # Reset the module-level engine
    from callscore import ingest
    ingest._engine = None
    ingest._async_session_maker = None

    engine = create_async_engine(TEST_DATABASE_URL)
    async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield async_session_maker

    await engine.dispose()

    # Reset engine
    ingest._engine = None
    ingest._async_session_maker = None


@pytest.fixture
async def db_session(in_memory_db):
    """Get a database session for testing."""
    async with in_memory_db() as session:
        yield session


@pytest.fixture
def fixture_loader():
    """Load test fixture JSON files."""

    def _load(fixture_name: str) -> dict:
        fixture_path = Path(__file__).parent / "fixtures" / f"{fixture_name}.json"
        with open(fixture_path) as f:
            return json.load(f)

    return _load
