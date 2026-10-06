"""Script to run the full pipeline locally with a fixture transcript."""

import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from callscore.ingest import create_tables, ingest_scores
from callscore.logger import configure_logging, get_logger
from callscore.models import Scorecard
from callscore.scoring import score_transcript
from callscore.validation import validate_scoring_result

# Load environment variables
load_dotenv()

# Configure logging
configure_logging()
logger = get_logger(__name__)


async def main() -> None:
    """Run the full pipeline with FX-001 fixture."""
    # Load fixture
    fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "FX-001.json"
    with open(fixture_path) as f:
        fixture = json.load(f)

    logger.info("Starting local pipeline run", call_id=fixture["call_id"])

    # Load scorecard
    scorecard_path = Path(__file__).parent.parent / "scorecards" / "SC-DEBT-V1.json"
    with open(scorecard_path) as f:
        scorecard_data = json.load(f)
    scorecard = Scorecard(**scorecard_data)

    # Create database tables
    await create_tables()

    # Score the transcript
    logger.info("Scoring transcript...")
    result = await score_transcript(
        transcript=fixture["transcript"],
        scorecard=scorecard,
        call_id=fixture["call_id"],
    )

    # Validate the result
    logger.info("Validating result...")
    validation = validate_scoring_result(result, scorecard)

    # Print results
    print("\n" + "=" * 60)
    print(f"Call ID: {result.call_id}")
    print(f"Scorecard: {result.scorecard_id} v{result.scorecard_version}")
    print(f"Weighted Total: {result.weighted_total:.2f}")
    print(f"Valid: {validation.valid}")
    print("=" * 60)

    print("\nCriterion Scores:")
    for cs in result.criteria_scores:
        score_str = f"{cs.score:.2f}" if cs.score is not None else "N/A"
        print(f"  {cs.criterion_id}: {score_str} (confidence: {cs.confidence:.2f})")
        print(f"    Rationale: {cs.rationale}")

    if validation.errors:
        print("\nErrors:")
        for error in validation.errors:
            print(f"  - {error}")

    if validation.warnings:
        print("\nWarnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")

    # Ingest to database
    if validation.valid:
        logger.info("Ingesting to database...")
        await ingest_scores(result, validation)
        print("\n✓ Results ingested to database")
    else:
        print("\n✗ Skipping ingestion due to validation errors")

    logger.info("Pipeline run complete")


if __name__ == "__main__":
    asyncio.run(main())
