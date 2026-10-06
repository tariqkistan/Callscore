"""Azure Function HTTP trigger for call scoring."""

import json
from pathlib import Path

import azure.functions as func

from callscore.ingest import create_tables, ingest_scores
from callscore.logger import configure_logging, get_logger
from callscore.models import Scorecard, ScoreRequest, ScoreResponse
from callscore.scoring import score_transcript
from callscore.validation import validate_scoring_result

# Configure logging on module load
configure_logging()
logger = get_logger(__name__)

app = func.FunctionApp()


def load_scorecard(scorecard_id: str, _version: str) -> Scorecard:
    """Load a scorecard from the scorecards directory."""
    scorecard_path = Path(__file__).parent.parent.parent.parent / "scorecards" / f"{scorecard_id}.json"
    with open(scorecard_path) as f:
        data = json.load(f)
    return Scorecard(**data)


@app.route(route="api/score", auth_level=func.AuthLevel.FUNCTION, methods=["POST"])
async def http_score(req: func.HttpRequest) -> func.HttpResponse:
    """HTTP trigger to score a call transcript."""
    logger.info("HTTP score request received")

    try:
        # Parse request body
        body = req.get_json()
        score_request = ScoreRequest(**body)

        logger.info(
            "Processing score request",
            call_id=score_request.call_id,
            scorecard_id=score_request.scorecard_id,
        )

        # Load scorecard
        scorecard = load_scorecard(score_request.scorecard_id, score_request.scorecard_version)

        # Ensure database tables exist
        await create_tables()

        # Score the transcript
        result = await score_transcript(
            transcript=score_request.transcript,
            scorecard=scorecard,
            call_id=score_request.call_id,
        )

        # Validate the result
        validation = validate_scoring_result(result, scorecard)

        # Ingest to database
        await ingest_scores(result, validation)

        # Build response
        response = ScoreResponse(
            call_id=result.call_id,
            weighted_total=result.weighted_total,
            criteria_scores=result.criteria_scores,
            valid=validation.valid,
            errors=validation.errors,
            warnings=validation.warnings,
        )

        logger.info(
            "Score request completed",
            call_id=score_request.call_id,
            valid=validation.valid,
            weighted_total=result.weighted_total,
        )

        return func.HttpResponse(
            body=response.model_dump_json(),
            status_code=200 if validation.valid else 422,
            mimetype="application/json",
        )

    except Exception as e:
        logger.error("Error processing score request", error=str(e))
        return func.HttpResponse(
            body=json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
