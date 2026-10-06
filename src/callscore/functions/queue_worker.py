"""Azure Function Service Bus trigger for call scoring."""

import json
import os
from pathlib import Path

import azure.functions as func

from callscore.ingest import create_tables, ingest_scores
from callscore.logger import configure_logging, get_logger
from callscore.models import Scorecard
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


@app.service_bus_queue_trigger(
    arg_name="msg",
    queue_name=os.getenv("SERVICE_BUS_QUEUE_NAME", "call-scoring-queue"),
    connection="AZURE_SERVICE_BUS_CONNECTION_STRING",
)
async def queue_worker(msg: func.ServiceBusMessage) -> None:
    """Service Bus trigger to score call transcripts from a queue."""
    logger.info("Service Bus message received")

    try:
        # Parse message body
        body = msg.get_body().decode("utf-8")
        payload = json.loads(body)

        # Support both direct transcript and transcript_url
        if "transcript" in payload:
            transcript = payload["transcript"]
        elif "transcript_url" in payload:
            # TODO: Load transcript from Azure Blob Storage
            raise NotImplementedError("transcript_url not yet implemented")
        else:
            raise ValueError("Payload must contain 'transcript' or 'transcript_url'")

        call_id = payload["call_id"]
        scorecard_id = payload.get("scorecard_id", "SC-DEBT-V1")
        scorecard_version = payload.get("scorecard_version", "1.0")

        logger.info(
            "Processing queue message",
            call_id=call_id,
            scorecard_id=scorecard_id,
        )

        # Load scorecard
        scorecard = load_scorecard(scorecard_id, scorecard_version)

        # Ensure database tables exist
        await create_tables()

        # Score the transcript
        result = await score_transcript(
            transcript=transcript,
            scorecard=scorecard,
            call_id=call_id,
        )

        # Validate the result
        validation = validate_scoring_result(result, scorecard)

        # Only ingest if validation passed
        if validation.valid:
            await ingest_scores(result, validation)
            logger.info(
                "Queue message processed successfully",
                call_id=call_id,
                weighted_total=result.weighted_total,
            )
        else:
            # Validation failed - raise to trigger retry
            logger.error(
                "Validation failed, triggering retry",
                call_id=call_id,
                errors=validation.errors,
            )
            raise ValueError(f"Validation failed: {validation.errors}")

    except Exception as e:
        logger.error("Error processing queue message", error=str(e))
        # Raise to trigger Service Bus retry (DLQ after max delivery count)
        raise
