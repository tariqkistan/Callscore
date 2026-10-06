"""GPT-4o scoring logic for call transcripts."""

import asyncio
import json
import os

from openai import AsyncAzureOpenAI
from pydantic import ValidationError

from callscore.logger import get_logger
from callscore.models import (
    CriterionScore,
    Scorecard,
    ScorecardCriterion,
    ScoringResult,
)

logger = get_logger(__name__)

# Module-level client reused across invocations
_client: AsyncAzureOpenAI | None = None


def get_client() -> AsyncAzureOpenAI:
    """Get or create the AsyncAzureOpenAI client."""
    global _client
    if _client is None:
        _client = AsyncAzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
    return _client


def _build_prompt(transcript: str, criterion: ScorecardCriterion) -> list[dict]:
    """Build the prompt for scoring a single criterion."""
    examples_text = "\n".join(
        f"Example {i+1}:\nTranscript: {ex.transcript_snippet}\nScore: {ex.score}\nReason: {ex.reason}"
        for i, ex in enumerate(criterion.examples)
    )

    system_message = f"""You are a quality assurance expert for debt collection calls. Your task is to score the following criterion:

Criterion: {criterion.name}
Description: {criterion.description}
Scoring Type: {criterion.scoring_type}
Rubric: {criterion.rubric}

Few-shot examples:
{examples_text}

Output a JSON object with the following schema:
{{
  "score": <float between 0.0 and 1.0>,
  "confidence": <float between 0.0 and 1.0>,
  "rationale": "<explanation for the score>"
}}

Be precise and objective in your scoring."""

    user_message = f"""Transcript to score:
{transcript}

Provide the score for criterion: {criterion.name}"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


async def _score_criterion(
    transcript: str, criterion: ScorecardCriterion, call_id: str
) -> CriterionScore:
    """Score a single criterion using GPT-4o."""
    logger.info(
        "Scoring criterion",
        call_id=call_id,
        criterion_id=criterion.id,
        criterion_name=criterion.name,
    )

    try:
        client = get_client()
        messages = _build_prompt(transcript, criterion)

        response = await client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            max_tokens=400,
        )

        raw_response = response.choices[0].message.content
        if raw_response is None:
            raise ValueError("Empty response from LLM")

        parsed = json.loads(raw_response)

        # Validate with Pydantic
        score_data = {
            "score": parsed.get("score"),
            "confidence": parsed.get("confidence", 0.5),
            "rationale": parsed.get("rationale", ""),
        }

        # Validate score is in range
        if score_data["score"] is not None and not (0.0 <= score_data["score"] <= 1.0):
            logger.warning(
                "Score out of range, setting to null",
                call_id=call_id,
                criterion_id=criterion.id,
                score=score_data["score"],
            )
            score_data["score"] = None

        criterion_score = CriterionScore(
            criterion_id=criterion.id,
            **score_data,
            raw_llm_response=raw_response,
        )

        logger.info(
            "Criterion scored",
            call_id=call_id,
            criterion_id=criterion.id,
            score=criterion_score.score,
            confidence=criterion_score.confidence,
        )

        return criterion_score

    except json.JSONDecodeError as e:
        logger.error(
            "Failed to parse LLM response as JSON",
            call_id=call_id,
            criterion_id=criterion.id,
            error=str(e),
        )
        return CriterionScore(
            criterion_id=criterion.id,
            score=None,
            confidence=0.0,
            rationale=f"Failed to parse LLM response: {e}",
            raw_llm_response=raw_response if "raw_response" in locals() else None,
        )
    except ValidationError as e:
        logger.error(
            "LLM response failed Pydantic validation",
            call_id=call_id,
            criterion_id=criterion.id,
            error=str(e),
        )
        return CriterionScore(
            criterion_id=criterion.id,
            score=None,
            confidence=0.0,
            rationale=f"Validation error: {e}",
            raw_llm_response=raw_response if "raw_response" in locals() else None,
        )
    except Exception as e:
        logger.error(
            "Unexpected error scoring criterion",
            call_id=call_id,
            criterion_id=criterion.id,
            error=str(e),
        )
        return CriterionScore(
            criterion_id=criterion.id,
            score=None,
            confidence=0.0,
            rationale=f"Unexpected error: {e}",
            raw_llm_response=None,
        )


async def score_transcript(
    transcript: str, scorecard: Scorecard, call_id: str
) -> ScoringResult:
    """Score a transcript against a scorecard using concurrent criterion scoring."""
    logger.info(
        "Starting transcript scoring",
        call_id=call_id,
        scorecard_id=scorecard.scorecard_id,
        scorecard_version=scorecard.version,
        num_criteria=len(scorecard.criteria),
    )

    # Score all criteria concurrently
    criteria_scores = await asyncio.gather(
        *[
            _score_criterion(transcript, criterion, call_id)
            for criterion in scorecard.criteria
        ]
    )

    # Calculate weighted total (exclude null scores)
    total_weight = 0.0
    weighted_sum = 0.0

    for criterion, score in zip(scorecard.criteria, criteria_scores, strict=True):
        if score.score is not None:
            weighted_sum += score.score * criterion.weight
            total_weight += criterion.weight

    weighted_total = weighted_sum / total_weight if total_weight > 0 else 0.0

    result = ScoringResult(
        call_id=call_id,
        scorecard_id=scorecard.scorecard_id,
        scorecard_version=scorecard.version,
        criteria_scores=criteria_scores,
        weighted_total=weighted_total,
    )

    logger.info(
        "Transcript scoring complete",
        call_id=call_id,
        weighted_total=weighted_total,
        num_scores=len([s for s in criteria_scores if s.score is not None]),
    )

    return result
