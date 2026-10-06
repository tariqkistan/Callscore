"""Pydantic models for callscore."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ScorecardExample(BaseModel):
    """Few-shot example for a scorecard criterion."""

    transcript_snippet: str = Field(..., description="Example transcript text")
    score: float = Field(..., ge=0.0, le=1.0, description="Score for this example")
    reason: str = Field(..., description="Explanation for the score")


class ScorecardCriterion(BaseModel):
    """A single criterion in a scorecard."""

    id: str = Field(..., description="Criterion ID (e.g., C01)")
    name: str = Field(..., description="Criterion name")
    description: str = Field(..., description="What this criterion evaluates")
    weight: float = Field(..., ge=0.0, le=1.0, description="Weight in total score")
    scoring_type: Literal["binary", "scale"] = Field(
        ..., description="binary: 0 or 1, scale: 0-1"
    )
    rubric: str = Field(..., description="Scoring rubric for the criterion")
    examples: list[ScorecardExample] = Field(
        default_factory=list, description="Few-shot examples"
    )


class Scorecard(BaseModel):
    """A complete scorecard with multiple criteria."""

    scorecard_id: str = Field(..., description="Scorecard identifier")
    version: str = Field(..., description="Scorecard version")
    criteria: list[ScorecardCriterion] = Field(..., description="List of criteria")

    @field_validator("criteria")
    @classmethod
    def validate_weights_sum_to_one(cls, v: list[ScorecardCriterion]) -> list[ScorecardCriterion]:
        """Ensure all criterion weights sum to 1.0."""
        total_weight = sum(criterion.weight for criterion in v)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            raise ValueError(f"Criterion weights must sum to 1.0, got {total_weight}")
        return v


class CriterionScore(BaseModel):
    """Score for a single criterion."""

    criterion_id: str = Field(..., description="Criterion ID")
    score: float | None = Field(None, ge=0.0, le=1.0, description="Score (null if failed)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence 0-1")
    rationale: str = Field(..., description="Explanation for the score")
    raw_llm_response: str | None = Field(None, description="Raw LLM response for audit")


class ScoringResult(BaseModel):
    """Complete scoring result for a call."""

    call_id: str = Field(..., description="Call identifier")
    scorecard_id: str = Field(..., description="Scorecard used")
    scorecard_version: str = Field(..., description="Scorecard version")
    criteria_scores: list[CriterionScore] = Field(..., description="Scores per criterion")
    weighted_total: float = Field(..., ge=0.0, le=1.0, description="Weighted total score")
    scored_at: datetime = Field(default_factory=lambda: datetime.now(UTC), description="When scored")


class ScoreRequest(BaseModel):
    """Request to score a call."""

    call_id: str = Field(..., description="Call identifier")
    transcript: str = Field(..., description="Full transcript text")
    scorecard_id: str = Field(default="SC-DEBT-V1", description="Scorecard to use")
    scorecard_version: str = Field(default="1.0", description="Scorecard version")


class ScoreResponse(BaseModel):
    """Response from scoring a call."""

    call_id: str = Field(..., description="Call identifier")
    weighted_total: float = Field(..., ge=0.0, le=1.0, description="Weighted total score")
    criteria_scores: list[CriterionScore] = Field(..., description="Scores per criterion")
    valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")


class ValidationResult(BaseModel):
    """Result of validating a scoring result."""

    valid: bool = Field(..., description="Whether validation passed")
    errors: list[str] = Field(default_factory=list, description="Validation errors")
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
