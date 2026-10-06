"""Validation logic for scoring results."""

from callscore.logger import get_logger
from callscore.models import Scorecard, ScoringResult, ValidationResult

logger = get_logger(__name__)


def validate_scoring_result(result: ScoringResult, scorecard: Scorecard) -> ValidationResult:
    """Validate a scoring result against business rules."""
    errors: list[str] = []
    warnings: list[str] = []

    logger.info(
        "Validating scoring result",
        call_id=result.call_id,
        scorecard_id=scorecard.scorecard_id,
    )

    # Check for missing criteria
    expected_criterion_ids = {criterion.id for criterion in scorecard.criteria}
    actual_criterion_ids = {score.criterion_id for score in result.criteria_scores}

    missing_criteria = expected_criterion_ids - actual_criterion_ids
    if missing_criteria:
        errors.append(f"Missing criteria: {', '.join(sorted(missing_criteria))}")

    # Check each criterion score
    for criterion_score in result.criteria_scores:
        if criterion_score.score is not None:
            # Check score is in valid range (0-1)
            if not (0.0 <= criterion_score.score <= 1.0):
                errors.append(
                    f"Criterion {criterion_score.criterion_id} score out of range: {criterion_score.score}"
                )

            # Check rationale is not empty
            if not criterion_score.rationale.strip():
                errors.append(
                    f"Criterion {criterion_score.criterion_id} has empty rationale"
                )

        # Check confidence threshold (warning)
        if criterion_score.confidence < 0.65:
            warnings.append(
                f"Criterion {criterion_score.criterion_id} has low confidence: {criterion_score.confidence:.2f}"
            )

    # Check weighted total is in valid range (0-1)
    if not (0.0 <= result.weighted_total <= 1.0):
        errors.append(f"Weighted total out of range: {result.weighted_total}")

    # Warning if total is very low
    if result.weighted_total < 0.3:
        warnings.append(f"Low weighted total: {result.weighted_total:.2f}")

    # Warning if all confidences are suspiciously high
    if result.criteria_scores:
        all_confidences = [cs.confidence for cs in result.criteria_scores]
        if all(conf > 0.98 for conf in all_confidences):
            warnings.append("All criterion confidences are > 0.98 (possible overconfidence)")

    valid = len(errors) == 0

    logger.info(
        "Validation complete",
        call_id=result.call_id,
        valid=valid,
        num_errors=len(errors),
        num_warnings=len(warnings),
    )

    return ValidationResult(valid=valid, errors=errors, warnings=warnings)
