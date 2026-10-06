"""Tests for Azure Functions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import azure.functions as func
import pytest

from callscore.functions.http_score import http_score


@pytest.mark.asyncio
async def test_http_score_valid_request():
    """Test HTTP trigger with valid request."""
    # Mock the request
    mock_request = MagicMock(spec=func.HttpRequest)
    mock_request.get_json.return_value = {
        "call_id": "TEST-001",
        "transcript": "Test transcript",
        "scorecard_id": "SC-DEBT-V1",
        "scorecard_version": "1.0",
    }

    # Mock the scoring and validation
    with patch("callscore.functions.http_score.score_transcript", new_callable=AsyncMock) as mock_score, \
         patch("callscore.functions.http_score.validate_scoring_result") as mock_validate, \
         patch("callscore.functions.http_score.ingest_scores", new_callable=AsyncMock), \
         patch("callscore.functions.http_score.create_tables", new_callable=AsyncMock):

        mock_score.return_value = MagicMock(
            call_id="TEST-001",
            scorecard_id="SC-DEBT-V1",
            scorecard_version="1.0",
            criteria_scores=[],
            weighted_total=0.8,
        )
        mock_validate.return_value = MagicMock(valid=True, errors=[], warnings=[])

        response = await http_score(mock_request)

        assert response.status_code == 200
        response_data = json.loads(response.get_body())
        assert response_data["call_id"] == "TEST-001"
        assert response_data["valid"] is True


@pytest.mark.asyncio
async def test_http_score_invalid_request():
    """Test HTTP trigger with invalid request (validation fails)."""
    # Mock the request
    mock_request = MagicMock(spec=func.HttpRequest)
    mock_request.get_json.return_value = {
        "call_id": "TEST-001",
        "transcript": "Test transcript",
    }

    # Mock the scoring and validation
    with patch("callscore.functions.http_score.score_transcript", new_callable=AsyncMock) as mock_score, \
         patch("callscore.functions.http_score.validate_scoring_result") as mock_validate, \
         patch("callscore.functions.http_score.ingest_scores", new_callable=AsyncMock), \
         patch("callscore.functions.http_score.create_tables", new_callable=AsyncMock):

        mock_score.return_value = MagicMock(
            call_id="TEST-001",
            scorecard_id="SC-DEBT-V1",
            scorecard_version="1.0",
            criteria_scores=[],
            weighted_total=0.8,
        )
        mock_validate.return_value = MagicMock(
            valid=False, errors=["Test error"], warnings=[]
        )

        response = await http_score(mock_request)

        assert response.status_code == 422
        response_data = json.loads(response.get_body())
        assert response_data["valid"] is False
        assert "Test error" in response_data["errors"]
