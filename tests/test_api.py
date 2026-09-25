# tests/test_api.py
# ============================================================
# Unit Tests for the FastAPI Prediction Endpoint
# ============================================================
"""
Tests the API contract using FastAPI's TestClient:

* Valid requests return 200 with correct schema.
* Invalid payloads are rejected with 422 (Pydantic validation).
* Batch endpoint works correctly.
* Health endpoint reports model status.

Note: These tests load the real trained model.  Run ``python run_pipeline.py``
before executing these tests.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import BEST_MODEL_PATH


# ── Fixtures ──────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Create a TestClient that triggers the app lifespan (model loading)."""
    with TestClient(app) as c:
        yield c


@pytest.fixture
def valid_payload() -> dict:
    """A well-formed employee payload."""
    return {
        "age": 35,
        "gender": "Male",
        "marital_status": "Married",
        "employment_type": "Full-time",
        "region": "West",
        "has_dependents": "Yes",
        "salary": 72000.0,
        "tenure_years": 5.2,
    }


@pytest.fixture
def model_exists() -> bool:
    """Check whether the trained model file exists."""
    return BEST_MODEL_PATH.exists()


# ── Tests: Root & Health ──────────────────────────────────
class TestGeneralEndpoints:
    """Tests for non-prediction endpoints."""

    def test_root(self, client):
        """GET / should return a welcome message."""
        resp = client.get("/")
        assert resp.status_code == 200
        body = resp.json()
        assert "message" in body

    def test_health(self, client):
        """GET /health should return status and model info."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "status" in body
        assert "model_loaded" in body


# ── Tests: Single Prediction ─────────────────────────────
class TestPredictEndpoint:
    """Tests for POST /predict."""

    def test_valid_prediction(self, client, valid_payload, model_exists):
        """A valid payload should return 200 with prediction fields."""
        if not model_exists:
            pytest.skip("Model not trained yet — run `python run_pipeline.py` first.")

        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 200

        body = resp.json()
        assert "enrolled_prediction" in body
        assert "enrollment_probability" in body
        assert body["enrolled_prediction"] in (0, 1)
        assert 0.0 <= body["enrollment_probability"] <= 1.0

    def test_invalid_age(self, client, valid_payload):
        """Age outside valid range should fail validation."""
        payload = {**valid_payload, "age": 5}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_invalid_gender(self, client, valid_payload):
        """Unknown gender value should fail validation."""
        payload = {**valid_payload, "gender": "Unknown"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_field(self, client, valid_payload):
        """Omitting a required field should fail validation."""
        del valid_payload["salary"]
        resp = client.post("/predict", json=valid_payload)
        assert resp.status_code == 422

    def test_negative_salary(self, client, valid_payload):
        """Negative salary should fail validation."""
        payload = {**valid_payload, "salary": -1000}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422


# ── Tests: Batch Prediction ──────────────────────────────
class TestBatchEndpoint:
    """Tests for POST /predict/batch."""

    def test_batch_prediction(self, client, valid_payload, model_exists):
        """Batch of valid payloads should return predictions list."""
        if not model_exists:
            pytest.skip("Model not trained yet.")

        resp = client.post("/predict/batch", json=[valid_payload, valid_payload])
        assert resp.status_code == 200

        body = resp.json()
        assert "predictions" in body
        assert len(body["predictions"]) == 2

    def test_empty_batch(self, client):
        """Empty list should be rejected."""
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 400
