"""Tests for the FastAPI app in api/index.py, including compatibility with the
response shape that public/script.js expects.

External failure modes (model load failure, internal prediction failure) are
mocked; the happy-path tests exercise the real committed model artifacts.
"""
import pytest
from fastapi.testclient import TestClient

import api.index as api_index
from tools.risk_prediction import ModelLoadError, PredictionError

VALID_PATIENT = {
    "age": 52,
    "sex": "Male",
    "cp": 0,
    "trestbps": 125,
    "chol": 212,
    "fbs": 0,
    "restecg": 1,
    "thalach": 168,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}


@pytest.fixture()
def client():
    return TestClient(api_index.app)


def test_health_endpoint_reports_healthy_when_model_loaded(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    # Phase 3 added additional fields (model_loaded, scaler_loaded, bedrock_*);
    # the original "status" field is unchanged.
    assert response.json()["status"] == "healthy"


def test_predict_endpoint_returns_expected_shape(client):
    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 200

    body = response.json()
    # Contract public/script.js relies on: result.prediction, result.probability.
    assert set(body.keys()) == {"prediction", "probability"}
    assert body["prediction"] in (0, 1)
    assert isinstance(body["probability"], float)
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_endpoint_rejects_missing_field(client):
    payload = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_rejects_extra_field(client):
    payload = {**VALID_PATIENT, "extra": "nope"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_categorical(client):
    payload = {**VALID_PATIENT, "thal": 99}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_sex(client):
    payload = {**VALID_PATIENT, "sex": "Unknown"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_handles_model_load_failure_without_leaking_internals(client, monkeypatch):
    def _raise_model_load_error(self, data):
        raise ModelLoadError("[Errno 2] No such file or directory: '/secret/internal/path.pkl'")

    monkeypatch.setattr(type(api_index._prediction_service), "predict", _raise_model_load_error)

    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "secret" not in detail
    assert "/internal/" not in detail


def test_predict_endpoint_handles_internal_prediction_failure_without_leaking_internals(client, monkeypatch):
    def _raise_prediction_error(self, data):
        raise PredictionError("Prediction failed: Traceback... at api_key=SECRET123")

    monkeypatch.setattr(type(api_index._prediction_service), "predict", _raise_prediction_error)

    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "SECRET123" not in detail


def test_health_endpoint_reports_degraded_when_model_not_ready(client, monkeypatch):
    monkeypatch.setattr(type(api_index._prediction_service), "is_ready", property(lambda self: False))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
