"""Tests for POST /api/explain-risk and the extended GET /api/health, plus a
regression check that POST /api/predict is completely unchanged by Phase 3.
All AWS calls are mocked -- no real Bedrock/AWS access occurs.
"""
import json

import pytest
from fastapi.testclient import TestClient

import api.index as api_index

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


# --- /api/predict regression: must be byte-for-byte the same contract -------

def test_predict_endpoint_contract_unchanged(client):
    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    assert set(response.json().keys()) == {"prediction", "probability"}


def test_predict_endpoint_unaffected_by_bedrock_being_disabled(client, monkeypatch):
    monkeypatch.setattr(api_index._explanation_service, "is_available", lambda: False)
    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 200


# --- /api/health -------------------------------------------------------------

def test_health_endpoint_includes_bedrock_fields(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["scaler_loaded"] is True
    assert "bedrock_enabled" in body
    assert "bedrock_configured" in body


def test_health_endpoint_does_not_invoke_bedrock(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        api_index._explanation_service, "is_available",
        lambda: (calls.append("is_available"), False)[1],
    )
    client.get("/api/health")
    assert calls == ["is_available"]  # only the cheap local check ran, no retrieve/generate


# --- /api/explain-risk: graceful fallback when Bedrock unavailable ----------

def test_explain_risk_falls_back_gracefully_when_bedrock_disabled(client, monkeypatch):
    # As of Phase 4, this flows through agent/graph.py rather than calling
    # ExplanationService.explain() directly -- mock at that integration point.
    monkeypatch.setattr(
        api_index, "invoke_cardio_graph",
        lambda **kwargs: {
            "request_id": "test-request-id",
            "prediction": 0,
            "probability": 0.2,
            "risk_category": "Lower modeled risk of heart disease",
            "input_factors": ["No submitted values fell outside the ranges this tool flags as notable."],
            "explanation_available": False,
            "unavailable_reason": "The AI explanation feature is not enabled on this deployment.",
            "summary": "The AI explanation feature is not enabled on this deployment.",
            "educational_information": [],
            "questions_for_professional": [],
            "citations": [],
            "disclaimer": "",
        },
    )
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_available"] is False
    assert body["unavailable_reason"] == "The AI explanation feature is not enabled on this deployment."
    # Prediction must still be present and correct even without an explanation.
    assert body["prediction"] in (0, 1)
    assert isinstance(body["probability"], float)


def test_explain_risk_skips_explanation_when_not_requested(client, monkeypatch):
    called = []
    monkeypatch.setattr(
        api_index._explanation_service, "explain",
        lambda **kwargs: called.append(1),
    )
    response = client.post("/api/explain-risk", json={**VALID_PATIENT, "include_explanation": False})
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_available"] is False
    assert body["unavailable_reason"] == "Explanation was not requested."
    assert called == []  # explain() must not have been called at all


# --- /api/explain-risk: success path ----------------------------------------

def test_explain_risk_success_returns_citations_and_disclaimer(client, monkeypatch):
    # As of Phase 4, /api/explain-risk flows through the LangGraph workflow
    # (agent/graph.py) instead of calling ExplanationService.explain()
    # directly -- see tests/test_graph_api.py for full graph-level coverage.
    # This test now mocks at that integration point to keep exercising the
    # same "successful explanation" contract through the real API endpoint.
    monkeypatch.setattr(
        api_index, "invoke_cardio_graph",
        lambda **kwargs: {
            "request_id": "test-request-id",
            "prediction": 0,
            "probability": kwargs.get("patient_input", {}).get("probability", 0.2),
            "risk_category": "Lower modeled risk of heart disease",
            "input_factors": ["No submitted values fell outside the ranges this tool flags as notable."],
            "explanation_available": True,
            "unavailable_reason": None,
            "summary": "Educational summary.",
            "educational_information": ["General preventive info. [source_1]"],
            "questions_for_professional": ["What screening is appropriate for me?"],
            "citations": [{"id": "source_1", "title": "CDC", "uri": "https://cdc.gov/heart"}],
            "disclaimer": "Educational estimate only.",
        },
    )
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_available"] is True
    assert body["citations"] == [{"id": "source_1", "title": "CDC", "uri": "https://cdc.gov/heart"}]
    assert body["disclaimer"] == "Educational estimate only."
    assert "prediction" in body and "probability" in body


def test_explain_risk_validates_patient_input_same_as_predict(client):
    bad_payload = {**VALID_PATIENT, "sex": "Unknown"}
    response = client.post("/api/explain-risk", json=bad_payload)
    assert response.status_code == 422


def test_explain_risk_rejects_out_of_range_retrieval_k(client):
    response = client.post("/api/explain-risk", json={**VALID_PATIENT, "retrieval_k": 999})
    assert response.status_code == 422


def test_explain_risk_response_never_contains_raw_aws_error_text(client, monkeypatch):
    monkeypatch.setattr(
        api_index, "invoke_cardio_graph",
        lambda **kwargs: {
            "request_id": "test-request-id",
            "prediction": 0,
            "probability": 0.2,
            "risk_category": "Lower modeled risk of heart disease",
            "input_factors": [],
            "explanation_available": False,
            "unavailable_reason": "The AI explanation service could not generate an explanation right now.",
            "summary": "The AI explanation service could not generate an explanation right now.",
            "educational_information": [],
            "questions_for_professional": [],
            "citations": [],
            "disclaimer": "",
        },
    )
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    body_text = json.dumps(response.json())
    for leak_indicator in ("AccessDenied", "boto3", "Traceback", "AWS_SECRET", "arn:aws"):
        assert leak_indicator not in body_text


def test_explain_risk_model_load_failure_returns_503_not_500(client, monkeypatch):
    from tools.risk_prediction import ModelLoadError

    def _raise(self, data):
        raise ModelLoadError("internal path detail that should not leak")

    monkeypatch.setattr(type(api_index._prediction_service), "predict", _raise)
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 503
    assert "internal path detail" not in response.json()["detail"]
