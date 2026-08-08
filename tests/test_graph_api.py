"""Tests for POST /api/explain-risk as wired to the LangGraph workflow (Phase 4),
plus regression checks that /api/predict and the Phase 2/3 response contracts
are unaffected. AWS/Bedrock calls are mocked at the point agent/nodes.py looks
up the explanation service.
"""
import json
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import agent.nodes as nodes
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


@pytest.fixture()
def bedrock_disabled(monkeypatch):
    service = MagicMock()
    service.is_available.return_value = False
    monkeypatch.setattr(nodes, "get_explanation_service", lambda: service)
    return service


# --- Regression: /api/predict is completely unaffected by Phase 4 ------------

def test_predict_endpoint_unchanged_by_langgraph_integration(client, bedrock_disabled):
    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    assert set(response.json().keys()) == {"prediction", "probability"}


def test_predict_endpoint_does_not_touch_the_graph(client, monkeypatch):
    """/api/predict must never invoke agent.graph at all."""
    called = []
    monkeypatch.setattr(api_index, "invoke_cardio_graph", lambda **kwargs: called.append(1))
    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    assert called == []


# --- /api/explain-risk now flows through LangGraph ----------------------------

def test_explain_risk_uses_graph_and_returns_request_id(client, bedrock_disabled):
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_available"] is False
    assert body["request_id"]  # graph-generated id present
    assert body["prediction"] in (0, 1)


def test_explain_risk_bedrock_disabled_fallback(client, bedrock_disabled):
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_available"] is False
    assert body["unavailable_reason"]
    assert "prediction" in body and "probability" in body


def test_explain_risk_success_path(client, monkeypatch):
    valid_json = {
        "summary": "Educational summary.",
        "risk_category": "Lower modeled risk of heart disease",
        "probability": 0.22,
        "input_factors": [],
        "educational_information": ["General preventive info. [source_1]"],
        "questions_for_professional": ["What screening is right for me?"],
        "citations": [{"id": "source_1", "title": "CDC", "uri": "https://cdc.gov/heart"}],
        "disclaimer": "Educational estimate only.",
    }
    from tools.knowledge_retrieval import RetrievedChunk
    chunk = RetrievedChunk(text="text", source_uri="https://cdc.gov/heart", score=0.9, metadata={})

    service = MagicMock()
    service.is_available.return_value = True
    service.retrieval_tool.retrieve.return_value = [chunk]
    service.client.generate_explanation.return_value = json.dumps(valid_json)
    monkeypatch.setattr(nodes, "get_explanation_service", lambda: service)

    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_available"] is True
    assert body["citations"] == [{"id": "source_1", "title": "CDC", "uri": "https://cdc.gov/heart"}]
    assert body["disclaimer"] == "Educational estimate only."


# --- Blocked medical requests return safe, structured responses --------------

def test_explain_risk_blocked_medication_request_returns_200_with_safety_status(client, bedrock_disabled):
    payload = {**VALID_PATIENT, "user_message": "What medication should I take?"}
    response = client.post("/api/explain-risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["safety_status"] == "blocked"
    assert body["safety_category"] == "medication_request"
    assert body["explanation_available"] is False
    # Prediction preserved even though the free-text question was refused.
    assert body["prediction"] in (0, 1)


def test_explain_risk_blocked_emergency_request(client, bedrock_disabled):
    payload = {**VALID_PATIENT, "user_message": "Am I having a heart attack right now?"}
    response = client.post("/api/explain-risk", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["safety_category"] == "emergency_request"


def test_explain_risk_user_message_too_long_is_rejected_by_schema(client, bedrock_disabled):
    payload = {**VALID_PATIENT, "user_message": "x" * 501}
    response = client.post("/api/explain-risk", json=payload)
    assert response.status_code == 422


# --- Validation: still returns 422 at the API layer for malformed input -----

def test_explain_risk_invalid_patient_input_returns_422(client, bedrock_disabled):
    payload = {**VALID_PATIENT, "sex": "Unknown"}
    response = client.post("/api/explain-risk", json=payload)
    assert response.status_code == 422


def test_explain_risk_missing_field_returns_422(client, bedrock_disabled):
    payload = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    response = client.post("/api/explain-risk", json=payload)
    assert response.status_code == 422


# --- No internal details leak through the API ---------------------------------

def test_explain_risk_never_exposes_raw_node_names_or_internals(client, bedrock_disabled):
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    body_text = json.dumps(response.json())
    for leak_indicator in ("Traceback", "boto3", "AccessDenied", "arn:aws", "AWS_SECRET"):
        assert leak_indicator not in body_text


def test_explain_risk_graph_execution_failure_returns_500_not_raw_error(client, monkeypatch):
    monkeypatch.setattr(
        api_index, "invoke_cardio_graph",
        lambda **kwargs: {
            "request_id": "abc123",
            "explanation_available": False,
            "error_code": "graph_execution_failed",
            "error_message": "An unexpected error occurred while processing this request.",
        },
    )
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 500
    assert "Traceback" not in response.json()["detail"]


def test_explain_risk_model_unavailable_returns_503(client, monkeypatch):
    monkeypatch.setattr(
        api_index, "invoke_cardio_graph",
        lambda **kwargs: {
            "request_id": "abc123",
            "explanation_available": False,
            "error_code": "model_unavailable",
            "error_message": "Prediction service is unavailable.",
        },
    )
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 503
