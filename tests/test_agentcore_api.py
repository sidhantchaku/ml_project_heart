"""Tests for the local-vs-AgentCore routing switch in api/index.py, and a
static check that no AWS credentials/clients are reachable from public/.
All AWS/AgentCore calls are mocked -- no real network access occurs.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.index as api_index
from services.agentcore_client import (
    AgentCoreAuthError,
    AgentCoreConfigurationError,
    AgentCoreUnavailableError,
)

VALID_PATIENT = {"age": 52, "sex": "Male", "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
                  "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3}


@pytest.fixture()
def client():
    return TestClient(api_index.app)


# --- regression: /api/predict never touches AgentCore at all -----------------

def test_predict_unaffected_by_agentcore_setting(client, monkeypatch):
    monkeypatch.setattr(api_index._app_settings, "use_agentcore", True)
    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    assert set(response.json().keys()) == {"prediction", "probability"}


def test_predict_never_calls_agentcore_client(client, monkeypatch):
    monkeypatch.setattr(api_index._app_settings, "use_agentcore", True)
    monkeypatch.setattr(api_index._agentcore_client, "invoke", MagicMock(side_effect=AssertionError("should not be called")))
    response = client.post("/api/predict", json=VALID_PATIENT)
    assert response.status_code == 200


# --- USE_AGENTCORE=false: local LangGraph path (default, unchanged) ---------

def test_local_routing_when_agentcore_disabled(client, monkeypatch):
    monkeypatch.setattr(api_index._app_settings, "use_agentcore", False)
    called = []
    monkeypatch.setattr(api_index._agentcore_client, "invoke", lambda **kwargs: called.append(1))
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    assert called == []  # AgentCore must never be touched when disabled


# --- USE_AGENTCORE=true: routes through AgentCore ----------------------------

def test_agentcore_routing_when_enabled_success(client, monkeypatch):
    monkeypatch.setattr(api_index._app_settings, "use_agentcore", True)
    agentcore_response = {
        "request_id": "agentcore-req-1",
        "prediction": 1,
        "probability": 0.8,
        "risk_category": "Higher modeled risk of heart disease",
        "input_factors": [],
        "explanation_available": True,
        "unavailable_reason": None,
        "summary": "Grounded explanation from AgentCore.",
        "educational_information": ["Some info. [source_1]"],
        "questions_for_professional": ["What follow-up is right for me?"],
        "citations": [{"id": "source_1", "title": "CDC", "uri": "https://cdc.gov/heart"}],
        "disclaimer": "Educational estimate only.",
    }
    monkeypatch.setattr(api_index._agentcore_client, "invoke", lambda **kwargs: agentcore_response)

    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "agentcore-req-1"
    assert body["explanation_available"] is True
    assert body["citations"] == agentcore_response["citations"]


def test_agentcore_unavailable_returns_prediction_not_fabricated_explanation(client, monkeypatch):
    monkeypatch.setattr(api_index._app_settings, "use_agentcore", True)

    def _raise(**kwargs):
        raise AgentCoreUnavailableError("runtime unreachable")

    monkeypatch.setattr(api_index._agentcore_client, "invoke", _raise)
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_available"] is False
    assert "unavailable" in body["unavailable_reason"].lower()
    # The real prediction must still be present and correct -- not fabricated.
    assert body["prediction"] in (0, 1)
    assert isinstance(body["probability"], float)


def test_agentcore_auth_error_does_not_leak_and_still_returns_prediction(client, monkeypatch):
    monkeypatch.setattr(api_index._app_settings, "use_agentcore", True)

    def _raise(**kwargs):
        raise AgentCoreAuthError("access denied detail with arn:aws:iam::123456789012:role/secret")

    monkeypatch.setattr(api_index._agentcore_client, "invoke", _raise)
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert "123456789012" not in str(body)
    assert body["prediction"] in (0, 1)


def test_agentcore_configuration_error_falls_back_to_prediction_only(client, monkeypatch):
    monkeypatch.setattr(api_index._app_settings, "use_agentcore", True)

    def _raise(**kwargs):
        raise AgentCoreConfigurationError("not configured")

    monkeypatch.setattr(api_index._agentcore_client, "invoke", _raise)
    response = client.post("/api/explain-risk", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_available"] is False
    assert body["prediction"] in (0, 1)


# --- health fields -------------------------------------------------------------

def test_health_includes_agentcore_fields(client):
    response = client.get("/api/health")
    body = response.json()
    assert "agentcore_enabled" in body
    assert "agentcore_configured" in body
    assert "agentcore_runtime_arn_present" in body
    assert "local_agent_available" in body


def test_health_never_invokes_agentcore(client, monkeypatch):
    monkeypatch.setattr(api_index._agentcore_client, "invoke", MagicMock(side_effect=AssertionError("must not be called")))
    response = client.get("/api/health")
    assert response.status_code == 200


# --- no frontend credential exposure (static check) --------------------------

def test_public_directory_never_references_aws_or_boto3():
    public_dir = Path(__file__).resolve().parent.parent / "public"
    forbidden_terms = ["boto3", "AWS_SECRET", "AWS_ACCESS_KEY", "agentRuntimeArn", "bedrock-agentcore"]
    for file_path in public_dir.rglob("*"):
        if file_path.is_file():
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for term in forbidden_terms:
                assert term not in content, f"{file_path} contains forbidden term '{term}'"
