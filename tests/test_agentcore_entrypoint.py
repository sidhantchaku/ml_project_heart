"""Tests for infrastructure/agentcore/agent_entrypoint.py -- the @app.entrypoint
handler itself, including its safety net around runtime_adapter.handle_request().
"""
import json

import infrastructure.agentcore.agent_entrypoint as entrypoint

VALID_PATIENT = {
    "age": 52, "sex": "Male", "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
}


def test_valid_payload_returns_structured_response():
    result = entrypoint.invoke({"patient_input": VALID_PATIENT})
    assert result["prediction"] in (0, 1)
    assert result["request_id"]


def test_invalid_payload_returns_controlled_error_not_a_crash():
    result = entrypoint.invoke({"patient_input": {"age": 999}})
    assert result["explanation_available"] is False
    assert result["error_code"] == "invalid_input"
    # The pydantic validation detail is fine to include (it's about the user's
    # own input, not AWS/internal state) -- but no raw traceback markers.
    assert "Traceback" not in result["error_message"]


def test_missing_patient_input_returns_controlled_error():
    result = entrypoint.invoke({})
    assert result["error_code"] == "invalid_input"


def test_unexpected_exception_in_adapter_is_caught_safely(monkeypatch):
    def _boom(payload):
        raise RuntimeError("simulated unexpected internal failure with secret_token=ABC123")

    monkeypatch.setattr(entrypoint, "handle_request", _boom)
    result = entrypoint.invoke({"patient_input": VALID_PATIENT})

    assert result["explanation_available"] is False
    assert result["error_code"] == "runtime_execution_failed"
    assert "secret_token" not in result["error_message"]
    assert "ABC123" not in result["error_message"]
    assert "Traceback" not in result["error_message"]


def test_response_is_always_json_serializable():
    for payload in ({"patient_input": VALID_PATIENT}, {}, {"patient_input": {"age": 1}}):
        result = entrypoint.invoke(payload)
        json.dumps(result)  # must not raise


def test_no_internal_graph_state_in_error_response():
    result = entrypoint.invoke({})
    for internal_field in ("normalized_input", "retrieved_documents", "generated_explanation"):
        assert internal_field not in result
