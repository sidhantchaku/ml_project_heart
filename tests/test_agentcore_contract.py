"""Tests for infrastructure/agentcore/runtime_adapter.py -- the AgentCore
request/response contract. No AWS calls (Bedrock stays disabled by default
in this test environment, same as every other local test module).
"""
import json

import pytest

from api.schemas import ExplanationResponse
from infrastructure.agentcore.runtime_adapter import RuntimeRequestError, handle_request

VALID_PATIENT = {
    "age": 52, "sex": "Male", "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
}


def test_valid_payload_returns_prediction_and_request_id():
    result = handle_request({"patient_input": VALID_PATIENT})
    assert result["prediction"] in (0, 1)
    assert isinstance(result["probability"], float)
    assert result["request_id"]


def test_missing_patient_input_raises_runtime_request_error():
    with pytest.raises(RuntimeRequestError):
        handle_request({})


def test_non_dict_payload_raises_runtime_request_error():
    with pytest.raises(RuntimeRequestError):
        handle_request("not a dict")


def test_patient_input_not_a_dict_raises_runtime_request_error():
    with pytest.raises(RuntimeRequestError):
        handle_request({"patient_input": "nope"})


def test_invalid_patient_input_raises_runtime_request_error():
    with pytest.raises(RuntimeRequestError):
        handle_request({"patient_input": {**VALID_PATIENT, "sex": "Unknown"}})


def test_missing_required_field_raises_runtime_request_error():
    incomplete = {k: v for k, v in VALID_PATIENT.items() if k != "age"}
    with pytest.raises(RuntimeRequestError):
        handle_request({"patient_input": incomplete})


def test_allowed_user_message_proceeds_normally():
    result = handle_request({
        "patient_input": VALID_PATIENT,
        "user_message": "Can you explain what this result means?",
    })
    assert result.get("safety_status") != "blocked"
    assert result["prediction"] in (0, 1)


def test_unsafe_user_message_is_safely_refused_but_prediction_preserved():
    result = handle_request({
        "patient_input": VALID_PATIENT,
        "user_message": "What medication should I take?",
    })
    assert result["safety_status"] == "blocked"
    assert result["safety_category"] == "medication_request"
    assert result["prediction"] in (0, 1)  # prediction still preserved


def test_bedrock_disabled_behaviour_returns_prediction_with_reason():
    result = handle_request({"patient_input": VALID_PATIENT})
    assert result["explanation_available"] is False
    assert result["unavailable_reason"]
    assert result["prediction"] in (0, 1)


def test_response_is_json_serializable():
    result = handle_request({"patient_input": VALID_PATIENT, "user_message": "Explain this."})
    # Must round-trip through json without error -- this is what actually
    # crosses the AgentCore Runtime wire.
    serialized = json.dumps(result)
    deserialized = json.loads(serialized)
    assert deserialized["request_id"] == result["request_id"]


def test_response_matches_explanation_response_schema():
    result = handle_request({"patient_input": VALID_PATIENT})
    # Must validate against the same schema /api/explain-risk uses locally --
    # no second, conflicting response format.
    validated = ExplanationResponse(**result)
    assert validated.prediction in (0, 1)


def test_no_internal_state_exposed_in_response():
    result = handle_request({"patient_input": VALID_PATIENT})
    # Only the final formatted response fields should be present -- not raw
    # LangGraph state like validation_errors, safety_reason, retrieved_documents.
    for internal_field in ("normalized_input", "retrieved_documents", "generated_explanation", "notable_input_factors"):
        assert internal_field not in result


def test_retrieval_k_is_validated_within_schema_bounds():
    with pytest.raises(RuntimeRequestError):
        handle_request({"patient_input": VALID_PATIENT, "retrieval_k": 999})


def test_user_message_length_limit_is_enforced():
    with pytest.raises(RuntimeRequestError):
        handle_request({"patient_input": VALID_PATIENT, "user_message": "x" * 501})
