"""Tests for services/explanation_service.py. Bedrock/AWS calls are mocked."""
import json
from unittest.mock import MagicMock

import pytest

from config.settings import Settings
from services.bedrock_client import BedrockServiceError, BedrockThrottledError
from services.explanation_service import ExplanationService
from tools.knowledge_retrieval import RetrievedChunk

VALID_NORMALIZED_INPUT = {
    "age": 63,
    "sex": 1,
    "cp": 0,
    "trestbps": 150,
    "chol": 260,
    "fbs": 1,
    "restecg": 1,
    "thalach": 90,
    "exang": 1,
    "oldpeak": 2.5,
    "slope": 2,
    "ca": 2,
    "thal": 3,
}

SAMPLE_CHUNKS = [
    RetrievedChunk("Regular exercise supports heart health.", "https://cdc.gov/heart", 0.9, {}),
    RetrievedChunk("A balanced diet lowers cardiovascular risk factors.", "https://heart.org/diet", 0.8, {}),
]

VALID_MODEL_JSON = {
    "summary": "Educational summary only.",
    "risk_category": "Higher modeled risk of heart disease",
    "probability": 0.71,
    "input_factors": ["Elevated blood pressure"],
    "educational_information": ["Regular exercise supports heart health. [source_1]"],
    "questions_for_professional": ["What lifestyle changes might help?"],
    "citations": [{"id": "source_1", "title": "CDC", "uri": "https://cdc.gov/heart"}],
    "disclaimer": "Educational estimate only, not a diagnosis.",
}


def _disabled_settings() -> Settings:
    return Settings(BEDROCK_ENABLED=False)


def _enabled_settings() -> Settings:
    return Settings(
        BEDROCK_ENABLED=True,
        AWS_REGION="us-east-1",
        BEDROCK_MODEL_ID="test-model",
        BEDROCK_KNOWLEDGE_BASE_ID="test-kb",
        BEDROCK_RETRIEVAL_K=2,
    )


def _service(settings, retrieve_return=None, retrieve_side_effect=None,
             generate_return=None, generate_side_effect=None):
    mock_client = MagicMock()
    mock_client.is_configured.return_value = settings.bedrock_enabled
    mock_retrieval_tool = MagicMock()
    if retrieve_side_effect is not None:
        mock_retrieval_tool.retrieve.side_effect = retrieve_side_effect
    else:
        mock_retrieval_tool.retrieve.return_value = retrieve_return or []
    if generate_side_effect is not None:
        mock_client.generate_explanation.side_effect = generate_side_effect
    else:
        mock_client.generate_explanation.return_value = generate_return or json.dumps(VALID_MODEL_JSON)

    return ExplanationService(
        settings=settings,
        bedrock_client=mock_client,
        retrieval_tool=mock_retrieval_tool,
    )


def test_explanation_unavailable_when_bedrock_disabled():
    service = _service(_disabled_settings())
    result = service.explain(prediction=1, probability=0.7, normalized_input=VALID_NORMALIZED_INPUT)
    assert result.explanation_available is False
    assert result.unavailable_reason is not None
    assert result.risk_category == "Higher modeled risk of heart disease"
    # Prediction context must still be present even without Bedrock.
    assert result.probability == 0.7
    assert len(result.input_factors) > 0


def test_explanation_unavailable_when_retrieval_fails():
    service = _service(_enabled_settings(), retrieve_side_effect=BedrockServiceError("down"))
    result = service.explain(prediction=0, probability=0.2, normalized_input=VALID_NORMALIZED_INPUT)
    assert result.explanation_available is False
    assert "retrieve" in result.unavailable_reason.lower()


def test_explanation_unavailable_when_generation_fails():
    service = _service(
        _enabled_settings(), retrieve_return=SAMPLE_CHUNKS,
        generate_side_effect=BedrockThrottledError("throttled"),
    )
    result = service.explain(prediction=1, probability=0.6, normalized_input=VALID_NORMALIZED_INPUT)
    assert result.explanation_available is False
    assert "generate" in result.unavailable_reason.lower()


def test_explanation_unavailable_on_empty_retrieval_still_attempts_generation():
    service = _service(_enabled_settings(), retrieve_return=[])
    result = service.explain(prediction=1, probability=0.6, normalized_input=VALID_NORMALIZED_INPUT)
    # Generation still runs (mock returns VALID_MODEL_JSON), but citations must be empty
    # since no chunks were retrieved -- nothing to cite.
    assert result.explanation_available is True
    assert result.citations == []


def test_successful_explanation_returns_structured_result():
    service = _service(_enabled_settings(), retrieve_return=SAMPLE_CHUNKS)
    result = service.explain(prediction=1, probability=0.71, normalized_input=VALID_NORMALIZED_INPUT)

    assert result.explanation_available is True
    assert result.summary == "Educational summary only."
    # Probability/risk_category always come from the real ML output, not the model's JSON.
    assert result.probability == 0.71
    assert result.risk_category == "Higher modeled risk of heart disease"
    assert len(result.citations) == 1
    assert result.citations[0]["id"] == "source_1"
    assert result.citations[0]["uri"] == "https://cdc.gov/heart"
    assert result.disclaimer == "Educational estimate only, not a diagnosis."


def test_invalid_json_returns_unavailable_not_raise():
    service = _service(_enabled_settings(), retrieve_return=SAMPLE_CHUNKS, generate_return="not json at all {{{")
    result = service.explain(prediction=1, probability=0.5, normalized_input=VALID_NORMALIZED_INPUT)
    assert result.explanation_available is False
    assert "unusable" in result.unavailable_reason.lower()


def test_json_repair_extracts_embedded_object():
    noisy = 'Sure, here is the JSON:\n' + json.dumps(VALID_MODEL_JSON) + '\nLet me know if you need more.'
    service = _service(_enabled_settings(), retrieve_return=SAMPLE_CHUNKS, generate_return=noisy)
    result = service.explain(prediction=1, probability=0.7, normalized_input=VALID_NORMALIZED_INPUT)
    assert result.explanation_available is True
    assert result.summary == "Educational summary only."


def test_missing_required_json_field_is_treated_as_invalid():
    incomplete = {k: v for k, v in VALID_MODEL_JSON.items() if k != "disclaimer"}
    service = _service(_enabled_settings(), retrieve_return=SAMPLE_CHUNKS, generate_return=json.dumps(incomplete))
    result = service.explain(prediction=1, probability=0.7, normalized_input=VALID_NORMALIZED_INPUT)
    assert result.explanation_available is False


def test_fabricated_citation_id_is_dropped():
    payload = {**VALID_MODEL_JSON, "citations": [
        {"id": "source_1", "title": "CDC", "uri": "https://cdc.gov/heart"},
        {"id": "source_99", "title": "Made up", "uri": "https://fake.example/made-up"},
    ]}
    service = _service(_enabled_settings(), retrieve_return=SAMPLE_CHUNKS, generate_return=json.dumps(payload))
    result = service.explain(prediction=1, probability=0.7, normalized_input=VALID_NORMALIZED_INPUT)
    assert len(result.citations) == 1
    assert result.citations[0]["id"] == "source_1"


def test_duplicate_citation_ids_are_deduplicated():
    payload = {**VALID_MODEL_JSON, "citations": [
        {"id": "source_1", "title": "CDC", "uri": "https://cdc.gov/heart"},
        {"id": "source_1", "title": "CDC again", "uri": "https://cdc.gov/heart"},
    ]}
    service = _service(_enabled_settings(), retrieve_return=SAMPLE_CHUNKS, generate_return=json.dumps(payload))
    result = service.explain(prediction=1, probability=0.7, normalized_input=VALID_NORMALIZED_INPUT)
    assert len(result.citations) == 1


def test_deterministic_input_factors_flag_notable_values():
    service = _service(_disabled_settings())
    result = service.explain(prediction=1, probability=0.7, normalized_input=VALID_NORMALIZED_INPUT)
    joined = " ".join(result.input_factors)
    assert "blood pressure" in joined.lower()
    assert "cholesterol" in joined.lower()
    assert "angina" in joined.lower()


def test_no_notable_factors_returns_clear_message():
    boring_input = {**VALID_NORMALIZED_INPUT, "age": 30, "trestbps": 110, "chol": 180,
                    "fbs": 0, "exang": 0, "oldpeak": 0.5, "ca": 0, "thalach": 160}
    service = _service(_disabled_settings())
    result = service.explain(prediction=0, probability=0.1, normalized_input=boring_input)
    assert any("no submitted values" in factor.lower() for factor in result.input_factors)
