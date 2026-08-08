"""Tests for services/bedrock_client.py. All AWS calls are mocked -- no real
AWS credentials or network access are used.
"""
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from config.settings import Settings
from services.bedrock_client import (
    BedrockAuthError,
    BedrockClient,
    BedrockConfigurationError,
    BedrockServiceError,
    BedrockThrottledError,
)


def _disabled_settings() -> Settings:
    return Settings(BEDROCK_ENABLED=False)


def _enabled_settings(**overrides) -> Settings:
    defaults = dict(
        BEDROCK_ENABLED=True,
        AWS_REGION="us-east-1",
        BEDROCK_MODEL_ID="test-model-id",
        BEDROCK_KNOWLEDGE_BASE_ID="test-kb-id",
        BEDROCK_RETRIEVAL_K=3,
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "SomeOperation")


def test_disabled_bedrock_never_constructs_clients():
    client = BedrockClient(settings=_disabled_settings())
    assert client.is_configured() is False
    health = client.health_check()
    assert health["enabled"] is False
    assert health["configured"] is False


def test_enabled_bedrock_constructs_clients(monkeypatch):
    client = BedrockClient(settings=_enabled_settings())
    assert client.is_configured() is True
    assert client.health_check()["configured"] is True


def test_missing_configuration_reason_when_missing_model_id():
    settings = _enabled_settings(BEDROCK_MODEL_ID=None)
    reason = settings.missing_configuration_reason()
    assert reason is not None
    assert "model" in reason.lower()


def test_retrieve_raises_configuration_error_when_disabled():
    client = BedrockClient(settings=_disabled_settings())
    with pytest.raises(BedrockConfigurationError):
        client.retrieve("query")


def test_retrieve_returns_raw_results_on_success():
    client = BedrockClient(settings=_enabled_settings())
    client._agent_runtime_client = MagicMock()
    client._agent_runtime_client.retrieve.return_value = {
        "retrievalResults": [{"content": {"text": "hello"}}]
    }
    results = client.retrieve("query")
    assert results == [{"content": {"text": "hello"}}]
    client._agent_runtime_client.retrieve.assert_called_once()


def test_retrieve_translates_access_denied():
    client = BedrockClient(settings=_enabled_settings())
    client._agent_runtime_client = MagicMock()
    client._agent_runtime_client.retrieve.side_effect = _client_error("AccessDeniedException")
    with pytest.raises(BedrockAuthError):
        client.retrieve("query")


def test_retrieve_translates_throttling():
    client = BedrockClient(settings=_enabled_settings())
    client._agent_runtime_client = MagicMock()
    client._agent_runtime_client.retrieve.side_effect = _client_error("ThrottlingException")
    with pytest.raises(BedrockThrottledError):
        client.retrieve("query")


def test_retrieve_translates_unknown_client_error_to_service_error():
    client = BedrockClient(settings=_enabled_settings())
    client._agent_runtime_client = MagicMock()
    client._agent_runtime_client.retrieve.side_effect = _client_error("ResourceNotFoundException")
    with pytest.raises(BedrockServiceError):
        client.retrieve("query")


def test_generate_explanation_returns_text_on_success():
    client = BedrockClient(settings=_enabled_settings())
    client._runtime_client = MagicMock()
    client._runtime_client.converse.return_value = {
        "output": {"message": {"content": [{"text": '{"summary": "ok"}'}]}}
    }
    text = client.generate_explanation("system prompt", "user message")
    assert text == '{"summary": "ok"}'


def test_generate_explanation_raises_when_no_text_content():
    client = BedrockClient(settings=_enabled_settings())
    client._runtime_client = MagicMock()
    client._runtime_client.converse.return_value = {"output": {"message": {"content": []}}}
    with pytest.raises(BedrockServiceError):
        client.generate_explanation("system prompt", "user message")


def test_generate_explanation_translates_client_error():
    client = BedrockClient(settings=_enabled_settings())
    client._runtime_client = MagicMock()
    client._runtime_client.converse.side_effect = _client_error("AccessDeniedException")
    with pytest.raises(BedrockAuthError):
        client.generate_explanation("system prompt", "user message")


def test_health_check_never_calls_aws():
    client = BedrockClient(settings=_enabled_settings())
    client._runtime_client = MagicMock()
    client._agent_runtime_client = MagicMock()
    client.health_check()
    client._runtime_client.assert_not_called()
    client._agent_runtime_client.assert_not_called()


def test_client_errors_do_not_leak_credentials_in_message(monkeypatch):
    """Guards against accidentally embedding secrets in a translated error message."""
    client = BedrockClient(settings=_enabled_settings(AWS_SECRET_ACCESS_KEY="super-secret-value"))
    client._agent_runtime_client = MagicMock()
    client._agent_runtime_client.retrieve.side_effect = _client_error("AccessDeniedException")
    with pytest.raises(BedrockAuthError) as exc_info:
        client.retrieve("query")
    assert "super-secret-value" not in str(exc_info.value)
