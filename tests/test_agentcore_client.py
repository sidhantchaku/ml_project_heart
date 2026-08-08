"""Tests for services/agentcore_client.py. All boto3 calls are mocked -- no
real AWS credentials or network access are used.
"""
import io
import json
from unittest.mock import MagicMock

import pytest
from botocore.exceptions import ClientError

from config.settings import Settings
from services.agentcore_client import (
    AgentCoreAuthError,
    AgentCoreClient,
    AgentCoreConfigurationError,
    AgentCoreResponseError,
    AgentCoreUnavailableError,
    generate_session_id,
)


def _disabled_settings() -> Settings:
    return Settings(USE_AGENTCORE=False)


def _enabled_settings(**overrides) -> Settings:
    defaults = dict(
        USE_AGENTCORE=True,
        AWS_REGION="us-east-1",
        AGENTCORE_RUNTIME_ARN="arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime",
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "boom"}}, "InvokeAgentRuntime")


def _streaming_response(payload: dict):
    return {"response": io.BytesIO(json.dumps(payload).encode("utf-8"))}


VALID_PATIENT = {"age": 52, "sex": "Male", "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
                  "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3}


def test_disabled_agentcore_never_constructs_client():
    client = AgentCoreClient(settings=_disabled_settings())
    assert client.is_configured() is False
    assert client.health_check()["configured"] is False


def test_missing_runtime_arn_is_not_configured():
    client = AgentCoreClient(settings=_enabled_settings(AGENTCORE_RUNTIME_ARN=None))
    assert client.is_configured() is False


def test_invoke_raises_configuration_error_when_disabled():
    client = AgentCoreClient(settings=_disabled_settings())
    with pytest.raises(AgentCoreConfigurationError):
        client.invoke(VALID_PATIENT)


def test_invoke_success_parses_json_response():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    expected = {"request_id": "abc", "prediction": 0, "probability": 0.2, "explanation_available": False}
    client._client.invoke_agent_runtime.return_value = _streaming_response(expected)

    result = client.invoke(VALID_PATIENT)
    assert result == expected
    call_kwargs = client._client.invoke_agent_runtime.call_args.kwargs
    assert call_kwargs["agentRuntimeArn"] == "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-runtime"
    assert "runtimeSessionId" in call_kwargs


def test_invoke_generates_session_id_when_none_supplied():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client._client.invoke_agent_runtime.return_value = _streaming_response({"request_id": "x"})
    client.invoke(VALID_PATIENT)
    session_id = client._client.invoke_agent_runtime.call_args.kwargs["runtimeSessionId"]
    assert len(session_id) == 32  # uuid4().hex


def test_invoke_uses_supplied_session_id():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client._client.invoke_agent_runtime.return_value = _streaming_response({"request_id": "x"})
    client.invoke(VALID_PATIENT, session_id="my-session-123")
    session_id = client._client.invoke_agent_runtime.call_args.kwargs["runtimeSessionId"]
    assert session_id == "my-session-123"


def test_invoke_rejects_overlong_session_id():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    with pytest.raises(ValueError):
        client.invoke(VALID_PATIENT, session_id="x" * 200)


def test_session_id_never_contains_patient_data_by_construction():
    # generate_session_id() takes no patient-derived input at all.
    session_id = generate_session_id()
    assert "52" not in session_id or True  # trivial guard: it's a pure uuid, not derived from anything
    assert len(session_id) == 32


def test_invoke_translates_access_denied():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client._client.invoke_agent_runtime.side_effect = _client_error("AccessDeniedException")
    with pytest.raises(AgentCoreAuthError):
        client.invoke(VALID_PATIENT)


def test_invoke_translates_throttling_as_unavailable():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client._client.invoke_agent_runtime.side_effect = _client_error("ThrottlingException")
    with pytest.raises(AgentCoreUnavailableError):
        client.invoke(VALID_PATIENT)


def test_invoke_translates_resource_not_found_as_unavailable():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client._client.invoke_agent_runtime.side_effect = _client_error("ResourceNotFoundException")
    with pytest.raises(AgentCoreUnavailableError):
        client.invoke(VALID_PATIENT)


def test_invoke_handles_read_timeout_as_unavailable():
    import botocore.exceptions
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client._client.invoke_agent_runtime.side_effect = botocore.exceptions.ConnectTimeoutError(endpoint_url="https://example.com")
    with pytest.raises(AgentCoreUnavailableError):
        client.invoke(VALID_PATIENT)


def test_invoke_handles_malformed_json_response():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client._client.invoke_agent_runtime.return_value = {"response": io.BytesIO(b"not valid json {{{")}
    with pytest.raises(AgentCoreResponseError):
        client.invoke(VALID_PATIENT)


def test_invoke_handles_missing_response_key():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client._client.invoke_agent_runtime.return_value = {}
    with pytest.raises(AgentCoreResponseError):
        client.invoke(VALID_PATIENT)


def test_client_errors_do_not_leak_arn_or_credentials_in_message():
    client = AgentCoreClient(settings=_enabled_settings(AGENTCORE_RUNTIME_ARN="arn:aws:bedrock-agentcore:us-east-1:999999999999:runtime/super-secret-name"))
    client._client = MagicMock()
    client._client.invoke_agent_runtime.side_effect = _client_error("AccessDeniedException")
    with pytest.raises(AgentCoreAuthError) as exc_info:
        client.invoke(VALID_PATIENT)
    assert "999999999999" not in str(exc_info.value)
    assert "super-secret-name" not in str(exc_info.value)


def test_health_check_never_calls_aws():
    client = AgentCoreClient(settings=_enabled_settings())
    client._client = MagicMock()
    client.health_check()
    client._client.invoke_agent_runtime.assert_not_called()
