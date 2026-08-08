"""Server-side client for invoking a deployed Amazon Bedrock AgentCore Runtime.

Used only by api/index.py (server-side FastAPI code) when USE_AGENTCORE=true.
Never imported by, or reachable from, public/ (frontend JavaScript) -- boto3
and any AWS credentials stay entirely server-side. Mirrors the client-reuse /
timeout / error-translation pattern already established in
services/bedrock_client.py.
"""
import json
import logging
import uuid
from typing import Any, Dict, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from config.settings import Settings, get_settings

logger = logging.getLogger("cardiorisk.agentcore_client")

_CONNECT_TIMEOUT_SECONDS = 5
_MAX_SESSION_ID_LENGTH = 100


class AgentCoreConfigurationError(RuntimeError):
    """Raised when AgentCore routing is disabled or under-configured."""


class AgentCoreAuthError(RuntimeError):
    """Raised when AWS credentials/permissions are invalid for this call."""


class AgentCoreUnavailableError(RuntimeError):
    """Raised when the runtime is unreachable, throttled, not found, or times out."""


class AgentCoreResponseError(RuntimeError):
    """Raised when the runtime returned a response that couldn't be parsed
    into the expected JSON shape."""


def generate_session_id() -> str:
    """A random, non-identifying session id.

    Never derived from patient data, an email address, or any other
    user-identifying string -- just a random token used to group requests
    within a single runtime session.
    """
    return uuid.uuid4().hex


def _validate_session_id(session_id: Optional[str]) -> str:
    if not session_id or not session_id.strip():
        return generate_session_id()
    if len(session_id) > _MAX_SESSION_ID_LENGTH:
        raise ValueError(f"session_id must be at most {_MAX_SESSION_ID_LENGTH} characters.")
    return session_id


class AgentCoreClient:
    """Owns the boto3 `bedrock-agentcore` client and translates AWS errors
    into domain-specific exceptions. Construct once and reuse (see
    get_default_client) -- do not create a new instance per request."""

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._client = None
        self._init_error: Optional[str] = None
        if self._settings.use_agentcore:
            self._init_client()

    def _init_client(self) -> None:
        try:
            config = Config(
                connect_timeout=_CONNECT_TIMEOUT_SECONDS,
                read_timeout=self._settings.agentcore_request_timeout_seconds,
                retries={"max_attempts": 2, "mode": "standard"},
            )
            kwargs: Dict[str, Any] = {}
            if self._settings.aws_region:
                kwargs["region_name"] = self._settings.aws_region
            self._client = boto3.client("bedrock-agentcore", config=config, **kwargs)
        except (BotoCoreError, NoCredentialsError) as exc:
            self._client = None
            self._init_error = str(exc)
            logger.warning("agentcore_client_init_failed error_type=%s", type(exc).__name__)

    def is_configured(self) -> bool:
        """Whether AgentCore routing is enabled, a runtime ARN is set, and
        the boto3 client was constructed. Does not verify the runtime is
        actually reachable or that IAM permissions are sufficient -- those
        surface at call time in invoke()."""
        return bool(
            self._settings.use_agentcore
            and self._settings.agentcore_runtime_arn
            and self._client is not None
        )

    def health_check(self) -> Dict[str, Any]:
        """Lightweight, local-only status -- does NOT call AWS."""
        return {
            "enabled": self._settings.use_agentcore,
            "configured": self.is_configured(),
            "runtime_arn_present": bool(self._settings.agentcore_runtime_arn),
        }

    def invoke(
        self,
        patient_input: Dict[str, Any],
        user_message: Optional[str] = None,
        retrieval_k: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Invokes the configured AgentCore Runtime and returns its parsed
        JSON response (the same final_response shape agent.graph produces).

        Raises AgentCoreConfigurationError / AgentCoreAuthError /
        AgentCoreUnavailableError / AgentCoreResponseError -- never leaks raw
        AWS error internals, ARNs, or credentials to callers.
        """
        if not self.is_configured():
            raise AgentCoreConfigurationError("AgentCore routing is not configured.")

        resolved_session_id = _validate_session_id(session_id)
        payload: Dict[str, Any] = {"patient_input": patient_input}
        if user_message:
            payload["user_message"] = user_message
        if retrieval_k is not None:
            payload["retrieval_k"] = retrieval_k

        call_kwargs: Dict[str, Any] = {
            "agentRuntimeArn": self._settings.agentcore_runtime_arn,
            "runtimeSessionId": resolved_session_id,
            "payload": json.dumps(payload).encode("utf-8"),
            "contentType": "application/json",
        }
        if self._settings.agentcore_runtime_qualifier:
            call_kwargs["qualifier"] = self._settings.agentcore_runtime_qualifier

        try:
            response = self._client.invoke_agent_runtime(**call_kwargs)
        except ClientError as exc:
            self._raise_domain_error(exc)
        except (BotoCoreError, NoCredentialsError) as exc:
            logger.warning("agentcore_invoke_failed error_type=%s", type(exc).__name__)
            raise AgentCoreUnavailableError(f"AgentCore invocation failed: {type(exc).__name__}") from exc

        try:
            body = response["response"]
            raw_bytes = body.read() if hasattr(body, "read") else body
            text = raw_bytes.decode("utf-8") if isinstance(raw_bytes, bytes) else raw_bytes
            return json.loads(text)
        except (KeyError, AttributeError, UnicodeDecodeError) as exc:
            raise AgentCoreResponseError(f"AgentCore response was not in the expected shape: {type(exc).__name__}") from exc
        except json.JSONDecodeError as exc:
            raise AgentCoreResponseError(f"AgentCore response was not valid JSON: {exc}") from exc

    @staticmethod
    def _raise_domain_error(exc: ClientError) -> None:
        error_code = exc.response.get("Error", {}).get("Code", "")
        logger.warning("agentcore_client_error error_code=%s", error_code)
        if error_code in ("AccessDeniedException", "UnauthorizedException"):
            raise AgentCoreAuthError("Access denied while invoking AgentCore Runtime.") from exc
        if error_code in ("ExpiredTokenException", "UnrecognizedClientException"):
            raise AgentCoreAuthError("Invalid or expired AWS credentials while invoking AgentCore Runtime.") from exc
        if error_code in ("ThrottlingException", "ServiceQuotaExceededException"):
            raise AgentCoreUnavailableError("AgentCore Runtime is currently throttled.") from exc
        if error_code in ("ResourceNotFoundException",):
            raise AgentCoreUnavailableError("AgentCore Runtime was not found.") from exc
        raise AgentCoreUnavailableError(f"AgentCore invocation failed ({error_code or 'unknown error'}).") from exc


_default_client: Optional[AgentCoreClient] = None


def get_default_client() -> AgentCoreClient:
    """Process-wide singleton so the boto3 client is constructed once and
    reused across requests within a warm serverless instance."""
    global _default_client
    if _default_client is None:
        _default_client = AgentCoreClient()
    return _default_client
