"""Thin, reusable wrapper around the two Bedrock boto3 clients used by CardioRisk-AI.

Kept deliberately separate from tools/knowledge_retrieval.py (normalization)
and services/explanation_service.py (orchestration) so retrieval, generation,
and AWS error handling each live in exactly one place. Pure boto3 -- no
langchain-aws.
"""
import logging
from typing import Any, Dict, List, Optional

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError

from config.settings import Settings, get_settings

logger = logging.getLogger("cardiorisk.bedrock_client")

_CONNECT_TIMEOUT_SECONDS = 5
_READ_TIMEOUT_SECONDS = 20
_MAX_RETRY_ATTEMPTS = 2  # bounded retry via botocore's standard mode, not unlimited


class BedrockConfigurationError(RuntimeError):
    """Raised when Bedrock is disabled or under-configured for the requested operation."""


class BedrockAuthError(RuntimeError):
    """Raised when AWS credentials are missing/invalid or access is denied."""


class BedrockThrottledError(RuntimeError):
    """Raised when Bedrock throttles the request."""


class BedrockServiceError(RuntimeError):
    """Raised for any other Bedrock/AWS-side failure (retrieval or generation)."""


def _client_config() -> Config:
    return Config(
        connect_timeout=_CONNECT_TIMEOUT_SECONDS,
        read_timeout=_READ_TIMEOUT_SECONDS,
        retries={"max_attempts": _MAX_RETRY_ATTEMPTS, "mode": "standard"},
    )


class BedrockClient:
    """Owns the two Bedrock boto3 clients and translates AWS errors into
    domain-specific exceptions. Construct once and reuse -- do not create a
    new instance per request (see get_default_client).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._runtime_client = None
        self._agent_runtime_client = None
        self._init_error: Optional[str] = None

        if self._settings.bedrock_enabled:
            self._init_clients()

    def _session_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self._settings.aws_region:
            kwargs["region_name"] = self._settings.aws_region
        # Only pass explicit credentials if BOTH are provided; otherwise let
        # boto3's standard credential chain (env vars, shared config/credentials
        # files, an attached IAM role, SSO, etc.) resolve them.
        if self._settings.aws_access_key_id and self._settings.aws_secret_access_key:
            kwargs["aws_access_key_id"] = self._settings.aws_access_key_id
            kwargs["aws_secret_access_key"] = self._settings.aws_secret_access_key
            if self._settings.aws_session_token:
                kwargs["aws_session_token"] = self._settings.aws_session_token
        return kwargs

    def _init_clients(self) -> None:
        try:
            session = boto3.Session(**self._session_kwargs())
            config = _client_config()
            self._runtime_client = session.client("bedrock-runtime", config=config)
            self._agent_runtime_client = session.client("bedrock-agent-runtime", config=config)
        except (BotoCoreError, NoCredentialsError) as exc:
            self._runtime_client = None
            self._agent_runtime_client = None
            self._init_error = str(exc)
            logger.warning("bedrock_client_init_failed error_type=%s", type(exc).__name__)

    def is_configured(self) -> bool:
        """Whether Bedrock is enabled AND the boto3 clients were constructed.

        Does not guarantee AWS will accept the credentials/permissions at call
        time -- only that local setup succeeded enough to attempt a call.
        """
        return (
            self._settings.bedrock_enabled
            and self._runtime_client is not None
            and self._agent_runtime_client is not None
        )

    def health_check(self) -> Dict[str, Any]:
        """Lightweight, local-only status check -- does NOT call AWS.
        Safe to call on every GET /api/health request."""
        return {
            "enabled": self._settings.bedrock_enabled,
            "configured": self.is_configured(),
            "reason": None if self.is_configured() else (
                self._init_error or self._settings.missing_configuration_reason()
            ),
        }

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Calls bedrock-agent-runtime Retrieve and returns the raw AWS
        `retrievalResults` list. Normalization happens in
        tools/knowledge_retrieval.py, not here.
        """
        if not self.is_configured():
            raise BedrockConfigurationError("Bedrock retrieval is not configured.")
        if not self._settings.bedrock_knowledge_base_id:
            raise BedrockConfigurationError("No Bedrock Knowledge Base ID configured.")

        k = top_k or self._settings.bedrock_retrieval_k
        try:
            response = self._agent_runtime_client.retrieve(
                knowledgeBaseId=self._settings.bedrock_knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": {"numberOfResults": k}
                },
            )
            return response.get("retrievalResults", [])
        except ClientError as exc:
            self._raise_domain_error(exc, "knowledge base retrieval")
        except (BotoCoreError, NoCredentialsError) as exc:
            logger.warning("bedrock_retrieval_failed error_type=%s", type(exc).__name__)
            raise BedrockServiceError(f"Knowledge base retrieval failed: {exc}") from exc

    def generate_explanation(self, system_prompt: str, user_message: str) -> str:
        """Calls bedrock-runtime's Converse API and returns the raw response text.
        JSON parsing/validation of that text is the caller's responsibility
        (services/explanation_service.py).
        """
        if not self.is_configured():
            raise BedrockConfigurationError("Bedrock generation is not configured.")
        if not self._settings.bedrock_model_id:
            raise BedrockConfigurationError("No Bedrock model ID configured.")

        try:
            response = self._runtime_client.converse(
                modelId=self._settings.bedrock_model_id,
                system=[{"text": system_prompt}],
                messages=[{"role": "user", "content": [{"text": user_message}]}],
                inferenceConfig={"temperature": 0.2, "maxTokens": 1024},
            )
            content_blocks = response.get("output", {}).get("message", {}).get("content", [])
            for block in content_blocks:
                if "text" in block:
                    return block["text"]
            raise BedrockServiceError("Bedrock response did not contain any text content.")
        except ClientError as exc:
            self._raise_domain_error(exc, "explanation generation")
        except (BotoCoreError, NoCredentialsError) as exc:
            logger.warning("bedrock_generation_failed error_type=%s", type(exc).__name__)
            raise BedrockServiceError(f"Explanation generation failed: {exc}") from exc

    @staticmethod
    def _raise_domain_error(exc: ClientError, operation: str) -> None:
        """Translates a botocore ClientError into one of our domain exceptions.
        Always raises -- never returns.
        """
        error_code = exc.response.get("Error", {}).get("Code", "")
        logger.warning("bedrock_client_error operation=%s error_code=%s", operation, error_code)
        if error_code in ("AccessDeniedException", "UnauthorizedException"):
            raise BedrockAuthError(f"Access denied during {operation}.") from exc
        if error_code in ("ExpiredTokenException", "UnrecognizedClientException"):
            raise BedrockAuthError(f"Invalid or expired AWS credentials during {operation}.") from exc
        if error_code == "ThrottlingException":
            raise BedrockThrottledError(f"Bedrock throttled the request during {operation}.") from exc
        raise BedrockServiceError(f"Bedrock {operation} failed ({error_code or 'unknown error'}).") from exc


_default_client: Optional[BedrockClient] = None


def get_default_client() -> BedrockClient:
    """Process-wide singleton so boto3 clients are constructed once and
    reused across requests within a warm serverless instance."""
    global _default_client
    if _default_client is None:
        _default_client = BedrockClient()
    return _default_client
