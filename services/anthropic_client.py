"""Thin wrapper around the Anthropic Messages API, used as an alternative to
Bedrock for the generation step of /api/explain-risk when AI_PROVIDER=anthropic.

Deliberately mirrors services/bedrock_client.py's public interface
(is_configured, health_check, retrieve, generate_explanation) and reuses its
exception classes -- both are generic RuntimeError subclasses despite the
"Bedrock" name, and reusing them means agent/nodes.py and
services/explanation_service.py need no changes to handle either provider.

This backend does not implement Knowledge Base retrieval -- there is no
Bedrock KB to query. `retrieve()` always raises BedrockConfigurationError,
which is the same signal the graph already treats as "no KB configured":
retrieve_context (agent/nodes.py) skips retrieval and proceeds to generation
without citations, exactly as it does when BEDROCK_KNOWLEDGE_BASE_ID is unset.
"""
import logging
from typing import Any, Dict, List, Optional

import anthropic

from config.settings import Settings, get_settings
from services.bedrock_client import (
    BedrockAuthError,
    BedrockConfigurationError,
    BedrockServiceError,
    BedrockThrottledError,
)

logger = logging.getLogger("cardiorisk.anthropic_client")

_MAX_TOKENS = 1024
_REQUEST_TIMEOUT_SECONDS = 20


class AnthropicClient:
    """Owns the Anthropic SDK client and translates its errors into the same
    domain exceptions services/bedrock_client.py raises. Construct once and
    reuse -- do not create a new instance per request (see get_default_client
    in services/ai_provider.py).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._client = None
        self._init_error: Optional[str] = None

        if self._settings.uses_anthropic() and self._settings.anthropic_api_key:
            self._init_client()

    def _init_client(self) -> None:
        try:
            self._client = anthropic.Anthropic(
                api_key=self._settings.anthropic_api_key,
                timeout=_REQUEST_TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - defensive; SDK construction is not expected to raise
            self._client = None
            self._init_error = str(exc)
            logger.warning("anthropic_client_init_failed error_type=%s", type(exc).__name__)

    def is_configured(self) -> bool:
        """Whether the Anthropic backend is selected AND the SDK client was
        constructed. Does not guarantee the API key is actually valid --
        only that local setup succeeded enough to attempt a call.
        """
        return self._settings.uses_anthropic() and self._client is not None

    def health_check(self) -> Dict[str, Any]:
        """Lightweight, local-only status check -- does NOT call the Anthropic
        API. Safe to call on every GET /api/health request."""
        return {
            "enabled": self._settings.uses_anthropic(),
            "configured": self.is_configured(),
            "reason": None if self.is_configured() else (
                self._init_error or self._settings.missing_configuration_reason()
            ),
        }

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """No Knowledge Base exists on this backend. Always raises so the
        caller's existing "no KB configured" handling applies."""
        raise BedrockConfigurationError("No knowledge base is configured for the Anthropic backend.")

    def generate_explanation(self, system_prompt: str, user_message: str) -> str:
        """Calls the Anthropic Messages API and returns the raw response text.
        JSON parsing/validation of that text is the caller's responsibility
        (services/explanation_service.py).
        """
        if not self.is_configured():
            raise BedrockConfigurationError("Anthropic generation is not configured.")

        try:
            response = self._client.messages.create(
                model=self._settings.anthropic_model_id,
                max_tokens=_MAX_TOKENS,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
        except anthropic.AuthenticationError as exc:
            logger.warning("anthropic_auth_failed")
            raise BedrockAuthError("Invalid or missing Anthropic API key.") from exc
        except anthropic.PermissionDeniedError as exc:
            logger.warning("anthropic_permission_denied")
            raise BedrockAuthError("Anthropic API key lacks required permissions.") from exc
        except anthropic.RateLimitError as exc:
            logger.warning("anthropic_rate_limited")
            raise BedrockThrottledError("Anthropic API rate limit exceeded.") from exc
        except anthropic.APIError as exc:
            logger.warning("anthropic_generation_failed error_type=%s", type(exc).__name__)
            raise BedrockServiceError(f"Explanation generation failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise BedrockServiceError("Anthropic declined to generate a response for this request.")

        for block in response.content:
            if getattr(block, "type", None) == "text":
                return block.text
        raise BedrockServiceError("Anthropic response did not contain any text content.")


_default_client: Optional[AnthropicClient] = None


def get_default_client() -> AnthropicClient:
    """Process-wide singleton so the SDK client is constructed once and
    reused across requests within a warm serverless instance."""
    global _default_client
    if _default_client is None:
        _default_client = AnthropicClient()
    return _default_client
