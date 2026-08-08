"""Thin wrapper around the Google Gemini API, used as an alternative to
Bedrock/Anthropic for the generation step of /api/explain-risk when
AI_PROVIDER=gemini.

Mirrors services/bedrock_client.py's and services/anthropic_client.py's
public interface (is_configured, health_check, retrieve, generate_explanation)
and reuses the same exception classes, so agent/nodes.py and
services/explanation_service.py need no changes to handle any of the three
providers.

Like services/anthropic_client.py, this backend implements generation only --
retrieval is handled separately by the local TF-IDF index (see
tools/local_knowledge_retrieval.py), so `retrieve()` always signals "no
Knowledge Base configured" the same way the Anthropic backend does.
"""
import logging
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from config.settings import Settings, get_settings
from services.bedrock_client import (
    BedrockAuthError,
    BedrockConfigurationError,
    BedrockServiceError,
    BedrockThrottledError,
)

logger = logging.getLogger("cardiorisk.gemini_client")

_MAX_OUTPUT_TOKENS = 1024
_REQUEST_TIMEOUT_MS = 20_000


class GeminiClient:
    """Owns the Gemini SDK client and translates its errors into the same
    domain exceptions services/bedrock_client.py raises. Construct once and
    reuse -- do not create a new instance per request (see
    services/ai_provider.py's singleton).
    """

    def __init__(self, settings: Optional[Settings] = None):
        self._settings = settings or get_settings()
        self._client = None
        self._init_error: Optional[str] = None

        if self._settings.uses_gemini() and self._settings.google_api_key:
            self._init_client()

    def _init_client(self) -> None:
        try:
            self._client = genai.Client(
                api_key=self._settings.google_api_key,
                http_options=genai_types.HttpOptions(timeout=_REQUEST_TIMEOUT_MS),
            )
        except Exception as exc:  # noqa: BLE001 - defensive; SDK construction is not expected to raise
            self._client = None
            self._init_error = str(exc)
            logger.warning("gemini_client_init_failed error_type=%s", type(exc).__name__)

    def is_configured(self) -> bool:
        """Whether the Gemini backend is selected AND the SDK client was
        constructed. Does not guarantee the API key is actually valid --
        only that local setup succeeded enough to attempt a call.
        """
        return self._settings.uses_gemini() and self._client is not None

    def health_check(self) -> Dict[str, Any]:
        """Lightweight, local-only status check -- does NOT call the Gemini
        API. Safe to call on every GET /api/health request."""
        return {
            "enabled": self._settings.uses_gemini(),
            "configured": self.is_configured(),
            "reason": None if self.is_configured() else (
                self._init_error or self._settings.missing_configuration_reason()
            ),
        }

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """No Knowledge Base exists on this backend. Always raises so the
        caller's existing "no KB configured" handling applies."""
        raise BedrockConfigurationError("No knowledge base is configured for the Gemini backend.")

    def generate_explanation(self, system_prompt: str, user_message: str) -> str:
        """Calls the Gemini API and returns the raw response text. JSON
        parsing/validation of that text is the caller's responsibility
        (services/explanation_service.py).
        """
        if not self.is_configured():
            raise BedrockConfigurationError("Gemini generation is not configured.")

        try:
            response = self._client.models.generate_content(
                model=self._settings.gemini_model_id,
                contents=user_message,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=_MAX_OUTPUT_TOKENS,
                    temperature=0.2,
                ),
            )
        except genai_errors.ClientError as exc:
            self._raise_domain_error(exc)
        except genai_errors.ServerError as exc:
            logger.warning("gemini_generation_failed error_type=%s", type(exc).__name__)
            raise BedrockServiceError(f"Explanation generation failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise BedrockServiceError("Gemini response did not contain any text content.")
        return text

    @staticmethod
    def _raise_domain_error(exc: "genai_errors.ClientError") -> None:
        """Translates a genai ClientError into one of our domain exceptions.
        Always raises -- never returns."""
        status_code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        logger.warning("gemini_client_error status_code=%s", status_code)
        if status_code in (401, 403):
            raise BedrockAuthError("Invalid or unauthorized Gemini API key.") from exc
        if status_code == 429:
            raise BedrockThrottledError("Gemini API rate limit exceeded.") from exc
        raise BedrockServiceError(f"Gemini generation failed ({status_code or 'unknown error'}).") from exc


_default_client: Optional[GeminiClient] = None


def get_default_client() -> GeminiClient:
    """Process-wide singleton so the SDK client is constructed once and
    reused across requests within a warm serverless instance."""
    global _default_client
    if _default_client is None:
        _default_client = GeminiClient()
    return _default_client
