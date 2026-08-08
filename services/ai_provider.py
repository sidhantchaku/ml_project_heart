"""Selects the generation backend for /api/explain-risk based on AI_PROVIDER.

Both services/bedrock_client.BedrockClient and services/anthropic_client.AnthropicClient
expose the same interface (is_configured, health_check, retrieve, generate_explanation)
and raise the same exception classes, so services/explanation_service.py and
agent/nodes.py work unchanged regardless of which one this returns.
"""
from typing import Optional, Union

from config.settings import Settings, get_settings
from services.anthropic_client import AnthropicClient
from services.anthropic_client import get_default_client as get_default_anthropic_client
from services.bedrock_client import BedrockClient
from services.bedrock_client import get_default_client as get_default_bedrock_client
from services.gemini_client import GeminiClient
from services.gemini_client import get_default_client as get_default_gemini_client

AIClient = Union[BedrockClient, AnthropicClient, GeminiClient]

_default_client: Optional[AIClient] = None


def get_default_client(settings: Optional[Settings] = None) -> AIClient:
    """Process-wide singleton, selected once per warm instance by AI_PROVIDER."""
    global _default_client
    if _default_client is None:
        settings = settings or get_settings()
        if settings.uses_anthropic():
            _default_client = get_default_anthropic_client()
        elif settings.uses_gemini():
            _default_client = get_default_gemini_client()
        else:
            _default_client = get_default_bedrock_client()
    return _default_client
