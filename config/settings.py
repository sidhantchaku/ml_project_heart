"""Environment-driven configuration for the optional Bedrock explanation layer.

Nothing in this module hardcodes credentials, ARNs, account IDs, model IDs, or
Knowledge Base IDs -- every AWS-specific value comes from an environment
variable and is allowed to be absent. `BEDROCK_ENABLED` defaults to false, so
the application is fully functional (prediction-only) with zero configuration.

AWS credentials are NOT required here: boto3's standard credential chain
(environment variables, shared config/credentials files, an attached IAM role,
SSO, etc.) is used by default. The explicit AWS_ACCESS_KEY_ID / SECRET /
SESSION_TOKEN variables below are supported only for cases where an operator
wants to pin credentials via environment variables -- they are optional and
boto3 is given the chance to resolve credentials on its own if they're absent.
"""
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Bedrock/AWS configuration, sourced entirely from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Master switch. Must be explicitly enabled -- absent/false means the
    # explanation layer is skipped entirely and /api/predict is unaffected.
    bedrock_enabled: bool = Field(default=False, alias="BEDROCK_ENABLED")

    aws_region: Optional[str] = Field(default=None, alias="AWS_REGION")

    # Optional explicit credentials. Prefer the default boto3 credential chain;
    # only set these if your deployment environment requires it.
    aws_access_key_id: Optional[str] = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")
    aws_session_token: Optional[str] = Field(default=None, alias="AWS_SESSION_TOKEN")

    bedrock_model_id: Optional[str] = Field(default=None, alias="BEDROCK_MODEL_ID")
    bedrock_knowledge_base_id: Optional[str] = Field(default=None, alias="BEDROCK_KNOWLEDGE_BASE_ID")
    bedrock_retrieval_k: int = Field(default=3, alias="BEDROCK_RETRIEVAL_K", ge=1, le=10)

    # --- Direct Anthropic / Gemini API generation backends ---
    # Alternatives to Bedrock for the generation step only (retrieval is
    # handled separately by the local TF-IDF knowledge index -- see
    # tools/local_knowledge_retrieval.py -- regardless of which generation
    # backend is selected here). AI_PROVIDER picks exactly one backend so
    # health/config fields never mix providers.
    ai_provider: str = Field(default="bedrock", alias="AI_PROVIDER")
    anthropic_api_key: Optional[str] = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_model_id: str = Field(default="claude-sonnet-5", alias="ANTHROPIC_MODEL_ID")
    # Named GOOGLE_API_KEY (Google's own SDK convention) rather than
    # GEMINI_API_KEY, matching how this key is actually provisioned/labeled.
    google_api_key: Optional[str] = Field(default=None, alias="GOOGLE_API_KEY")
    gemini_model_id: str = Field(default="gemini-2.5-flash", alias="GEMINI_MODEL_ID")

    # --- AgentCore Runtime (Phase 6) ---
    # Master switch, mirrors bedrock_enabled's spirit: absent/false means
    # /api/explain-risk keeps using the local LangGraph path (Phases 3-5),
    # completely unaffected by anything below.
    use_agentcore: bool = Field(default=False, alias="USE_AGENTCORE")
    agentcore_runtime_arn: Optional[str] = Field(default=None, alias="AGENTCORE_RUNTIME_ARN")
    agentcore_runtime_qualifier: Optional[str] = Field(default=None, alias="AGENTCORE_RUNTIME_QUALIFIER")
    agentcore_request_timeout_seconds: int = Field(default=30, alias="AGENTCORE_REQUEST_TIMEOUT_SECONDS", ge=1, le=120)

    @field_validator(
        "aws_region", "aws_access_key_id", "aws_secret_access_key",
        "aws_session_token", "bedrock_model_id", "bedrock_knowledge_base_id",
        "agentcore_runtime_arn", "agentcore_runtime_qualifier", "anthropic_api_key",
        "google_api_key",
        mode="before",
    )
    @classmethod
    def _blank_string_to_none(cls, value):
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("ai_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value):
        if isinstance(value, str):
            return value.strip().lower() or "bedrock"
        return value

    def uses_anthropic(self) -> bool:
        return self.ai_provider == "anthropic"

    def uses_gemini(self) -> bool:
        return self.ai_provider == "gemini"

    def is_anthropic_ready(self) -> bool:
        """Whether enough configuration exists to attempt a direct Anthropic
        API call. Does not verify the key is valid -- only that it's present."""
        return bool(self.uses_anthropic() and self.anthropic_api_key)

    def is_gemini_ready(self) -> bool:
        """Whether enough configuration exists to attempt a direct Gemini
        API call. Does not verify the key is valid -- only that it's present."""
        return bool(self.uses_gemini() and self.google_api_key)

    def is_bedrock_ready(self) -> bool:
        """Whether enough configuration exists to attempt a Bedrock call.

        This does NOT verify credentials or network reachability -- only that
        the feature flag is on and the two required identifiers are present.
        Actual AWS auth/permission problems surface at call time in
        services/bedrock_client.py.
        """
        return bool(
            self.bedrock_enabled
            and self.bedrock_model_id
            and self.bedrock_knowledge_base_id
        )

    def is_agentcore_ready(self) -> bool:
        """Whether enough configuration exists to attempt an AgentCore Runtime
        invocation. Does not verify credentials, IAM permissions, or that the
        runtime is actually reachable -- only that the flag and ARN are set."""
        return bool(self.use_agentcore and self.agentcore_runtime_arn)

    def missing_agentcore_configuration_reason(self) -> Optional[str]:
        if not self.use_agentcore:
            return "AgentCore routing is not enabled on this deployment."
        if not self.agentcore_runtime_arn:
            return "AgentCore routing is enabled but no runtime ARN is configured."
        return None

    def missing_configuration_reason(self) -> Optional[str]:
        """A short, non-technical reason the explanation feature is unavailable,
        suitable for returning to API consumers. Returns None if configuration
        looks sufficient (does not guarantee the provider will actually accept it)."""
        if self.uses_anthropic():
            if not self.anthropic_api_key:
                return "The AI explanation feature is not fully configured (missing Anthropic API key)."
            return None

        if self.uses_gemini():
            if not self.google_api_key:
                return "The AI explanation feature is not fully configured (missing Gemini API key)."
            return None

        if not self.bedrock_enabled:
            return "The AI explanation feature is not enabled on this deployment."
        if not self.bedrock_model_id:
            return "The AI explanation feature is not fully configured (missing model configuration)."
        if not self.bedrock_knowledge_base_id:
            return "The AI explanation feature is not fully configured (missing knowledge base configuration)."
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings instance (env vars are read once per instance)."""
    return Settings()
