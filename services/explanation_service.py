"""Orchestrates the optional Bedrock-grounded explanation: existing ML prediction
+ Knowledge Base retrieval + foundation model generation + citation/safety
formatting.

This module does not touch AWS directly (see services/bedrock_client.py and
tools/knowledge_retrieval.py) and does not run the scikit-learn model itself
(see tools/risk_prediction.py) -- it wires the pieces together and is
responsible for making sure nothing unsafe or fabricated reaches the API layer.
"""
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config.settings import Settings, get_settings
from services.ai_provider import AIClient, get_default_client
from services.bedrock_client import (
    BedrockAuthError,
    BedrockConfigurationError,
    BedrockServiceError,
    BedrockThrottledError,
)
from services.prompts import SYSTEM_PROMPT, build_user_message
from tools.knowledge_retrieval import KnowledgeRetrievalTool, RetrievedChunk
from tools.local_knowledge_retrieval import LocalKnowledgeRetrievalTool, is_index_available

logger = logging.getLogger("cardiorisk.explanation_service")

DISCLAIMER = (
    "This is an educational, machine-learning-based risk estimate only. It is not a medical "
    "diagnosis, is not clinically validated, and does not replace evaluation by a qualified "
    "healthcare professional. If you are experiencing chest pain, difficulty breathing, or "
    "other symptoms that feel urgent, contact your local emergency services or a qualified "
    "healthcare professional right away."
)

REQUIRED_JSON_FIELDS = {
    "summary", "risk_category", "probability", "input_factors",
    "educational_information", "questions_for_professional", "citations", "disclaimer",
}


class ExplanationServiceError(RuntimeError):
    """Raised when a grounded explanation cannot be produced at all (caller
    should fall back to a prediction-only response, not surface this raw)."""


@dataclass
class ExplanationResult:
    explanation_available: bool
    summary: str = ""
    risk_category: str = ""
    probability: Optional[float] = None
    input_factors: List[str] = field(default_factory=list)
    educational_information: List[str] = field(default_factory=list)
    questions_for_professional: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)
    disclaimer: str = DISCLAIMER
    unavailable_reason: Optional[str] = None


# --- Deterministic, rule-based input factors --------------------------------
# These are observations about the submitted values, not causal claims about
# heart disease risk. Thresholds are illustrative/educational, not clinical
# cutoffs, and are documented here so they are easy to audit.

def rule_based_input_factors(normalized_input: Dict[str, Any]) -> List[str]:
    factors: List[str] = []

    if normalized_input.get("age", 0) >= 55:
        factors.append(f"Age of {normalized_input['age']} is in an older adult range.")
    if normalized_input.get("trestbps", 0) >= 140:
        factors.append(f"Resting blood pressure of {normalized_input['trestbps']} mm Hg is elevated.")
    if normalized_input.get("chol", 0) >= 240:
        factors.append(f"Cholesterol of {normalized_input['chol']} mg/dl is elevated.")
    if normalized_input.get("fbs") == 1:
        factors.append("Fasting blood sugar was reported above 120 mg/dl.")
    if normalized_input.get("exang") == 1:
        factors.append("Exercise-induced angina was reported.")
    if normalized_input.get("oldpeak", 0) >= 2.0:
        factors.append(f"ST depression (oldpeak) of {normalized_input['oldpeak']} is notably elevated.")
    if normalized_input.get("ca", 0) >= 1:
        factors.append(f"{normalized_input['ca']} major vessel(s) showed narrowing on prior imaging.")
    if normalized_input.get("thalach", 999) < 100:
        factors.append(f"Maximum heart rate achieved of {normalized_input['thalach']} bpm was relatively low.")

    if not factors:
        factors.append("No submitted values fell outside the ranges this tool flags as notable.")
    return factors


def compute_risk_category(prediction: int) -> str:
    return "Higher modeled risk of heart disease" if prediction == 1 else "Lower modeled risk of heart disease"


def build_retrieval_query(normalized_input: Dict[str, Any], risk_category: str) -> str:
    sex_label = "male" if normalized_input.get("sex") == 1 else "female"
    return (
        f"Cardiovascular risk education for a {normalized_input.get('age')}-year-old {sex_label} "
        f"patient with {risk_category.lower()}, resting blood pressure "
        f"{normalized_input.get('trestbps')}, cholesterol {normalized_input.get('chol')}, "
        f"and exercise-induced angina={'yes' if normalized_input.get('exang') == 1 else 'no'}. "
        f"General preventive cardiovascular health information."
    )


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    """Parses `raw_text` as JSON, attempting exactly one safe repair (extracting
    the outermost {...} substring) if the first parse fails. Raises
    ExplanationServiceError if both attempts fail -- never invents fields.
    """
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ExplanationServiceError("Bedrock returned a response that could not be parsed as JSON.")


def validate_model_json(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ExplanationServiceError("Bedrock JSON response was not a JSON object.")
    missing = REQUIRED_JSON_FIELDS - payload.keys()
    if missing:
        raise ExplanationServiceError(f"Bedrock JSON response is missing required fields: {sorted(missing)}.")
    return payload


def sanitize_citations(
    raw_citations: Any, retrieved_chunks: List[RetrievedChunk],
) -> List[Dict[str, str]]:
    """Keeps only citations whose id corresponds to an actually-retrieved
    source, in the exact order sources were presented to the model. This is
    the anti-fabrication guard: the model cannot introduce a source that was
    not really retrieved.
    """
    valid_ids = {f"source_{i}": chunk for i, chunk in enumerate(retrieved_chunks, start=1)}
    if not isinstance(raw_citations, list):
        return []

    sanitized: List[Dict[str, str]] = []
    seen_ids = set()
    for entry in raw_citations:
        if not isinstance(entry, dict):
            continue
        citation_id = entry.get("id")
        if citation_id not in valid_ids or citation_id in seen_ids:
            continue
        seen_ids.add(citation_id)
        chunk = valid_ids[citation_id]
        sanitized.append({
            "id": citation_id,
            "title": str(entry.get("title") or "Knowledge base source"),
            "uri": chunk.source_uri or "",
        })
    return sanitized


def as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, (str, int, float))]


class ExplanationService:
    """High-level entry point used by api/index.py's /api/explain-risk handler."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        bedrock_client: Optional[AIClient] = None,
        retrieval_tool: Optional[KnowledgeRetrievalTool] = None,
    ):
        self._settings = settings or get_settings()
        self._client = bedrock_client or get_default_client(self._settings)
        # Prefer the local (no-AWS) retrieval index when it has been built --
        # this is what makes RAG work out of the box regardless of which
        # generation backend (Bedrock/Anthropic) is configured. Falls back to
        # the Bedrock Knowledge Base retrieval tool only if the local index
        # hasn't been built, preserving the original Phase 3 behavior.
        if retrieval_tool is not None:
            self._retrieval_tool = retrieval_tool
        elif is_index_available():
            self._retrieval_tool = LocalKnowledgeRetrievalTool()
        else:
            self._retrieval_tool = KnowledgeRetrievalTool(self._client)

    def is_available(self) -> bool:
        return self._client.is_configured()

    @property
    def bedrock_enabled(self) -> bool:
        """Whether an AI explanation backend (Bedrock, Anthropic, or Gemini)
        is enabled, regardless of whether it successfully configured (see
        is_available for that). Name kept for backward compatibility with
        existing callers/health fields."""
        return (
            self._settings.bedrock_enabled
            or self._settings.uses_anthropic()
            or self._settings.uses_gemini()
        )

    @property
    def kb_configured(self) -> bool:
        """Whether some retrieval backend has content to search -- the local
        TF-IDF index (see tools/local_knowledge_retrieval.py) or a configured
        Bedrock Knowledge Base ID. Generation can still run without either
        (see agent/nodes.py retrieve_context) -- this only controls whether
        retrieval/citations are attempted."""
        return isinstance(self._retrieval_tool, LocalKnowledgeRetrievalTool) or bool(
            self._settings.bedrock_knowledge_base_id
        )

    @property
    def client(self) -> AIClient:
        """The shared BedrockClient/AnthropicClient instance -- exposed so the
        LangGraph nodes (agent/nodes.py) can reuse the same warm client for
        generation without constructing their own."""
        return self._client

    @property
    def retrieval_tool(self) -> KnowledgeRetrievalTool:
        """The shared KnowledgeRetrievalTool instance -- exposed for the same
        reason as `client` above."""
        return self._retrieval_tool

    def unavailable_reason(self) -> str:
        return self._settings.missing_configuration_reason() or "The AI explanation service is unavailable."

    def explain(
        self,
        prediction: int,
        probability: Optional[float],
        normalized_input: Dict[str, Any],
        retrieval_k: Optional[int] = None,
    ) -> ExplanationResult:
        """Produces a grounded explanation, or a clearly-flagged unavailable
        result if Bedrock is disabled/unconfigured/failing. Never raises for
        those expected cases -- callers can always show *something* to the user.
        """
        risk_category = compute_risk_category(prediction)
        input_factors = rule_based_input_factors(normalized_input)

        if not self.is_available():
            reason = self._settings.missing_configuration_reason() or "The AI explanation service is unavailable."
            logger.info("explanation_skipped reason=configuration_unavailable")
            return ExplanationResult(
                explanation_available=False,
                risk_category=risk_category,
                probability=probability,
                input_factors=input_factors,
                unavailable_reason=reason,
            )

        query = build_retrieval_query(normalized_input, risk_category)
        try:
            chunks = self._retrieval_tool.retrieve(query, top_k=retrieval_k)
            logger.info("retrieval_succeeded chunk_count=%d", len(chunks))
        except (BedrockConfigurationError, BedrockAuthError, BedrockThrottledError, BedrockServiceError) as exc:
            logger.warning("retrieval_failed error_type=%s", type(exc).__name__)
            return ExplanationResult(
                explanation_available=False,
                risk_category=risk_category,
                probability=probability,
                input_factors=input_factors,
                unavailable_reason="The AI explanation service could not retrieve supporting information right now.",
            )

        user_message = build_user_message(
            prediction=prediction,
            probability=probability,
            risk_category=risk_category,
            normalized_input=normalized_input,
            input_factors=input_factors,
            retrieved_chunks=chunks,
        )

        try:
            raw_text = self._client.generate_explanation(SYSTEM_PROMPT, user_message)
            logger.info("generation_succeeded")
        except (BedrockConfigurationError, BedrockAuthError, BedrockThrottledError, BedrockServiceError) as exc:
            logger.warning("generation_failed error_type=%s", type(exc).__name__)
            return ExplanationResult(
                explanation_available=False,
                risk_category=risk_category,
                probability=probability,
                input_factors=input_factors,
                unavailable_reason="The AI explanation service could not generate an explanation right now.",
            )

        try:
            payload = validate_model_json(extract_json_object(raw_text))
        except ExplanationServiceError as exc:
            logger.warning("generation_invalid_json error=%s", str(exc))
            return ExplanationResult(
                explanation_available=False,
                risk_category=risk_category,
                probability=probability,
                input_factors=input_factors,
                unavailable_reason="The AI explanation service returned an unusable response.",
            )

        citations = sanitize_citations(payload.get("citations"), chunks)

        return ExplanationResult(
            explanation_available=True,
            summary=str(payload.get("summary") or ""),
            # risk_category/probability always come from the real ML output,
            # never from the model's echoed JSON, so Bedrock cannot alter them.
            risk_category=risk_category,
            probability=probability,
            input_factors=input_factors,
            educational_information=as_string_list(payload.get("educational_information")),
            questions_for_professional=as_string_list(payload.get("questions_for_professional")),
            citations=citations,
            disclaimer=str(payload.get("disclaimer") or DISCLAIMER),
        )


_default_service: Optional[ExplanationService] = None


def get_default_service() -> ExplanationService:
    global _default_service
    if _default_service is None:
        _default_service = ExplanationService()
    return _default_service
