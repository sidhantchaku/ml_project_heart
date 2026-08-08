"""Typed state passed between LangGraph nodes for the /api/explain-risk workflow.

Only plain, JSON-serializable data lives here -- no boto3 clients, model
objects, or other service instances (those are looked up by nodes via the
existing service singletons in tools/ and services/, same as Phase 3).
"""
from typing import Any, Dict, List, Optional, TypedDict


class CardioGraphState(TypedDict, total=False):
    # --- request context ---
    request_id: str
    patient_input: Dict[str, Any]
    user_message: Optional[str]
    retrieval_k: Optional[int]

    # --- validate_input ---
    validation_errors: List[str]

    # --- screen_request (input safety) ---
    safety_status: str  # "allowed" | "blocked"
    safety_category: Optional[str]
    safety_reason: Optional[str]

    # --- predict_risk ---
    prediction: Optional[int]
    probability: Optional[float]
    normalized_input: Dict[str, Any]
    notable_input_factors: List[str]
    risk_category: str

    # --- check_bedrock ---
    bedrock_available: bool

    # --- retrieve_context ---
    retrieval_query: Optional[str]
    retrieved_documents: List[Dict[str, Any]]
    retrieval_error: Optional[str]
    kb_configured: bool  # False when BEDROCK_KNOWLEDGE_BASE_ID is unset -- generation may still proceed without citations

    # --- generate_explanation ---
    generated_explanation: Optional[Dict[str, Any]]
    citations: List[Dict[str, str]]
    generation_error: Optional[str]

    # --- validate_output (output safety) ---
    output_safety_status: str  # "valid" | "unsafe" | "not_checked"
    output_safety_category: Optional[str]

    # --- error tracking ---
    error_code: Optional[str]
    error_message: Optional[str]

    # --- terminal flags ---
    explanation_available: bool
    unavailable_reason: Optional[str]
    fallback_path: Optional[str]  # which non-happy-path node produced the final response, for logs

    # --- format_response ---
    final_response: Dict[str, Any]
