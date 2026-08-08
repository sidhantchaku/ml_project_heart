"""LangGraph node functions for the /api/explain-risk workflow.

Each node takes the current CardioGraphState and returns a partial state
update (a dict), per LangGraph convention. Nodes reuse the existing Phase 2/3
services (tools/risk_prediction.py, tools/knowledge_retrieval.py,
services/bedrock_client.py, services/explanation_service.py) rather than
duplicating model loading, feature preprocessing, or boto3 calls. No node
raises on expected service failures -- those are converted into state fields
that routing functions (agent/graph.py) and format_response read.
"""
import logging
from typing import Any, Dict, List

from pydantic import ValidationError as PydanticValidationError

from agent.prompts import SYSTEM_PROMPT, build_full_user_message
from agent.safety import (
    is_user_message_too_long,
    screen_request,
    validate_generated_output,
)
from agent.state import CardioGraphState
from api.schemas import PatientInput
from config.settings import get_settings
from services.bedrock_client import (
    BedrockAuthError,
    BedrockConfigurationError,
    BedrockServiceError,
    BedrockThrottledError,
)
from services.explanation_service import (
    DISCLAIMER,
    ExplanationServiceError,
    as_string_list,
    build_retrieval_query,
    compute_risk_category,
    extract_json_object,
    get_default_service as get_explanation_service,
    rule_based_input_factors,
    sanitize_citations,
    validate_model_json,
)
from tools.knowledge_retrieval import RetrievedChunk
from tools.risk_prediction import (
    ModelLoadError,
    PredictionError,
    PredictionValidationError,
    get_default_service as get_prediction_service,
)

logger = logging.getLogger("cardiorisk.agent.nodes")


def _chunks_from_state(state: CardioGraphState) -> List[RetrievedChunk]:
    """Reconstructs RetrievedChunk objects from the plain-dict form stored in
    state (state must stay JSON-serializable, so chunks are stored as dicts).
    """
    return [
        RetrievedChunk(
            text=doc["text"], source_uri=doc.get("source_uri"),
            score=doc.get("score"), metadata=doc.get("metadata") or {},
        )
        for doc in state.get("retrieved_documents", [])
    ]


# --- 1. validate_input --------------------------------------------------------

def validate_input(state: CardioGraphState) -> Dict[str, Any]:
    patient_input = state.get("patient_input") or {}
    try:
        PatientInput(**patient_input)
    except PydanticValidationError as exc:
        errors = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()]
        logger.info("validate_input_failed request_id=%s error_count=%d", state.get("request_id"), len(errors))
        return {"validation_errors": errors}

    logger.info("validate_input_passed request_id=%s", state.get("request_id"))
    return {"validation_errors": []}


# --- 2. screen_request --------------------------------------------------------

def screen_request_node(state: CardioGraphState) -> Dict[str, Any]:
    user_message = state.get("user_message")

    if is_user_message_too_long(user_message):
        logger.info("screen_request_blocked request_id=%s category=message_too_long", state.get("request_id"))
        return {
            "safety_status": "blocked",
            "safety_category": "message_too_long",
            "safety_reason": "Your message is too long -- please ask a shorter, more specific question.",
        }

    result = screen_request(user_message)
    if not result["allowed"]:
        logger.info("screen_request_blocked request_id=%s category=%s", state.get("request_id"), result["category"])
        return {
            "safety_status": "blocked",
            "safety_category": result["category"],
            "safety_reason": result["reason"],
        }

    logger.info("screen_request_allowed request_id=%s", state.get("request_id"))
    return {"safety_status": "allowed", "safety_category": None, "safety_reason": None}


# --- 3. predict_risk -----------------------------------------------------------

def predict_risk(state: CardioGraphState) -> Dict[str, Any]:
    service = get_prediction_service()
    try:
        result = service.predict(state["patient_input"])
    except ModelLoadError:
        logger.warning("predict_risk_failed request_id=%s error_code=model_unavailable", state.get("request_id"))
        return {"error_code": "model_unavailable", "error_message": "Prediction service is unavailable."}
    except PredictionValidationError as exc:
        logger.warning("predict_risk_failed request_id=%s error_code=invalid_input", state.get("request_id"))
        return {"error_code": "invalid_input", "error_message": str(exc)}
    except PredictionError:
        logger.warning("predict_risk_failed request_id=%s error_code=prediction_failed", state.get("request_id"))
        return {"error_code": "prediction_failed", "error_message": "An error occurred while generating the prediction."}

    prediction = result["prediction"]
    normalized_input = result["normalized_input"]
    risk_category = compute_risk_category(prediction)
    input_factors = rule_based_input_factors(normalized_input)

    logger.info("predict_risk_completed request_id=%s prediction=%s", state.get("request_id"), prediction)
    return {
        "prediction": prediction,
        "probability": result["probability"],
        "normalized_input": normalized_input,
        "risk_category": risk_category,
        "notable_input_factors": input_factors,
    }


# --- 4. check_bedrock ----------------------------------------------------------

def check_bedrock(state: CardioGraphState) -> Dict[str, Any]:
    available = get_explanation_service().is_available()
    logger.info("check_bedrock request_id=%s available=%s", state.get("request_id"), available)
    return {"bedrock_available": available}


# --- 5. retrieve_context -------------------------------------------------------

def retrieve_context(state: CardioGraphState) -> Dict[str, Any]:
    """Retrieves grounding context, if a Knowledge Base is configured.

    A missing BEDROCK_KNOWLEDGE_BASE_ID is treated as "RAG not set up" rather
    than a retrieval failure: generation can still proceed (without citations)
    on the model's own general knowledge, per the graceful-degradation
    requirement (Bedrock generation should not be blocked just because the KB
    isn't configured yet). A KB that IS configured but returns nothing, or
    errors, still routes to limited_explanation -- generation never runs
    without evidence when RAG is actually enabled.
    """
    explanation_service = get_explanation_service()
    if not explanation_service.kb_configured:
        logger.info("retrieve_context_skipped request_id=%s reason=no_knowledge_base_configured", state.get("request_id"))
        return {"retrieval_query": None, "retrieved_documents": [], "retrieval_error": None, "kb_configured": False}

    query = build_retrieval_query(state["normalized_input"], state["risk_category"])

    try:
        chunks = explanation_service.retrieval_tool.retrieve(query, top_k=state.get("retrieval_k"))
    except (BedrockConfigurationError, BedrockAuthError, BedrockThrottledError, BedrockServiceError) as exc:
        logger.warning("retrieve_context_failed request_id=%s error_type=%s", state.get("request_id"), type(exc).__name__)
        return {"retrieval_query": query, "retrieved_documents": [], "retrieval_error": type(exc).__name__, "kb_configured": True}

    logger.info("retrieve_context_completed request_id=%s chunk_count=%d", state.get("request_id"), len(chunks))
    serializable = [
        {"text": c.text, "source_uri": c.source_uri, "score": c.score, "metadata": c.metadata}
        for c in chunks
    ]
    return {"retrieval_query": query, "retrieved_documents": serializable, "retrieval_error": None, "kb_configured": True}


# --- 6. generate_explanation ---------------------------------------------------

def generate_explanation(state: CardioGraphState) -> Dict[str, Any]:
    chunks = _chunks_from_state(state)
    explanation_service = get_explanation_service()

    user_message = build_full_user_message(
        prediction=state["prediction"],
        probability=state["probability"],
        risk_category=state["risk_category"],
        normalized_input=state["normalized_input"],
        input_factors=state.get("notable_input_factors", []),
        retrieved_chunks=chunks,
        user_message=state.get("user_message"),
    )

    try:
        raw_text = explanation_service.client.generate_explanation(SYSTEM_PROMPT, user_message)
    except (BedrockConfigurationError, BedrockAuthError, BedrockThrottledError, BedrockServiceError) as exc:
        logger.warning("generate_explanation_failed request_id=%s error_type=%s", state.get("request_id"), type(exc).__name__)
        return {"generation_error": type(exc).__name__}

    try:
        payload = validate_model_json(extract_json_object(raw_text))
    except ExplanationServiceError:
        logger.warning("generate_explanation_invalid_json request_id=%s", state.get("request_id"))
        return {"generation_error": "invalid_json"}

    citations = sanitize_citations(payload.get("citations"), chunks)
    generated = {
        "summary": str(payload.get("summary") or ""),
        "educational_information": as_string_list(payload.get("educational_information")),
        "questions_for_professional": as_string_list(payload.get("questions_for_professional")),
        "disclaimer": str(payload.get("disclaimer") or DISCLAIMER),
    }

    logger.info("generate_explanation_completed request_id=%s citation_count=%d", state.get("request_id"), len(citations))
    return {
        "generated_explanation": generated,
        "citations": citations,
        "generation_error": None,
        "explanation_available": True,
    }


# --- 7. validate_output ---------------------------------------------------------

def validate_output(state: CardioGraphState) -> Dict[str, Any]:
    # citations live in a separate state field (not inside generated_explanation
    # itself) -- merge them in for the safety check, which looks for a
    # "citations" key on the explanation payload.
    explanation = {**(state.get("generated_explanation") or {}), "citations": state.get("citations", [])}
    result = validate_generated_output(explanation, len(state.get("retrieved_documents", [])))

    if result["valid"]:
        return {"output_safety_status": "valid", "output_safety_category": None}

    logger.info("validate_output_blocked request_id=%s category=%s", state.get("request_id"), result["category"])
    return {"output_safety_status": "unsafe", "output_safety_category": result["category"]}


# --- fallback / refusal terminal nodes -----------------------------------------

def prediction_fallback(state: CardioGraphState) -> Dict[str, Any]:
    reason = get_settings().missing_configuration_reason() or "The AI explanation feature is unavailable."
    logger.info("prediction_fallback request_id=%s", state.get("request_id"))
    return {"explanation_available": False, "unavailable_reason": reason, "fallback_path": "prediction_fallback"}


def limited_explanation(state: CardioGraphState) -> Dict[str, Any]:
    """Handles both 'no retrieved context' outcomes: empty/failed retrieval,
    and (reached via the same fallback path) a failed generation call --
    either way the user still gets their prediction, just no grounded
    explanation.
    """
    if state.get("generation_error"):
        reason = "The AI explanation service could not generate an explanation right now."
    elif state.get("retrieval_error"):
        reason = "The AI explanation service could not retrieve supporting information right now."
    else:
        reason = "No supporting knowledge base content was found for this request."
    logger.info(
        "limited_explanation request_id=%s retrieval_error=%s generation_error=%s",
        state.get("request_id"), state.get("retrieval_error"), state.get("generation_error"),
    )
    return {"explanation_available": False, "unavailable_reason": reason, "fallback_path": "limited_explanation"}


def safe_refusal(state: CardioGraphState) -> Dict[str, Any]:
    """Handles a safety-blocked free-text request. Per spec, the underlying
    prediction is still preserved when the structured patient input was
    valid -- only the free-text explanation is refused.
    """
    prediction = None
    probability = None
    risk_category = ""
    input_factors: List[str] = []

    try:
        result = get_prediction_service().predict(state["patient_input"])
        prediction = result["prediction"]
        probability = result["probability"]
        risk_category = compute_risk_category(prediction)
        input_factors = rule_based_input_factors(result["normalized_input"])
    except (ModelLoadError, PredictionValidationError, PredictionError):
        logger.warning("safe_refusal_prediction_unavailable request_id=%s", state.get("request_id"))

    logger.info("safe_refusal request_id=%s category=%s", state.get("request_id"), state.get("safety_category"))
    return {
        "prediction": prediction,
        "probability": probability,
        "risk_category": risk_category,
        "notable_input_factors": input_factors,
        "explanation_available": False,
        "unavailable_reason": state.get("safety_reason"),
        "fallback_path": "safe_refusal",
    }


def safe_fallback(state: CardioGraphState) -> Dict[str, Any]:
    logger.info("safe_fallback request_id=%s category=%s", state.get("request_id"), state.get("output_safety_category"))
    return {
        "explanation_available": False,
        "unavailable_reason": (
            "The generated explanation could not be safely returned. Please consult a qualified "
            "healthcare professional for questions about this result."
        ),
        "fallback_path": "safe_fallback",
    }


# --- format_response -----------------------------------------------------------

def format_response(state: CardioGraphState) -> Dict[str, Any]:
    """Builds the single, stable final_response dict returned to the API layer.
    Never includes stack traces, AWS internals, or raw node names."""
    request_id = state.get("request_id")

    if state.get("validation_errors"):
        return {"final_response": {
            "request_id": request_id,
            "explanation_available": False,
            "error_code": "invalid_input",
            "error_message": "Invalid patient input.",
            "validation_errors": state["validation_errors"],
        }}

    if state.get("error_code"):
        return {"final_response": {
            "request_id": request_id,
            "explanation_available": False,
            "error_code": state["error_code"],
            "error_message": state.get("error_message") or "An error occurred while processing this request.",
        }}

    explanation_available = bool(state.get("explanation_available", False))
    generated = state.get("generated_explanation") or {}

    if explanation_available:
        summary = generated.get("summary", "")
        educational_information = generated.get("educational_information", [])
        questions_for_professional = generated.get("questions_for_professional", [])
        citations = state.get("citations", [])
        disclaimer = generated.get("disclaimer", DISCLAIMER)
        unavailable_reason = None
    else:
        unavailable_reason = state.get("unavailable_reason") or "An explanation is not available for this request."
        summary = unavailable_reason
        educational_information = []
        questions_for_professional = []
        citations = []
        disclaimer = DISCLAIMER

    response: Dict[str, Any] = {
        "request_id": request_id,
        "prediction": state.get("prediction"),
        "probability": state.get("probability"),
        "risk_category": state.get("risk_category", ""),
        "explanation_available": explanation_available,
        "unavailable_reason": unavailable_reason,
        "summary": summary,
        "input_factors": state.get("notable_input_factors", []),
        "educational_information": educational_information,
        "questions_for_professional": questions_for_professional,
        "citations": citations,
        "disclaimer": disclaimer,
    }

    if state.get("safety_status") == "blocked":
        response["safety_status"] = "blocked"
        response["safety_category"] = state.get("safety_category")

    logger.info(
        "format_response request_id=%s explanation_available=%s fallback_path=%s",
        request_id, explanation_available, state.get("fallback_path"),
    )
    return {"final_response": response}
