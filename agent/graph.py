"""Builds and compiles the LangGraph workflow for /api/explain-risk, and
exposes a single invocation helper (`invoke_cardio_graph`) for the API layer.

The graph is compiled once at module import time (no checkpointer, no
persistence, no human-in-the-loop) and reused across requests within a warm
serverless instance -- the same pattern used for the prediction/explanation
service singletons in tools/ and services/.

    START
      -> validate_input --(invalid)--> format_response
                          --(valid)--> screen_request
      screen_request --(blocked)--> safe_refusal --> format_response
                     --(allowed)--> predict_risk
      predict_risk --(failed)--> format_response
                   --(success)--> check_bedrock
      check_bedrock --(unavailable)--> prediction_fallback --> format_response
                    --(available)--> retrieve_context
      retrieve_context --(empty/error)--> limited_explanation --> format_response
                        --(found)--> generate_explanation
      generate_explanation --(error)--> limited_explanation --> format_response
                           --(ok)--> validate_output
      validate_output --(unsafe)--> safe_fallback --> format_response
                       --(valid)--> format_response
      format_response -> END
"""
import logging
import uuid
from typing import Any, Dict, Optional

from langgraph.graph import END, StateGraph

from agent.nodes import (
    check_bedrock,
    format_response,
    generate_explanation,
    limited_explanation,
    predict_risk,
    prediction_fallback,
    retrieve_context,
    safe_fallback,
    safe_refusal,
    screen_request_node,
    validate_input,
    validate_output,
)
from agent.state import CardioGraphState

logger = logging.getLogger("cardiorisk.agent.graph")


# --- Conditional routing functions ---------------------------------------------

def route_after_validate_input(state: CardioGraphState) -> str:
    return "invalid" if state.get("validation_errors") else "valid"


def route_after_screen_request(state: CardioGraphState) -> str:
    return "blocked" if state.get("safety_status") == "blocked" else "allowed"


def route_after_predict_risk(state: CardioGraphState) -> str:
    return "failed" if state.get("error_code") else "success"


def route_after_check_bedrock(state: CardioGraphState) -> str:
    return "available" if state.get("bedrock_available") else "unavailable"


def route_after_retrieve_context(state: CardioGraphState) -> str:
    if state.get("retrieval_error"):
        return "empty"
    if not state.get("kb_configured", True):
        # No Knowledge Base configured at all -- proceed to generation without
        # citations rather than blocking the whole explanation on RAG setup.
        return "found"
    if not state.get("retrieved_documents"):
        return "empty"
    return "found"


def route_after_generate_explanation(state: CardioGraphState) -> str:
    return "failed" if state.get("generation_error") else "ok"


def route_after_validate_output(state: CardioGraphState) -> str:
    return "unsafe" if state.get("output_safety_status") == "unsafe" else "valid"


def _build_graph():
    graph = StateGraph(CardioGraphState)

    graph.add_node("validate_input", validate_input)
    graph.add_node("screen_request", screen_request_node)
    graph.add_node("predict_risk", predict_risk)
    graph.add_node("check_bedrock", check_bedrock)
    graph.add_node("retrieve_context", retrieve_context)
    graph.add_node("generate_explanation", generate_explanation)
    graph.add_node("validate_output", validate_output)
    graph.add_node("prediction_fallback", prediction_fallback)
    graph.add_node("limited_explanation", limited_explanation)
    graph.add_node("safe_refusal", safe_refusal)
    graph.add_node("safe_fallback", safe_fallback)
    graph.add_node("format_response", format_response)

    graph.set_entry_point("validate_input")

    graph.add_conditional_edges(
        "validate_input", route_after_validate_input,
        {"invalid": "format_response", "valid": "screen_request"},
    )
    graph.add_conditional_edges(
        "screen_request", route_after_screen_request,
        {"blocked": "safe_refusal", "allowed": "predict_risk"},
    )
    graph.add_conditional_edges(
        "predict_risk", route_after_predict_risk,
        {"failed": "format_response", "success": "check_bedrock"},
    )
    graph.add_conditional_edges(
        "check_bedrock", route_after_check_bedrock,
        {"unavailable": "prediction_fallback", "available": "retrieve_context"},
    )
    graph.add_conditional_edges(
        "retrieve_context", route_after_retrieve_context,
        {"empty": "limited_explanation", "found": "generate_explanation"},
    )
    graph.add_conditional_edges(
        "generate_explanation", route_after_generate_explanation,
        {"failed": "limited_explanation", "ok": "validate_output"},
    )
    graph.add_conditional_edges(
        "validate_output", route_after_validate_output,
        {"unsafe": "safe_fallback", "valid": "format_response"},
    )

    graph.add_edge("safe_refusal", "format_response")
    graph.add_edge("prediction_fallback", "format_response")
    graph.add_edge("limited_explanation", "format_response")
    graph.add_edge("safe_fallback", "format_response")
    graph.add_edge("format_response", END)

    # No checkpointer: stateless per invocation, no persistence, Vercel-safe.
    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    """Process-wide singleton -- the graph is compiled once per warm instance."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_graph()
    return _compiled_graph


def invoke_cardio_graph(
    patient_input: Dict[str, Any],
    user_message: Optional[str] = None,
    retrieval_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Runs the full explain-risk workflow and returns the final structured
    response dict (the same shape format_response produces). Handles
    unexpected graph failures cleanly -- callers never see a raw traceback.
    """
    request_id = uuid.uuid4().hex[:12]
    initial_state: CardioGraphState = {
        "request_id": request_id,
        "patient_input": patient_input,
        "user_message": user_message,
        "retrieval_k": retrieval_k,
    }

    logger.info("graph_invocation_started request_id=%s", request_id)
    try:
        final_state = get_compiled_graph().invoke(initial_state)
    except Exception:  # noqa: BLE001 - last-resort guard; graph nodes already catch expected errors
        logger.exception("graph_invocation_failed request_id=%s", request_id)
        return {
            "request_id": request_id,
            "explanation_available": False,
            "error_code": "graph_execution_failed",
            "error_message": "An unexpected error occurred while processing this request.",
        }

    logger.info("graph_invocation_completed request_id=%s", request_id)
    return final_state.get("final_response", {
        "request_id": request_id,
        "explanation_available": False,
        "error_code": "graph_execution_failed",
        "error_message": "The workflow completed without producing a response.",
    })
