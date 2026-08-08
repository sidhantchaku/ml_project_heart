"""Thin adapter between the AgentCore Runtime request/response contract and
the existing, already-tested application services.

Deliberately does NOT duplicate model loading, feature preprocessing,
Bedrock retrieval/generation, safety logic, prompts, or LangGraph nodes --
it validates the incoming payload with the same Pydantic schema
/api/explain-risk uses, then calls the same compiled LangGraph workflow via
agent.graph.invoke_cardio_graph(). Kept separate from agent_entrypoint.py so
this logic is importable/testable without the bedrock_agentcore SDK.
"""
import logging
from typing import Any, Dict

from pydantic import ValidationError

from agent.graph import invoke_cardio_graph
from api.schemas import ExplainRiskRequest

logger = logging.getLogger("cardiorisk.agentcore.runtime_adapter")


class RuntimeRequestError(ValueError):
    """Raised when the incoming AgentCore payload fails validation."""


def handle_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validates `payload` and runs it through the existing LangGraph workflow.

    Expected payload shape:
        {
          "patient_input": {... same 13 fields as api.schemas.PatientInput ...},
          "user_message": "optional free text, max 500 chars",
          "retrieval_k": optional int 1-5
        }

    Returns the same final_response shape agent.graph.invoke_cardio_graph()
    already produces (see agent/nodes.py: format_response) -- this is the
    single response contract; nothing new is invented here.

    Raises RuntimeRequestError for a malformed payload. Does not raise for
    expected downstream failures (model/Bedrock/safety) -- those already
    come back as a structured response from invoke_cardio_graph().
    """
    if not isinstance(payload, dict):
        raise RuntimeRequestError("Request payload must be a JSON object.")

    patient_input = payload.get("patient_input")
    if not isinstance(patient_input, dict):
        raise RuntimeRequestError("Request payload must include a 'patient_input' object.")

    try:
        validated = ExplainRiskRequest(
            **patient_input,
            user_message=payload.get("user_message"),
            retrieval_k=payload.get("retrieval_k"),
        )
    except ValidationError as exc:
        raise RuntimeRequestError(f"Invalid patient input: {exc}") from exc

    patient_data = validated.model_dump(exclude={"include_explanation", "retrieval_k", "user_message"})

    logger.info("agentcore_request_received")
    result = invoke_cardio_graph(
        patient_input=patient_data,
        user_message=validated.user_message,
        retrieval_k=validated.retrieval_k,
    )
    logger.info(
        "agentcore_request_completed request_id=%s explanation_available=%s",
        result.get("request_id"), result.get("explanation_available"),
    )
    return result
