"""FastAPI application serving the CardioRisk-AI heart disease risk prediction API.

This module is the Vercel serverless entry point referenced by vercel.json's
rewrites, and is also importable locally, e.g.:

    uvicorn api.index:app --reload

Prediction logic lives in tools/risk_prediction.py so it is not duplicated
between deployment targets. As of Phase 4, /api/explain-risk is orchestrated
by the LangGraph workflow in agent/graph.py (validation, safety screening,
prediction, retrieval, generation, output safety, response formatting);
/api/predict's behaviour is unchanged by any of this.
"""
import logging
import os
import sys
import uuid

# Make the repository root importable so "tools", "api", "services", "config",
# and "agent" resolve as packages, whether Vercel loads this file directly or
# it is imported as "api.index" locally.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException  # noqa: E402

from agent.graph import invoke_cardio_graph  # noqa: E402
from api.schemas import (  # noqa: E402
    ExplainRiskRequest,
    ExplanationResponse,
    HealthResponse,
    PatientInput,
    PredictionResponse,
)
from config.settings import get_settings as get_app_settings  # noqa: E402
from tools.risk_prediction import (  # noqa: E402
    ModelLoadError,
    PredictionError,
    PredictionValidationError,
    get_default_service as get_prediction_service,
)
from services.agentcore_client import (  # noqa: E402
    AgentCoreAuthError,
    AgentCoreConfigurationError,
    AgentCoreResponseError,
    AgentCoreUnavailableError,
    get_default_client as get_agentcore_client,
)
from services.explanation_service import (  # noqa: E402
    DISCLAIMER,
    compute_risk_category,
    rule_based_input_factors,
    get_default_service as get_explanation_service,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cardiorisk.api")

app = FastAPI()

_prediction_service = get_prediction_service()
_explanation_service = get_explanation_service()
_agentcore_client = get_agentcore_client()
_app_settings = get_app_settings()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Lightweight status check. Never invokes AWS -- Bedrock/AgentCore
    fields reflect local configuration/client-construction state only, not
    live reachability. AgentCore is never invoked by this endpoint.
    """
    agentcore_health = _agentcore_client.health_check()
    return HealthResponse(
        status="healthy" if _prediction_service.is_ready else "degraded",
        model_loaded=_prediction_service.is_ready,
        scaler_loaded=_prediction_service.is_ready,
        ai_provider=_app_settings.ai_provider,
        bedrock_enabled=_explanation_service.bedrock_enabled,
        bedrock_configured=_explanation_service.is_available(),
        agentcore_enabled=agentcore_health["enabled"],
        agentcore_configured=agentcore_health["configured"],
        agentcore_runtime_arn_present=agentcore_health["runtime_arn_present"],
        local_agent_available=_prediction_service.is_ready,
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(data: PatientInput) -> PredictionResponse:
    try:
        result = _prediction_service.predict(data.model_dump())
    except ModelLoadError:
        raise HTTPException(
            status_code=503,
            detail="Prediction service is unavailable: the model could not be loaded.",
        )
    except PredictionValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except PredictionError:
        raise HTTPException(
            status_code=500,
            detail="An error occurred while generating the prediction.",
        )

    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
    )


def _local_prediction_only_response(patient_data: dict, reason: str) -> dict:
    """Builds a safe, non-fabricated response when AgentCore is unavailable:
    the real scikit-learn prediction (computed locally, via the same shared
    service -- no preprocessing duplicated), but explanation_available=False
    with a clear, honest reason. This is NOT an automatic fallback to the
    local LangGraph/Bedrock explanation path -- that would silently change
    which system produced the explanation, so it is deliberately not done
    here. If a local-explanation fallback is ever wanted, it must be an
    explicit, separately-configured decision, not a hidden default.
    """
    try:
        result = _prediction_service.predict(patient_data)
        prediction = result["prediction"]
        probability = result["probability"]
        risk_category = compute_risk_category(prediction)
        input_factors = rule_based_input_factors(result["normalized_input"])
    except (ModelLoadError, PredictionValidationError, PredictionError):
        return {
            "request_id": uuid.uuid4().hex[:12],
            "explanation_available": False,
            "error_code": "model_unavailable",
            "error_message": "Prediction service is unavailable: the model could not be loaded.",
        }

    return {
        "request_id": uuid.uuid4().hex[:12],
        "prediction": prediction,
        "probability": probability,
        "risk_category": risk_category,
        "input_factors": input_factors,
        "explanation_available": False,
        "unavailable_reason": reason,
        "summary": reason,
        "educational_information": [],
        "questions_for_professional": [],
        "citations": [],
        "disclaimer": DISCLAIMER,
    }


_ERROR_STATUS_CODES = {
    "model_unavailable": 503,
    "invalid_input": 422,
    "prediction_failed": 500,
    "graph_execution_failed": 500,
}


@app.post("/api/explain-risk", response_model=ExplanationResponse)
def explain_risk(data: ExplainRiskRequest) -> ExplanationResponse:
    """Runs the existing scikit-learn prediction, then -- via the LangGraph
    workflow in agent/graph.py -- safety-screens the request, retrieves
    Knowledge Base context, generates a grounded explanation, and validates
    the output, all with graceful fallback at every step. Bedrock being
    disabled/unavailable/failing, or the request being screened as unsafe,
    never crashes this endpoint: a structured response is always returned.
    """
    patient_data = data.model_dump(exclude={"include_explanation", "retrieval_k", "user_message"})

    if not data.include_explanation:
        try:
            result = _prediction_service.predict(patient_data)
        except ModelLoadError:
            raise HTTPException(
                status_code=503,
                detail="Prediction service is unavailable: the model could not be loaded.",
            )
        except PredictionValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except PredictionError:
            raise HTTPException(
                status_code=500,
                detail="An error occurred while generating the prediction.",
            )

        logger.info("explain_risk_skipped reason=include_explanation_false")
        return ExplanationResponse(
            prediction=result["prediction"],
            probability=result["probability"],
            risk_category=compute_risk_category(result["prediction"]),
            input_factors=[],
            explanation_available=False,
            unavailable_reason="Explanation was not requested.",
            disclaimer="",
        )

    if _app_settings.use_agentcore:
        logger.info("explain_risk_requested route=agentcore")
        try:
            final_response = _agentcore_client.invoke(
                patient_input=patient_data,
                user_message=data.user_message,
                retrieval_k=data.retrieval_k,
            )
        except AgentCoreConfigurationError:
            final_response = _local_prediction_only_response(patient_data, reason=(
                _app_settings.missing_agentcore_configuration_reason()
                or "The AI explanation service (AgentCore Runtime) is not configured."
            ))
        except (AgentCoreAuthError, AgentCoreUnavailableError, AgentCoreResponseError) as exc:
            logger.warning("agentcore_invocation_failed error_type=%s", type(exc).__name__)
            final_response = _local_prediction_only_response(
                patient_data, reason="The AI explanation service (AgentCore Runtime) is currently unavailable.",
            )
    else:
        logger.info("explain_risk_requested route=local_langgraph bedrock_enabled=%s", _explanation_service.bedrock_enabled)
        final_response = invoke_cardio_graph(
            patient_input=patient_data,
            user_message=data.user_message,
            retrieval_k=data.retrieval_k,
        )

    error_code = final_response.get("error_code")
    if error_code:
        status_code = _ERROR_STATUS_CODES.get(error_code, 500)
        raise HTTPException(status_code=status_code, detail=final_response.get("error_message", "An error occurred."))

    logger.info(
        "explain_risk_completed request_id=%s explanation_available=%s",
        final_response.get("request_id"), final_response.get("explanation_available"),
    )

    return ExplanationResponse(**final_response)
