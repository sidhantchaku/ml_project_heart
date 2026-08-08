"""Pydantic request/response schemas for the CardioRisk-AI prediction API.

Numeric/categorical constraints mirror the min/max/select options already
present in public/index.html. Response field names (`prediction`, `probability`,
`status`) match the contract public/script.js already expects -- do not rename
without updating the frontend.
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class PatientInput(BaseModel):
    """Raw patient data submitted from the frontend form."""

    model_config = ConfigDict(extra="forbid")

    age: int = Field(..., ge=1, le=120)
    sex: Literal["Male", "Female"]
    cp: Literal[0, 1, 2, 3]
    trestbps: int = Field(..., ge=50, le=250)
    chol: int = Field(..., ge=50, le=600)
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: int = Field(..., ge=50, le=250)
    exang: Literal[0, 1]
    oldpeak: float = Field(..., ge=0.0, le=10.0)
    slope: Literal[0, 1, 2]
    ca: Literal[0, 1, 2, 3, 4]
    thal: Literal[0, 1, 2, 3]


class PredictionResponse(BaseModel):
    """Response returned by POST /api/predict."""

    prediction: int
    probability: Optional[float] = None


class HealthResponse(BaseModel):
    """Response returned by GET /api/health.

    `status` is preserved exactly as Phase 2 defined it for backward
    compatibility. The additional fields are new in Phase 3 and are additive
    only -- no existing field was removed or renamed.
    """

    status: str
    model_loaded: Optional[bool] = None
    scaler_loaded: Optional[bool] = None
    ai_provider: Optional[str] = None
    bedrock_enabled: Optional[bool] = None
    bedrock_configured: Optional[bool] = None
    agentcore_enabled: Optional[bool] = None
    agentcore_configured: Optional[bool] = None
    agentcore_runtime_arn_present: Optional[bool] = None
    local_agent_available: Optional[bool] = None


class ErrorResponse(BaseModel):
    """Documents the shape of error responses (matches FastAPI's default HTTPException body)."""

    detail: str


class ExplainRiskRequest(PatientInput):
    """Request body for POST /api/explain-risk. Reuses every PatientInput field
    and validation rule, plus explanation-specific controls.

    `user_message` is optional free text for questions like "explain what
    this result means" or "what should I ask a doctor?" -- it is screened by
    agent/safety.py before use and NEVER used to derive the prediction itself,
    which always comes from the validated structured fields above.
    """

    include_explanation: bool = True
    retrieval_k: Optional[int] = Field(default=None, ge=1, le=5)
    user_message: Optional[str] = Field(default=None, max_length=500)


class Citation(BaseModel):
    """A single source citation, always backed by an actually-retrieved chunk
    -- never fabricated (see services/explanation_service.py's sanitization)."""

    id: str
    title: str
    uri: str


class ExplanationResponse(BaseModel):
    """Response returned by POST /api/explain-risk.

    `prediction` and `probability` always reflect the real scikit-learn
    output -- the same values /api/predict would return for this input.
    `request_id` and `safety_status`/`safety_category` are populated by the
    LangGraph workflow (agent/graph.py) as of Phase 4; they are additive and
    optional so this remains compatible with the Phase 3 response shape.
    """

    request_id: Optional[str] = None

    prediction: Optional[int] = None
    probability: Optional[float] = None
    risk_category: str = ""
    input_factors: List[str] = Field(default_factory=list)

    explanation_available: bool
    unavailable_reason: Optional[str] = None

    summary: str = ""
    educational_information: List[str] = Field(default_factory=list)
    questions_for_professional: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    disclaimer: str = ""

    safety_status: Optional[str] = None
    safety_category: Optional[str] = None
