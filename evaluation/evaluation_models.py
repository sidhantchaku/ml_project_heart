"""Typed models for the evaluation framework.

These validate the evaluation datasets (rag_test_cases.json,
safety_test_cases.json) before any evaluation script runs, and give the
various evaluation modules a consistent shape for results. Kept deliberately
simple -- this is an evaluation tool, not production API surface.
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ValidationError

__all__ = [
    "RagTestCase",
    "SafetyTestCase",
    "RetrievalResult",
    "GenerationResult",
    "MetricResult",
    "EvaluationSummary",
    "load_rag_test_cases",
    "load_safety_test_cases",
    "DatasetValidationError",
]


class DatasetValidationError(RuntimeError):
    """Raised when an evaluation dataset file is malformed. Fails loudly and
    clearly rather than silently skipping bad cases."""


class RagTestCase(BaseModel):
    """One retrieval/generation evaluation case (evaluation/rag_test_cases.json)."""

    id: str
    category: str
    query: str
    expected_keywords: List[str] = Field(default_factory=list)
    expected_source_keywords: List[str] = Field(default_factory=list)
    expected_document_ids: List[str] = Field(default_factory=list)
    minimum_relevant_chunks: int = 1
    notes: str = ""


class SafetyTestCase(BaseModel):
    """One safety-screening evaluation case (evaluation/safety_test_cases.json)."""

    id: str
    input: str
    expected_allowed: bool
    expected_category: str


class RetrievalResult(BaseModel):
    """Normalized retrieval outcome for one test case, independent of whether
    it came from a mock corpus or a real Bedrock Knowledge Base call."""

    case_id: str
    query: str
    chunks: List[Dict[str, Any]] = Field(default_factory=list)  # text/source_uri/score/metadata
    error: Optional[str] = None


class GenerationResult(BaseModel):
    """Normalized generation outcome for one test case."""

    case_id: str
    explanation: Optional[Dict[str, Any]] = None
    citations: List[Dict[str, str]] = Field(default_factory=list)
    raw_text: Optional[str] = None
    error: Optional[str] = None


class MetricResult(BaseModel):
    """A single named metric value, with optional supporting detail."""

    name: str
    value: float
    details: Dict[str, Any] = Field(default_factory=dict)


class EvaluationSummary(BaseModel):
    """Top-level result of running one evaluation script."""

    mode: str  # "mock" | "live"
    timestamp: str
    total_cases: int
    passed: int
    failed: int
    metrics: Dict[str, float] = Field(default_factory=dict)
    failed_cases: List[Dict[str, Any]] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)


def load_rag_test_cases(path: str) -> List[RagTestCase]:
    return _load_cases(path, RagTestCase)


def load_safety_test_cases(path: str) -> List[SafetyTestCase]:
    return _load_cases(path, SafetyTestCase)


def _load_cases(path: str, model_cls):
    import json

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"Could not read/parse dataset file '{path}': {exc}") from exc

    if not isinstance(raw, list):
        raise DatasetValidationError(f"Dataset file '{path}' must contain a JSON array of cases.")

    cases = []
    errors = []
    for index, entry in enumerate(raw):
        try:
            cases.append(model_cls(**entry))
        except ValidationError as exc:
            errors.append(f"case[{index}] ({entry.get('id', '?')}): {exc}")

    if errors:
        raise DatasetValidationError(
            f"Dataset file '{path}' has {len(errors)} malformed case(s):\n" + "\n".join(errors)
        )

    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise DatasetValidationError(f"Dataset file '{path}' has duplicate ids: {sorted(duplicates)}")

    return cases
