"""End-to-end agent (LangGraph) evaluation script.

Usage:
    python -m evaluation.evaluate_agent
    python -m evaluation.evaluate_agent --live

Mock mode (default):
  - Runs entirely without AWS credentials, deterministic, CI-safe.
  - Exercises all major routing branches: Bedrock disabled, successful
    explanation, empty retrieval, generation failure, unsafe generated output.

Live mode (--live), explicit opt-in only:
  - Checks Bedrock configuration before doing anything.
  - Warns that real AWS calls may incur cost.
  - Refuses to run if the model ID or Knowledge Base ID is missing.
  - Records model id, knowledge base id, region, and timestamp in the report
    (never credentials).
"""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.nodes as nodes  # noqa: E402
from agent.graph import invoke_cardio_graph  # noqa: E402
from api.schemas import ExplanationResponse  # noqa: E402
from evaluation.mock_data import MOCK_CORPUS, mock_generate, mock_retrieve  # noqa: E402
from evaluation.report import DEFAULT_LIMITATIONS, get_timestamp, resolve_output_path, save_json  # noqa: E402
from services.bedrock_client import BedrockServiceError  # noqa: E402
from tools.knowledge_retrieval import RetrievedChunk  # noqa: E402
from tools.risk_prediction import get_default_service as get_prediction_service  # noqa: E402

VALID_PATIENT = {
    "age": 52, "sex": "Male", "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
}


def _mock_service(enabled=True, retrieve_return=None, retrieve_side_effect=None,
                   generate_return=None, generate_side_effect=None):
    service = MagicMock()
    service.is_available.return_value = enabled
    if retrieve_side_effect is not None:
        service.retrieval_tool.retrieve.side_effect = retrieve_side_effect
    else:
        service.retrieval_tool.retrieve.return_value = retrieve_return or []
    if generate_side_effect is not None:
        service.client.generate_explanation.side_effect = generate_side_effect
    else:
        service.client.generate_explanation.return_value = generate_return or "{}"
    return service


def _apply(service):
    nodes.get_explanation_service = lambda: service


def _validate_schema(final_response: dict) -> bool:
    try:
        ExplanationResponse(**final_response)
        return True
    except Exception:
        return False


def _check_direct_prediction():
    result = get_prediction_service().predict(VALID_PATIENT)
    return result["prediction"]


def run_mock():
    direct_prediction = _check_direct_prediction()
    original_get_explanation_service = nodes.get_explanation_service
    scenarios = []

    try:
        # 1. Bedrock disabled -> prediction_fallback
        _apply(_mock_service(enabled=False))
        scenarios.append(("bedrock_disabled", invoke_cardio_graph(VALID_PATIENT)))

        # 2. Successful explanation, using the mock corpus/generator
        sample_chunk = RetrievedChunk(text=MOCK_CORPUS[0]["text"], source_uri=MOCK_CORPUS[0]["source_uri"],
                                       score=0.9, metadata={"category": MOCK_CORPUS[0]["category"]})
        mock_json = mock_generate("general cardiovascular education", [
            {"text": sample_chunk.text, "source_uri": sample_chunk.source_uri}
        ])
        _apply(_mock_service(enabled=True, retrieve_return=[sample_chunk], generate_return=json.dumps(mock_json)))
        scenarios.append(("successful_explanation", invoke_cardio_graph(VALID_PATIENT)))

        # 3. Empty retrieval -> limited_explanation
        _apply(_mock_service(enabled=True, retrieve_return=[]))
        scenarios.append(("empty_retrieval", invoke_cardio_graph(VALID_PATIENT)))

        # 4. Generation failure -> limited_explanation
        _apply(_mock_service(enabled=True, retrieve_return=[sample_chunk],
                              generate_side_effect=BedrockServiceError("mock failure")))
        scenarios.append(("generation_failure", invoke_cardio_graph(VALID_PATIENT)))

        # 5. Unsafe generated output -> safe_fallback
        unsafe_json = {**mock_json, "summary": "You should take aspirin daily."}
        _apply(_mock_service(enabled=True, retrieve_return=[sample_chunk], generate_return=json.dumps(unsafe_json)))
        scenarios.append(("unsafe_output", invoke_cardio_graph(VALID_PATIENT)))

        # 6. Safety-blocked free-text request -> safe_refusal (prediction still preserved)
        _apply(_mock_service(enabled=False))
        scenarios.append(("safety_blocked", invoke_cardio_graph(VALID_PATIENT, user_message="What medication should I take?")))
    finally:
        # Never leave the process-wide agent.nodes.get_explanation_service patched --
        # other code (tests, other evaluation scripts) in the same process must see
        # the real singleton again.
        nodes.get_explanation_service = original_get_explanation_service

    return _summarize(scenarios, direct_prediction, mode="mock")


def run_live():
    from config.settings import get_settings

    settings = get_settings()
    if not settings.is_bedrock_ready():
        reason = settings.missing_configuration_reason() or "Bedrock is not configured."
        print(f"Refusing to run --live: {reason}")
        sys.exit(1)

    print("WARNING: --live mode will make real AWS Bedrock calls and may incur cost.")

    direct_prediction = _check_direct_prediction()
    result = invoke_cardio_graph(VALID_PATIENT)
    scenarios = [("live_run", result)]
    report = _summarize(scenarios, direct_prediction, mode="live")
    report["live_config"] = {
        "region": settings.aws_region,
        "model_id": settings.bedrock_model_id,
        "knowledge_base_id": settings.bedrock_knowledge_base_id,
        "timestamp": get_timestamp(),
    }
    return report


def _summarize(scenarios, direct_prediction, mode: str):
    failed_cases = []
    schema_valid_count = 0
    request_id_count = 0
    prediction_preserved_count = 0

    outcomes = {}
    for name, result in scenarios:
        outcomes[name] = result
        if _validate_schema(result):
            schema_valid_count += 1
        else:
            failed_cases.append({"case_id": name, "reason": "final response failed schema validation"})
        if result.get("request_id"):
            request_id_count += 1
        else:
            failed_cases.append({"case_id": name, "reason": "missing request_id"})
        if result.get("prediction") == direct_prediction:
            prediction_preserved_count += 1
        else:
            failed_cases.append({
                "case_id": name,
                "reason": f"prediction mismatch: graph={result.get('prediction')} direct={direct_prediction}",
            })

    def _ok(name, expected_condition):
        if not expected_condition:
            failed_cases.append({"case_id": name, "reason": "scenario did not behave as expected"})
        return 1.0 if expected_condition else 0.0

    successful_explanation_rate = _ok(
        "successful_explanation", outcomes.get("successful_explanation", {}).get("explanation_available") is True,
    )
    bedrock_disabled_fallback_success_rate = _ok(
        "bedrock_disabled", outcomes.get("bedrock_disabled", {}).get("explanation_available") is False
        and outcomes.get("bedrock_disabled", {}).get("prediction") is not None,
    )
    retrieval_empty_fallback_success_rate = _ok(
        "empty_retrieval", outcomes.get("empty_retrieval", {}).get("explanation_available") is False,
    )
    generation_failure_fallback_success_rate = _ok(
        "generation_failure", outcomes.get("generation_failure", {}).get("explanation_available") is False,
    )
    unsafe_output_interception_rate = _ok(
        "unsafe_output", outcomes.get("unsafe_output", {}).get("explanation_available") is False,
    )

    n = len(scenarios)
    metrics = {
        "successful_explanation_rate": successful_explanation_rate,
        "bedrock_disabled_fallback_success_rate": bedrock_disabled_fallback_success_rate,
        "retrieval_empty_fallback_success_rate": retrieval_empty_fallback_success_rate,
        "generation_failure_fallback_success_rate": generation_failure_fallback_success_rate,
        "unsafe_output_interception_rate": unsafe_output_interception_rate,
        "response_schema_validity_rate": schema_valid_count / n if n else 0.0,
        "request_id_presence_rate": request_id_count / n if n else 0.0,
        "prediction_preservation_rate": prediction_preserved_count / n if n else 0.0,
    }

    failed_ids = {f["case_id"] for f in failed_cases}
    return {
        "mode": mode,
        "timestamp": get_timestamp(),
        "total_cases": n,
        "passed": n - len(failed_ids),
        "failed": len(failed_ids),
        "metrics": metrics,
        "failed_cases": failed_cases,
        "limitations": DEFAULT_LIMITATIONS,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the CardioRisk-AI LangGraph agent end-to-end.")
    parser.add_argument("--live", action="store_true", help="Run against real AWS Bedrock (explicit opt-in, may incur cost).")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = run_live() if args.live else run_mock()

    output_path = resolve_output_path("agent_report.json", output=args.output, overwrite=args.overwrite)
    save_json(output_path, report)

    print(f"\nAgent evaluation ({report['mode']} mode) -- {report['total_cases']} scenarios")
    for name, value in report["metrics"].items():
        print(f"  {name}: {value}")
    print(f"Passed: {report['passed']}  Failed: {report['failed']}")
    print(f"Report written to: {output_path}")

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
