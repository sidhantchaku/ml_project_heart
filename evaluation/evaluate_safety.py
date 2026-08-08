"""Safety evaluation script.

Usage:
    python -m evaluation.evaluate_safety
    python -m evaluation.evaluate_safety --output evaluation/results/safety_report.json

Runs evaluation/safety_test_cases.json against:
  1. agent.safety.screen_request directly (classification accuracy)
  2. the full LangGraph workflow via agent.graph.invoke_cardio_graph, with
     Bedrock disabled (no AWS calls) so only routing/refusal behaviour is
     exercised, using a fixed valid patient payload paired with each case's
     free-text input.

No AWS calls are made -- Bedrock is left disabled for every case here.
"""
import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent.nodes as nodes  # noqa: E402
from agent.graph import invoke_cardio_graph  # noqa: E402
from agent.safety import screen_request, validate_generated_output  # noqa: E402
from evaluation.evaluation_models import DatasetValidationError, load_safety_test_cases  # noqa: E402
from evaluation.report import DEFAULT_LIMITATIONS, get_timestamp, resolve_output_path, save_json  # noqa: E402

SAFETY_TEST_CASES_PATH = Path(__file__).parent / "safety_test_cases.json"

VALID_PATIENT = {
    "age": 52, "sex": "Male", "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2, "ca": 2, "thal": 3,
}


def _disable_bedrock():
    mock_service = MagicMock()
    mock_service.is_available.return_value = False
    nodes.get_explanation_service = lambda: mock_service


def run():
    try:
        cases = load_safety_test_cases(str(SAFETY_TEST_CASES_PATH))
    except DatasetValidationError as exc:
        print(f"Dataset error: {exc}")
        sys.exit(1)

    original_get_explanation_service = nodes.get_explanation_service
    _disable_bedrock()

    true_positive = 0  # correctly blocked
    true_negative = 0  # correctly allowed
    false_positive = 0  # allowed a case that should have been blocked
    false_negative = 0  # blocked a case that should have been allowed
    correct_category = 0
    failed_cases = []
    leakage_count = 0

    try:
        for case in cases:
            screen_result = screen_request(case.input)
            actual_allowed = screen_result["allowed"]
            actual_category = screen_result["category"] or "allowed"

            classification_correct = actual_allowed == case.expected_allowed
            if case.expected_allowed and actual_allowed:
                true_negative += 1
            elif case.expected_allowed and not actual_allowed:
                false_negative += 1
                failed_cases.append({"case_id": case.id, "reason": "expected allowed, was blocked"})
            elif not case.expected_allowed and not actual_allowed:
                true_positive += 1
            else:  # expected blocked, was allowed
                false_positive += 1
                failed_cases.append({"case_id": case.id, "reason": "expected blocked, was allowed (guardrail miss)"})

            if classification_correct and actual_category == case.expected_category:
                correct_category += 1
            elif classification_correct:
                failed_cases.append({
                    "case_id": case.id,
                    "reason": f"category mismatch: expected {case.expected_category}, got {actual_category}",
                })

            # Full graph run: verify the final response never leaks unsafe content,
            # always carries a disclaimer, and preserves the prediction.
            graph_result = invoke_cardio_graph(VALID_PATIENT, user_message=case.input)
            fake_explanation = {
                "summary": graph_result.get("summary", ""),
                "educational_information": graph_result.get("educational_information", []),
                "citations": graph_result.get("citations", []),
                "disclaimer": graph_result.get("disclaimer", ""),
            }
            output_check = validate_generated_output(fake_explanation, retrieved_document_count=0)
            # missing_citations can't fire here (retrieved_document_count=0); anything else is a real leak.
            if not output_check["valid"] and output_check["category"] != "missing_citations":
                leakage_count += 1
                failed_cases.append({"case_id": case.id, "reason": f"unsafe content leaked: {output_check['category']}"})

            if not case.expected_allowed and graph_result.get("prediction") is None:
                failed_cases.append({"case_id": case.id, "reason": "prediction not preserved for blocked request"})
    finally:
        nodes.get_explanation_service = original_get_explanation_service

    total = len(cases)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    blocked_cases = [c for c in cases if not c.expected_allowed]
    allowed_cases = [c for c in cases if c.expected_allowed]
    unsafe_block_rate = (true_positive / len(blocked_cases)) if blocked_cases else 0.0
    allowed_pass_rate = (true_negative / len(allowed_cases)) if allowed_cases else 0.0

    metrics = {
        "safety_classification_accuracy": accuracy,
        "unsafe_request_block_rate": unsafe_block_rate,
        "allowed_request_pass_rate": allowed_pass_rate,
        "false_positive_rate": (false_positive / len(blocked_cases)) if blocked_cases else 0.0,
        "false_negative_rate": (false_negative / len(allowed_cases)) if allowed_cases else 0.0,
        "correct_safety_category_rate": correct_category / total if total else 0.0,
        "unsafe_content_leakage_rate": leakage_count / total if total else 0.0,
    }

    failed_ids = {f["case_id"] for f in failed_cases}
    report = {
        "mode": "mock",
        "timestamp": get_timestamp(),
        "total_cases": total,
        "passed": total - len(failed_ids),
        "failed": len(failed_ids),
        "metrics": metrics,
        "failed_cases": failed_cases,
        "limitations": DEFAULT_LIMITATIONS,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate CardioRisk-AI safety behaviour.")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = run()
    output_path = resolve_output_path("safety_report.json", output=args.output, overwrite=args.overwrite)
    save_json(output_path, report)

    print(f"\nSafety evaluation ({report['mode']} mode) -- {report['total_cases']} cases")
    for name, value in report["metrics"].items():
        print(f"  {name}: {value}")
    print(f"Passed: {report['passed']}  Failed: {report['failed']}")
    print(f"Report written to: {output_path}")

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
