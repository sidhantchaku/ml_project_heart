"""CI-friendly orchestrator: runs the complete offline evaluation suite.

    python -m evaluation.run_all

Runs retrieval, generation, safety, and agent evaluation in mock mode only
(no AWS calls), writes all four JSON reports plus evaluation/results/summary.md,
checks the quality gates in evaluation/thresholds.py, and exits non-zero if any
critical gate fails. Deterministic -- safe for CI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation import evaluate_agent, evaluate_generation, evaluate_retrieval, evaluate_safety  # noqa: E402
from evaluation import thresholds  # noqa: E402
from evaluation.report import RESULTS_DIR, build_summary_markdown, save_json  # noqa: E402


def main():
    print("Running CardioRisk-AI offline evaluation suite (mock mode, no AWS calls)...\n")

    retrieval_report = evaluate_retrieval.run(top_k=3, live=False)
    generation_report = evaluate_generation.run(top_k=3)
    safety_report = evaluate_safety.run()
    agent_report = evaluate_agent.run_mock()

    reports = {
        "retrieval": retrieval_report,
        "generation": generation_report,
        "safety": safety_report,
        "agent": agent_report,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_json(RESULTS_DIR / "retrieval_report.json", retrieval_report)
    save_json(RESULTS_DIR / "generation_report.json", generation_report)
    save_json(RESULTS_DIR / "safety_report.json", safety_report)
    save_json(RESULTS_DIR / "agent_report.json", agent_report)

    limitations = retrieval_report.get("limitations", [])
    summary_md = build_summary_markdown(reports, limitations)
    (RESULTS_DIR / "summary.md").write_text(summary_md, encoding="utf-8")

    print("--- Quality gate checks ---")
    gates = [
        thresholds.check_gate(
            "response_schema_validity_rate",
            agent_report["metrics"]["response_schema_validity_rate"],
            minimum=thresholds.RESPONSE_SCHEMA_VALIDITY_MIN,
        ),
        thresholds.check_gate(
            "safety_classification_accuracy",
            safety_report["metrics"]["safety_classification_accuracy"],
            minimum=thresholds.SAFETY_BLOCK_ACCURACY_MIN,
        ),
        thresholds.check_gate(
            "unsafe_content_leakage_rate",
            safety_report["metrics"]["unsafe_content_leakage_rate"],
            maximum=thresholds.UNSAFE_CONTENT_LEAKAGE_MAX,
        ),
        thresholds.check_gate(
            "citation_support_rate",
            generation_report["metrics"]["citation_support_rate"],
            minimum=thresholds.CITATION_SUPPORT_RATE_MIN,
        ),
        thresholds.check_gate(
            "prediction_preservation_rate",
            agent_report["metrics"]["prediction_preservation_rate"],
            minimum=thresholds.PREDICTION_PRESERVATION_MIN,
        ),
    ]

    fallback_metrics = [
        agent_report["metrics"]["bedrock_disabled_fallback_success_rate"],
        agent_report["metrics"]["retrieval_empty_fallback_success_rate"],
        agent_report["metrics"]["generation_failure_fallback_success_rate"],
        agent_report["metrics"]["unsafe_output_interception_rate"],
    ]
    mean_fallback_success = sum(fallback_metrics) / len(fallback_metrics)
    gates.append(thresholds.check_gate(
        "mean_fallback_success_rate", mean_fallback_success, minimum=thresholds.FALLBACK_SUCCESS_RATE_MIN,
    ))

    if thresholds.ENFORCE_RETRIEVAL_HIT_RATE:
        gates.append(thresholds.check_gate(
            "retrieval_hit_rate_at_k", retrieval_report["metrics"]["hit_rate_at_k"],
            minimum=thresholds.RETRIEVAL_HIT_RATE_AT_K_MIN,
        ))
    else:
        print(f"  retrieval_hit_rate_at_k (informational only): {retrieval_report['metrics']['hit_rate_at_k']}")

    any_failed = False
    for gate in gates:
        status = "PASS" if gate["passed"] else "FAIL"
        print(f"  [{status}] {gate['name']} = {gate['value']} "
              f"(min={gate['minimum']}, max={gate['maximum']})")
        if not gate["passed"]:
            any_failed = True

    print(f"\nSummary written to: {RESULTS_DIR / 'summary.md'}")

    if any_failed:
        print("\nOne or more quality gates FAILED.")
        return 1

    print("\nAll quality gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
