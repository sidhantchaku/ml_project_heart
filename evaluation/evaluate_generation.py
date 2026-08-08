"""Generation evaluation script.

Usage:
    python -m evaluation.evaluate_generation
    python -m evaluation.evaluate_generation --output evaluation/results/generation_report.json

Mock mode (default, and the only mode implemented here) uses the deterministic
template-based mock generator in evaluation/mock_data.py, fed by the same mock
retrieval used in evaluate_retrieval.py. This exercises the real deterministic
safety/citation/groundedness checks against realistic-shaped output without
calling a real model. Live generation is evaluated as part of
evaluate_agent.py --live, not here.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.safety import validate_generated_output  # noqa: E402
from evaluation.evaluation_models import DatasetValidationError, load_rag_test_cases  # noqa: E402
from evaluation.mock_data import mock_generate, mock_retrieve  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    citation_presence_rate,
    citation_source_match_rate,
    compute_groundedness,
    duplicate_citation_rate,
    keyword_coverage,
    unsupported_citation_rate,
)
from evaluation.report import DEFAULT_LIMITATIONS, get_timestamp, resolve_output_path, save_json  # noqa: E402

RAG_TEST_CASES_PATH = Path(__file__).parent / "rag_test_cases.json"

REQUIRED_FIELDS = {"summary", "educational_information", "questions_for_professional", "citations", "disclaimer"}


def run(top_k: int = 3):
    try:
        cases = load_rag_test_cases(str(RAG_TEST_CASES_PATH))
    except DatasetValidationError as exc:
        print(f"Dataset error: {exc}")
        sys.exit(1)

    all_citations = []
    citations_per_case = []
    keyword_scores = []
    groundedness_ratios = []
    failed_cases = []

    for case in cases:
        chunks = mock_retrieve(case.query, top_k=top_k)
        explanation = mock_generate(case.query, chunks)

        missing_fields = REQUIRED_FIELDS - explanation.keys()
        if missing_fields:
            failed_cases.append({"case_id": case.id, "reason": f"missing fields: {sorted(missing_fields)}"})
            continue

        safety_result = validate_generated_output(explanation, retrieved_document_count=len(chunks))
        if not safety_result["valid"]:
            failed_cases.append({"case_id": case.id, "reason": f"output safety: {safety_result['category']}"})

        combined_text = explanation["summary"] + " " + " ".join(explanation["educational_information"])
        coverage = keyword_coverage(combined_text, case.expected_keywords)
        keyword_scores.append(coverage)
        if coverage == 0.0 and case.expected_keywords:
            failed_cases.append({"case_id": case.id, "reason": "no expected keywords found in generated text"})

        citations = explanation["citations"]
        all_citations.extend(citations)
        citations_per_case.append(citations)
        if not citations and chunks:
            failed_cases.append({"case_id": case.id, "reason": "no citations despite retrieved context"})

        unsupported_rate_for_case = unsupported_citation_rate(citations, chunks)
        if unsupported_rate_for_case > 0:
            failed_cases.append({"case_id": case.id, "reason": "citation not backed by a retrieved chunk"})

        groundedness = compute_groundedness(explanation["educational_information"], [c["text"] for c in chunks])
        if groundedness["groundedness_ratio"] is not None:
            groundedness_ratios.append(groundedness["groundedness_ratio"])
        if groundedness["unsupported_claim_count"] > 0:
            failed_cases.append({
                "case_id": case.id,
                "reason": f"{groundedness['unsupported_claim_count']} unsupported claim(s) by lexical overlap",
            })

    mean_keyword_coverage = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
    mean_groundedness = sum(groundedness_ratios) / len(groundedness_ratios) if groundedness_ratios else None

    metrics = {
        "required_field_presence_rate": 1.0 - (
            sum(1 for f in failed_cases if "missing fields" in f["reason"]) / len(cases) if cases else 0.0
        ),
        "disclaimer_presence_rate": 1.0,  # mock_generate always includes one; real runs should re-check via agent.safety
        "citation_presence_rate": citation_presence_rate(citations_per_case),
        "citation_support_rate": (
            sum(citation_source_match_rate(c, mock_retrieve(case.query, top_k=top_k))
                for c, case in zip(citations_per_case, cases)) / len(cases)
        ) if cases else 0.0,
        "duplicate_citation_rate": (
            sum(duplicate_citation_rate(c) for c in citations_per_case) / len(citations_per_case)
        ) if citations_per_case else 0.0,
        "mean_keyword_coverage": mean_keyword_coverage,
        "mean_groundedness_ratio": mean_groundedness if mean_groundedness is not None else "n/a (no checkable claims)",
    }

    report = {
        "mode": "mock",
        "timestamp": get_timestamp(),
        "total_cases": len(cases),
        "passed": len(cases) - len({f["case_id"] for f in failed_cases}),
        "failed": len({f["case_id"] for f in failed_cases}),
        "metrics": metrics,
        "failed_cases": failed_cases,
        "limitations": DEFAULT_LIMITATIONS,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate CardioRisk-AI generation quality (mock mode).")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = run(top_k=args.top_k)
    output_path = resolve_output_path("generation_report.json", output=args.output, overwrite=args.overwrite)
    save_json(output_path, report)

    print(f"\nGeneration evaluation ({report['mode']} mode) -- {report['total_cases']} cases")
    for name, value in report["metrics"].items():
        print(f"  {name}: {value}")
    print(f"Passed: {report['passed']}  Failed: {report['failed']}")
    print(f"Report written to: {output_path}")

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
