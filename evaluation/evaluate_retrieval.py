"""Retrieval evaluation script.

Usage:
    python -m evaluation.evaluate_retrieval
    python -m evaluation.evaluate_retrieval --live
    python -m evaluation.evaluate_retrieval --top-k 3
    python -m evaluation.evaluate_retrieval --output evaluation/results/retrieval_report.json

Default mode uses the small mock corpus in evaluation/mock_data.py -- no AWS
credentials or network access are used. --live calls the real Bedrock
Knowledge Base via the existing tools/knowledge_retrieval.py service, and
refuses to run without Bedrock properly configured.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.evaluation_models import DatasetValidationError, load_rag_test_cases  # noqa: E402
from evaluation.mock_data import mock_retrieve  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    empty_retrieval_rate,
    expected_source_retrieval_rate,
    hit_rate_at_k,
    keyword_coverage,
    mean_retrieval_score,
)
from evaluation.report import DEFAULT_LIMITATIONS, get_timestamp, resolve_output_path, save_json  # noqa: E402

RAG_TEST_CASES_PATH = Path(__file__).parent / "rag_test_cases.json"


def _run_mock(cases, top_k: int):
    per_case_results = []
    for case in cases:
        chunks = mock_retrieve(case.query, top_k=top_k)
        per_case_results.append({"case": case.model_dump(), "chunks": chunks})
    return per_case_results, []


def _run_live(cases, top_k: int):
    from config.settings import get_settings
    from services.bedrock_client import get_default_client
    from tools.knowledge_retrieval import KnowledgeRetrievalTool

    settings = get_settings()
    if not settings.is_bedrock_ready():
        reason = settings.missing_configuration_reason() or "Bedrock is not configured."
        print(f"Refusing to run --live: {reason}")
        print("Set BEDROCK_ENABLED=true, BEDROCK_MODEL_ID, and BEDROCK_KNOWLEDGE_BASE_ID to run live evaluation.")
        sys.exit(1)

    print("WARNING: --live mode will make real AWS Bedrock calls and may incur cost.")
    print(f"region={settings.aws_region} model_id={settings.bedrock_model_id} "
          f"knowledge_base_id={settings.bedrock_knowledge_base_id}")

    tool = KnowledgeRetrievalTool(get_default_client())
    per_case_results = []
    errors = []
    for case in cases:
        try:
            chunks = tool.retrieve(case.query, top_k=top_k)
            chunk_dicts = [
                {"text": c.text, "source_uri": c.source_uri, "score": c.score, "metadata": c.metadata}
                for c in chunks
            ]
            per_case_results.append({"case": case.model_dump(), "chunks": chunk_dicts})
        except Exception as exc:  # noqa: BLE001 -- record and continue, don't crash the whole run
            per_case_results.append({"case": case.model_dump(), "chunks": []})
            errors.append({"case_id": case.id, "reason": f"retrieval error: {type(exc).__name__}"})
    return per_case_results, errors


def run(top_k: int = 3, live: bool = False):
    try:
        cases = load_rag_test_cases(str(RAG_TEST_CASES_PATH))
    except DatasetValidationError as exc:
        print(f"Dataset error: {exc}")
        sys.exit(1)

    per_case_results, errors = (_run_live(cases, top_k) if live else _run_mock(cases, top_k))

    hit_rate = hit_rate_at_k(per_case_results)
    expected_source_rate = expected_source_retrieval_rate(per_case_results)
    empty_rate = empty_retrieval_rate(per_case_results)

    keyword_scores = []
    failed_cases = []
    for item in per_case_results:
        case = item["case"]
        combined_text = " ".join(chunk.get("text", "") for chunk in item["chunks"])
        coverage = keyword_coverage(combined_text, case.get("expected_keywords", []))
        keyword_scores.append(coverage)
        if not item["chunks"]:
            failed_cases.append({"case_id": case["id"], "reason": "no chunks retrieved"})
        elif coverage == 0.0 and case.get("expected_keywords"):
            failed_cases.append({"case_id": case["id"], "reason": "no expected keywords found in retrieved text"})

    for error in errors:
        failed_cases.append(error)

    mean_keyword_coverage = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0

    all_scores = []
    for item in per_case_results:
        score = mean_retrieval_score(item["chunks"])
        if score is not None:
            all_scores.append(score)
    mean_score = (sum(all_scores) / len(all_scores)) if all_scores else None

    passed = len(per_case_results) - len(failed_cases)
    report = {
        "mode": "live" if live else "mock",
        "timestamp": get_timestamp(),
        "top_k": top_k,
        "total_cases": len(cases),
        "passed": max(passed, 0),
        "failed": len(failed_cases),
        "metrics": {
            "hit_rate_at_k": hit_rate,
            "expected_source_retrieval_rate": expected_source_rate,
            "mean_keyword_coverage": mean_keyword_coverage,
            "empty_retrieval_rate": empty_rate,
            "mean_retrieval_score": mean_score if mean_score is not None else "n/a (no comparable scores)",
        },
        "failed_cases": failed_cases,
        "limitations": DEFAULT_LIMITATIONS,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate CardioRisk-AI retrieval quality.")
    parser.add_argument("--live", action="store_true", help="Use real Bedrock Knowledge Base retrieval.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per case.")
    parser.add_argument("--output", type=str, default=None, help="Explicit output path for the JSON report.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite evaluation/results/retrieval_report.json if it exists.")
    args = parser.parse_args()

    report = run(top_k=args.top_k, live=args.live)

    output_path = resolve_output_path("retrieval_report.json", output=args.output, overwrite=args.overwrite)
    save_json(output_path, report)

    print(f"\nRetrieval evaluation ({report['mode']} mode) -- {report['total_cases']} cases")
    for name, value in report["metrics"].items():
        print(f"  {name}: {value}")
    print(f"Passed: {report['passed']}  Failed: {report['failed']}")
    print(f"Report written to: {output_path}")

    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
