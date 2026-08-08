"""Tests for the evaluation runner scripts (mock mode) and report generation.
No AWS calls occur anywhere in this file.
"""
import json

import pytest

from evaluation import evaluate_agent, evaluate_generation, evaluate_retrieval, evaluate_safety
from evaluation.report import build_summary_markdown, resolve_output_path, save_json


def test_evaluate_retrieval_mock_mode_is_stable():
    report_1 = evaluate_retrieval.run(top_k=3, live=False)
    report_2 = evaluate_retrieval.run(top_k=3, live=False)
    assert report_1["metrics"] == report_2["metrics"]
    assert report_1["mode"] == "mock"
    assert report_1["total_cases"] > 0


def test_evaluate_generation_mock_mode_runs_and_reports_metrics():
    report = evaluate_generation.run(top_k=3)
    assert report["mode"] == "mock"
    assert report["total_cases"] > 0
    assert "citation_support_rate" in report["metrics"]


def test_evaluate_safety_mock_mode_makes_no_aws_calls_and_reports_accuracy():
    report = evaluate_safety.run()
    assert report["mode"] == "mock"
    assert 0.0 <= report["metrics"]["safety_classification_accuracy"] <= 1.0


def test_evaluate_agent_mock_mode_covers_all_branches():
    report = evaluate_agent.run_mock()
    assert report["mode"] == "mock"
    for key in (
        "successful_explanation_rate", "bedrock_disabled_fallback_success_rate",
        "retrieval_empty_fallback_success_rate", "generation_failure_fallback_success_rate",
        "unsafe_output_interception_rate", "response_schema_validity_rate",
        "request_id_presence_rate", "prediction_preservation_rate",
    ):
        assert key in report["metrics"]


def test_evaluate_agent_prediction_preservation_is_perfect_in_mock_mode():
    report = evaluate_agent.run_mock()
    assert report["metrics"]["prediction_preservation_rate"] == 1.0


def test_evaluate_agent_live_mode_refuses_without_configuration(monkeypatch):
    with pytest.raises(SystemExit) as exc_info:
        evaluate_agent.run_live()
    assert exc_info.value.code == 1


def test_evaluate_retrieval_live_mode_refuses_without_configuration():
    with pytest.raises(SystemExit) as exc_info:
        evaluate_retrieval.run(top_k=3, live=True)
    assert exc_info.value.code == 1


# --- report generation -----------------------------------------------------------

def test_save_json_writes_readable_file(tmp_path):
    path = tmp_path / "report.json"
    save_json(path, {"a": 1, "b": [1, 2, 3]})
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == {"a": 1, "b": [1, 2, 3]}


def test_resolve_output_path_uses_explicit_output(tmp_path):
    explicit = tmp_path / "custom" / "out.json"
    resolved = resolve_output_path("ignored.json", output=str(explicit))
    assert resolved == explicit


def test_resolve_output_path_avoids_overwriting_without_flag(tmp_path, monkeypatch):
    import evaluation.report as report_module
    monkeypatch.setattr(report_module, "RESULTS_DIR", tmp_path)

    first = resolve_output_path("retrieval_report.json")
    save_json(first, {"x": 1})
    assert first == tmp_path / "retrieval_report.json"

    second = resolve_output_path("retrieval_report.json", overwrite=False)
    assert second != first  # must be a new (timestamped) path, not clobbering the first


def test_resolve_output_path_overwrite_flag_reuses_same_path(tmp_path, monkeypatch):
    import evaluation.report as report_module
    monkeypatch.setattr(report_module, "RESULTS_DIR", tmp_path)

    first = resolve_output_path("retrieval_report.json")
    save_json(first, {"x": 1})

    second = resolve_output_path("retrieval_report.json", overwrite=True)
    assert second == first


def test_build_summary_markdown_includes_all_sections():
    reports = {
        "retrieval": {"mode": "mock", "timestamp": "t", "total_cases": 2, "passed": 2, "failed": 0,
                      "metrics": {"hit_rate_at_k": 1.0}, "failed_cases": []},
        "safety": {"mode": "mock", "timestamp": "t", "total_cases": 1, "passed": 1, "failed": 0,
                   "metrics": {"safety_classification_accuracy": 1.0}, "failed_cases": []},
    }
    markdown = build_summary_markdown(reports, ["Some limitation."])
    assert "Retrieval Evaluation" in markdown
    assert "Safety Evaluation" in markdown
    assert "Known Limitations" in markdown
    assert "Some limitation." in markdown


def test_build_summary_markdown_handles_failed_cases_table():
    reports = {
        "retrieval": {"mode": "mock", "timestamp": "t", "total_cases": 1, "passed": 0, "failed": 1,
                      "metrics": {}, "failed_cases": [{"case_id": "rag_001", "reason": "no chunks"}]},
    }
    markdown = build_summary_markdown(reports, [])
    assert "rag_001" in markdown
    assert "no chunks" in markdown
