"""Report writing helpers shared by the evaluation scripts.

Nothing here makes AWS calls or is imported by the production API -- it only
reads the in-memory results each evaluate_*.py script already computed and
writes them to evaluation/results/.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

RESULTS_DIR = Path(__file__).parent / "results"


def get_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_output_path(filename: str, output: str = None, overwrite: bool = False) -> Path:
    """Decides where a report file should be written.

    - If `output` is given explicitly, always use it (the caller asked for a
      specific path).
    - Otherwise, write to evaluation/results/<filename>. If that file already
      exists and `overwrite` is False, write into a timestamped subfolder
      instead so historical reports are never silently clobbered.
    """
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    default_path = RESULTS_DIR / filename
    if not default_path.exists() or overwrite:
        return default_path

    timestamp_dir = RESULTS_DIR / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamp_dir.mkdir(parents=True, exist_ok=True)
    return timestamp_dir / filename


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _format_metrics_table(metrics: Dict[str, Any]) -> str:
    if not metrics:
        return "_No metrics recorded._\n"
    lines = ["| Metric | Value |", "|---|---|"]
    for name, value in metrics.items():
        if isinstance(value, float):
            value = round(value, 4)
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines) + "\n"


def _format_failed_cases_table(failed_cases: List[Dict[str, Any]]) -> str:
    if not failed_cases:
        return "_No failed cases._\n"
    lines = ["| Case ID | Reason |", "|---|---|"]
    for case in failed_cases:
        case_id = case.get("case_id", case.get("id", "?"))
        reason = str(case.get("reason", "")).replace("|", "/")
        lines.append(f"| {case_id} | {reason} |")
    return "\n".join(lines) + "\n"


def build_summary_markdown(reports: Dict[str, Dict[str, Any]], limitations: List[str]) -> str:
    """Builds the evaluation/results/summary.md content from the four report dicts:
    {"retrieval": ..., "generation": ..., "safety": ..., "agent": ...}
    """
    lines = ["# CardioRisk-AI Evaluation Summary", ""]
    lines.append(f"Generated: {get_timestamp()}")
    lines.append("")

    for section_name, report in reports.items():
        if report is None:
            continue
        lines.append(f"## {section_name.title()} Evaluation")
        lines.append("")
        lines.append(f"- Mode: `{report.get('mode', 'unknown')}`")
        lines.append(f"- Timestamp: {report.get('timestamp', 'unknown')}")
        lines.append(f"- Total cases: {report.get('total_cases', 0)}")
        lines.append(f"- Passed: {report.get('passed', 0)}")
        lines.append(f"- Failed: {report.get('failed', 0)}")
        lines.append("")
        lines.append("### Metrics")
        lines.append("")
        lines.append(_format_metrics_table(report.get("metrics", {})))
        lines.append("### Failed cases")
        lines.append("")
        lines.append(_format_failed_cases_table(report.get("failed_cases", [])))

    lines.append("## Known Limitations")
    lines.append("")
    for limitation in limitations:
        lines.append(f"- {limitation}")
    lines.append("")

    return "\n".join(lines)


DEFAULT_LIMITATIONS = [
    "Retrieval relevance and groundedness are judged by keyword/lexical overlap, not true "
    "semantic understanding -- treat scores as directional signals, not proof of correctness.",
    "Mock mode uses a small hand-written illustrative corpus and template-based generation, "
    "not the real (currently unpopulated) Bedrock Knowledge Base or a real foundation model.",
    "Safety screening is deterministic pattern matching and is not a complete guarantee of "
    "medical safety; novel phrasing can potentially evade it.",
    "Live mode results depend entirely on whatever content has actually been loaded into the "
    "real Knowledge Base at run time.",
]
