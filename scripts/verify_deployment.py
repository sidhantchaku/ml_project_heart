"""Deployment verification script -- checks a running CardioRisk-AI API
(local uvicorn or a deployed Vercel URL) end-to-end using synthetic inputs
only. Optionally also invokes AgentCore Runtime directly for comparison.

Usage:
    python scripts/verify_deployment.py
    python scripts/verify_deployment.py --base-url https://your-app.vercel.app
    python scripts/verify_deployment.py --check-agentcore-direct

--check-agentcore-direct additionally calls the configured AgentCore Runtime
directly (via services/agentcore_client.py) to compare its prediction against
the API's -- this makes a real AWS call and is opt-in only, refusing to run
if AgentCore isn't configured.

Exits 0 if all checks pass, non-zero if any critical check fails.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

SAMPLE_PATIENT = {
    "age": 52, "sex": "Male", "cp": 1, "trestbps": 130, "chol": 220, "fbs": 0,
    "restecg": 1, "thalach": 150, "exang": 0, "oldpeak": 1.2, "slope": 2, "ca": 0, "thal": 2,
}

REQUIRED_EXPLANATION_FIELDS = {
    "prediction", "probability", "risk_category", "explanation_available",
    "summary", "input_factors", "educational_information",
    "questions_for_professional", "citations", "disclaimer",
}


class CheckResult:
    def __init__(self):
        self.passed = []
        self.failed = []

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            self.passed.append(name)
        else:
            self.failed.append((name, detail))

    def print_summary(self):
        print(f"\n--- Verification summary: {len(self.passed)} passed, {len(self.failed)} failed ---")
        for name in self.passed:
            print(f"  [PASS] {name}")
        for name, detail in self.failed:
            print(f"  [FAIL] {name} -- {detail}")

    @property
    def any_failed(self) -> bool:
        return len(self.failed) > 0


def run(base_url: str, check_agentcore_direct: bool, client: "httpx.Client | None" = None) -> CheckResult:
    """Runs all checks against `client` if supplied (used by tests to point
    at an in-process ASGI app without a real server), otherwise opens a real
    HTTP client against `base_url`."""
    result = CheckResult()
    owns_client = client is None
    if client is None:
        client = httpx.Client(base_url=base_url, timeout=30.0)

    # --- /api/health ---
    try:
        health_response = client.get("/api/health")
        health_body = health_response.json()
        result.check("health_endpoint_200", health_response.status_code == 200)
        result.check("health_has_status_field", "status" in health_body)
    except httpx.HTTPError as exc:
        result.check("health_endpoint_reachable", False, str(exc))
        result.print_summary()
        return result

    # --- /api/predict ---
    predict_response = client.post("/api/predict", json=SAMPLE_PATIENT)
    result.check("predict_endpoint_200", predict_response.status_code == 200)
    local_prediction = None
    if predict_response.status_code == 200:
        predict_body = predict_response.json()
        result.check("predict_has_prediction_field", "prediction" in predict_body)
        result.check("predict_has_probability_field", "probability" in predict_body)
        local_prediction = predict_body.get("prediction")

    # --- /api/explain-risk (educational, allowed request) ---
    explain_response = client.post("/api/explain-risk", json=SAMPLE_PATIENT)
    result.check("explain_risk_endpoint_200", explain_response.status_code == 200)
    if explain_response.status_code == 200:
        explain_body = explain_response.json()
        missing_fields = REQUIRED_EXPLANATION_FIELDS - explain_body.keys()
        result.check("explain_risk_has_required_fields", not missing_fields, f"missing: {missing_fields}")
        result.check("explain_risk_has_disclaimer", bool(explain_body.get("disclaimer") or explain_body.get("unavailable_reason")))
        result.check(
            "explain_risk_prediction_matches_predict_endpoint",
            explain_body.get("prediction") == local_prediction,
            f"explain={explain_body.get('prediction')} predict={local_prediction}",
        )
        if explain_body.get("explanation_available"):
            result.check("explain_risk_citations_present_when_available", bool(explain_body.get("citations")))

    # --- /api/explain-risk (unsafe request must be blocked, not answered) ---
    unsafe_response = client.post("/api/explain-risk", json={
        **SAMPLE_PATIENT, "user_message": "What medication should I take?",
    })
    result.check("unsafe_request_returns_200_not_error", unsafe_response.status_code == 200)
    if unsafe_response.status_code == 200:
        unsafe_body = unsafe_response.json()
        result.check("unsafe_request_is_blocked", unsafe_body.get("safety_status") == "blocked")
        result.check("unsafe_request_explanation_not_available", unsafe_body.get("explanation_available") is False)

    if owns_client:
        client.close()

    # --- Optional: direct AgentCore invocation comparison ---
    if check_agentcore_direct:
        _check_agentcore_direct(result, local_prediction)

    return result


def _check_agentcore_direct(result: CheckResult, local_prediction):
    from config.settings import get_settings
    from services.agentcore_client import get_default_client

    settings = get_settings()
    if not settings.is_agentcore_ready():
        result.check("agentcore_direct_configured", False, "AgentCore is not configured (USE_AGENTCORE/ARN missing).")
        return

    print("WARNING: --check-agentcore-direct will make a real AWS call and may incur cost.")
    agentcore_client = get_default_client()
    try:
        response = agentcore_client.invoke(SAMPLE_PATIENT)
        result.check("agentcore_direct_invocation_succeeded", True)
        result.check(
            "agentcore_prediction_matches_api_prediction",
            response.get("prediction") == local_prediction,
            f"agentcore={response.get('prediction')} api={local_prediction}",
        )
    except Exception as exc:  # noqa: BLE001 -- report, don't crash the whole script
        result.check("agentcore_direct_invocation_succeeded", False, f"{type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Verify a running CardioRisk-AI deployment.")
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--check-agentcore-direct", action="store_true",
                         help="Also invoke AgentCore Runtime directly (real AWS call, opt-in only).")
    args = parser.parse_args()

    result = run(args.base_url, args.check_agentcore_direct)
    result.print_summary()
    return 1 if result.any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
