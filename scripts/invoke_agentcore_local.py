"""Local sanity-check for the AgentCore entry point -- no server, no AWS calls
unless Bedrock/AgentCore are actually enabled via environment variables.

Usage:
    python scripts/invoke_agentcore_local.py
    python scripts/invoke_agentcore_local.py --user-message "What questions should I ask a doctor?"

Loads a synthetic (not-real) sample request and prints the structured
response produced by infrastructure/agentcore/runtime_adapter.handle_request(),
the exact function the deployed AgentCore Runtime calls.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from infrastructure.agentcore.runtime_adapter import RuntimeRequestError, handle_request  # noqa: E402

# Synthetic sample patient -- not real patient data.
SAMPLE_REQUEST = {
    "patient_input": {
        "age": 52,
        "sex": "Male",
        "cp": 1,
        "trestbps": 130,
        "chol": 220,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 2,
    },
    "user_message": "Explain what this risk estimate means.",
}


def main():
    parser = argparse.ArgumentParser(description="Invoke the AgentCore entry point locally with a sample request.")
    parser.add_argument("--user-message", type=str, default=None, help="Override the sample user_message.")
    args = parser.parse_args()

    request = dict(SAMPLE_REQUEST)
    if args.user_message is not None:
        request["user_message"] = args.user_message

    print("Request:")
    print(json.dumps(request, indent=2))
    print()

    try:
        response = handle_request(request)
    except RuntimeRequestError as exc:
        print(f"Request rejected: {exc}")
        return 1

    print("Response:")
    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
