"""AgentCore Runtime entry point for CardioRisk-AI.

This is the file the AgentCore CLI packages and deploys (see
infrastructure/agentcore/README.md for exact commands). It is intentionally
a thin wrapper: request/response glue only. All real logic -- validation,
safety, prediction, retrieval, generation, LangGraph routing -- lives in the
existing application code and is reused unchanged via runtime_adapter.py.

Local testing (no deployment, no AWS calls unless Bedrock is enabled):
    python infrastructure/agentcore/agent_entrypoint.py
        starts the AgentCore local dev server (bedrock_agentcore SDK), which
        listens for invocations the same way the deployed runtime would.

    python scripts/invoke_agentcore_local.py
        calls handle_request() directly, no server, for a quick sanity check.
"""
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from bedrock_agentcore.runtime import BedrockAgentCoreApp  # noqa: E402

from infrastructure.agentcore.runtime_adapter import RuntimeRequestError, handle_request  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cardiorisk.agentcore.entrypoint")

# Built once at import time (cold start), reused for every invocation within
# this runtime process -- the compiled LangGraph workflow and the prediction/
# explanation service singletons it imports transitively are not rebuilt or
# reloaded per request (see agent/graph.py, tools/risk_prediction.py).
app = BedrockAgentCoreApp()


@app.entrypoint
def invoke(payload, context=None):
    """AgentCore Runtime invocation handler.

    `payload` is the JSON request body AgentCore passes through (see
    runtime_adapter.handle_request for the expected shape). Always returns a
    JSON-serializable dict and never raises -- internal exceptions are caught
    here and converted to a safe, structured error response so no stack
    trace or AWS/internal detail ever reaches the caller. No internal
    LangGraph state is exposed, only the final formatted response.
    """
    try:
        return handle_request(payload)
    except RuntimeRequestError as exc:
        logger.info("agentcore_request_rejected reason=validation_error")
        return {
            "explanation_available": False,
            "error_code": "invalid_input",
            "error_message": str(exc),
        }
    except Exception:  # noqa: BLE001 -- last-resort guard; invoke_cardio_graph already
        # catches its own internal failures, so reaching here means something
        # unexpected happened in the adapter itself (e.g. a malformed payload
        # shape not caught by RuntimeRequestError).
        logger.exception("agentcore_request_failed_unexpectedly")
        return {
            "explanation_available": False,
            "error_code": "runtime_execution_failed",
            "error_message": "An unexpected error occurred while processing this request.",
        }


if __name__ == "__main__":
    app.run()
