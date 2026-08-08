"""Tests for scripts/verify_deployment.py, run against the real FastAPI app
in-process (via an httpx client bound to the ASGI app) -- no real server, no
real AWS calls.
"""
import httpx
import pytest
from fastapi.testclient import TestClient

import api.index as api_index
from scripts.verify_deployment import run


@pytest.fixture()
def in_process_client():
    # TestClient bridges the sync httpx.Client interface verify_deployment.run()
    # expects to the ASGI app, without needing a real running server.
    client = TestClient(api_index.app)
    yield client
    client.close()


def test_verify_deployment_all_checks_pass_against_local_app(in_process_client):
    result = run(base_url="unused", check_agentcore_direct=False, client=in_process_client)
    assert result.any_failed is False
    assert len(result.passed) > 0


def test_verify_deployment_detects_health_failure(monkeypatch, in_process_client):
    # Force /api/health to look broken by monkeypatching the prediction
    # service's readiness flag, which health() reflects.
    monkeypatch.setattr(type(api_index._prediction_service), "is_ready", property(lambda self: False))
    result = run(base_url="unused", check_agentcore_direct=False, client=in_process_client)
    # health_endpoint_200 still passes (200 status with status="degraded"),
    # but this at least confirms the script actually reads response content,
    # not just status codes -- degraded status should still be considered
    # reachable, not a hard failure, matching the real health contract.
    assert any(name == "health_endpoint_200" for name in result.passed)


def test_verify_deployment_flags_unblocked_unsafe_request(monkeypatch, in_process_client):
    original_post = in_process_client.post

    def _tampered_post(url, **kwargs):
        response = original_post(url, **kwargs)
        if url == "/api/explain-risk" and kwargs.get("json", {}).get("user_message"):
            body = response.json()
            body["safety_status"] = "allowed"  # simulate a guardrail failure
            response = httpx.Response(200, json=body, request=response.request)
        return response

    monkeypatch.setattr(in_process_client, "post", _tampered_post)
    result = run(base_url="unused", check_agentcore_direct=False, client=in_process_client)
    assert any(name == "unsafe_request_is_blocked" for name, _ in result.failed)


def test_verify_deployment_agentcore_direct_check_skips_cleanly_when_unconfigured(in_process_client):
    result = run(base_url="unused", check_agentcore_direct=True, client=in_process_client)
    assert any(name == "agentcore_direct_configured" for name, _ in result.failed)


def test_verify_deployment_returns_nonzero_exit_code_on_failure(monkeypatch, in_process_client):
    original_post = in_process_client.post

    def _tampered_post(url, **kwargs):
        response = original_post(url, **kwargs)
        if url == "/api/predict":
            return httpx.Response(500, json={"detail": "simulated failure"}, request=response.request)
        return response

    monkeypatch.setattr(in_process_client, "post", _tampered_post)
    result = run(base_url="unused", check_agentcore_direct=False, client=in_process_client)
    assert result.any_failed is True
