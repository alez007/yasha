"""Tests for API key authentication middleware."""

import os
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import JSONResponse

from yasha.openai.auth import ApiKeyMiddleware, get_api_keys


def _make_app(api_keys: set[str]) -> FastAPI:
    """Build a minimal FastAPI app with the auth middleware for testing."""
    app = FastAPI()
    app.add_middleware(ApiKeyMiddleware, api_keys=api_keys)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list", "data": []}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return JSONResponse({"id": "test"})

    return app


VALID_KEY = "sk-test-key-123"
OTHER_KEY = "sk-other-key-456"
KEYS = {VALID_KEY, OTHER_KEY}


class TestApiKeyMiddleware:
    def test_valid_key_allows_request(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {VALID_KEY}"})
        assert resp.status_code == 200

    def test_other_valid_key_allows_request(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": f"Bearer {OTHER_KEY}"})
        assert resp.status_code == 200

    def test_missing_auth_header_returns_401(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models")
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["error"]["message"]

    def test_empty_bearer_returns_401(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": "Bearer "})
        assert resp.status_code == 401
        assert "Missing API key" in resp.json()["error"]["message"]

    def test_invalid_key_returns_401(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
        assert resp.status_code == 401
        assert "Invalid API key" in resp.json()["error"]["message"]

    def test_non_bearer_auth_returns_401(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models", headers={"Authorization": f"Basic {VALID_KEY}"})
        assert resp.status_code == 401

    def test_health_endpoint_bypasses_auth(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_post_endpoint_requires_auth(self):
        client = TestClient(_make_app(KEYS))
        resp = client.post("/v1/chat/completions", json={})
        assert resp.status_code == 401

    def test_post_endpoint_with_valid_key(self):
        client = TestClient(_make_app(KEYS))
        resp = client.post(
            "/v1/chat/completions",
            json={},
            headers={"Authorization": f"Bearer {VALID_KEY}"},
        )
        assert resp.status_code == 200

    def test_error_response_format(self):
        client = TestClient(_make_app(KEYS))
        resp = client.get("/v1/models")
        body = resp.json()
        assert "error" in body
        assert body["error"]["type"] == "auth_error"
        assert body["error"]["code"] == 401


class TestGetApiKeys:
    def test_returns_keys_from_env(self):
        with patch.dict(os.environ, {"YASHA_API_KEYS": "sk-a,sk-b,sk-c"}):
            keys = get_api_keys()
        assert keys == {"sk-a", "sk-b", "sk-c"}

    def test_strips_whitespace(self):
        with patch.dict(os.environ, {"YASHA_API_KEYS": " sk-a , sk-b "}):
            keys = get_api_keys()
        assert keys == {"sk-a", "sk-b"}

    def test_empty_env_returns_empty_set(self):
        with patch.dict(os.environ, {"YASHA_API_KEYS": ""}):
            keys = get_api_keys()
        assert keys == set()

    def test_unset_env_returns_empty_set(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = get_api_keys()
        assert keys == set()

    def test_ignores_empty_entries(self):
        with patch.dict(os.environ, {"YASHA_API_KEYS": "sk-a,,,,sk-b,"}):
            keys = get_api_keys()
        assert keys == {"sk-a", "sk-b"}
