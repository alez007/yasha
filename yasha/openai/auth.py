import hmac
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from yasha.logging import get_logger
from yasha.metrics import AUTH_FAILURES_TOTAL

logger = get_logger("api.auth")

_PUBLIC_PATHS = {"/health"}


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Validates ``Authorization: Bearer <key>`` against a set of allowed API keys."""

    def __init__(self, app, api_keys: set[str]):
        super().__init__(app)
        self.api_keys = api_keys

    async def dispatch(self, request: Request, call_next):
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""

        if not token:
            AUTH_FAILURES_TOTAL.inc(tags={"reason": "missing"})
            logger.warning("auth failed (missing key): %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Missing API key. Use Authorization: Bearer <key>.",
                        "type": "auth_error",
                        "code": 401,
                    }
                },
            )

        if not any(hmac.compare_digest(token, key) for key in self.api_keys):
            AUTH_FAILURES_TOTAL.inc(tags={"reason": "invalid"})
            logger.warning("auth failed (invalid key): %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid API key.", "type": "auth_error", "code": 401}},
            )

        return await call_next(request)


def get_api_keys() -> set[str]:
    """Read allowed API keys from the ``YASHA_API_KEYS`` environment variable (comma-separated)."""
    raw = os.environ.get("YASHA_API_KEYS", "")
    return {k.strip() for k in raw.split(",") if k.strip()}
