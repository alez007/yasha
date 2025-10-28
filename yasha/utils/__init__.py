from typing import Optional
from fastapi import Request
import uuid

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def base_request_id(raw_request: Optional[Request], default: Optional[str] = None) -> Optional[str]:
    """Pulls the request id to use from a header, if provided"""
    default = default or random_uuid()
    if raw_request is None:
        return default

