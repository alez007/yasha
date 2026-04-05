import os
import uuid

import requests
from fastapi import Request


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def base_request_id(raw_request: Request | None, default: str | None = None) -> str | None:
    """Pulls the request id to use from a header, if provided"""
    default = default or random_uuid()
    if raw_request is None:
        return default


def download(url: str, file_path: str, overwrite: bool = False):
    if os.path.isfile(file_path) is False:
        get_response = requests.get(url, stream=True)
        with open(file_path, "wb") as f:
            for chunk in get_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)


def cache_dir() -> str:
    path = os.environ.get("YASHA_CACHE_DIR", "/yasha/.cache/models")
    os.makedirs(path, exist_ok=True)
    return path
