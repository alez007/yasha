from typing import Optional
from fastapi import Request
import uuid
import os
import requests

def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def base_request_id(raw_request: Optional[Request], default: Optional[str] = None) -> Optional[str]:
    """Pulls the request id to use from a header, if provided"""
    default = default or random_uuid()
    if raw_request is None:
        return default

def download(url: str, file_path: str, overwrite: bool = False):
    if os.path.isfile(file_path) is False:
        get_response = requests.get(url, stream=True)
        with open(file_path, 'wb') as f:
            for chunk in get_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

