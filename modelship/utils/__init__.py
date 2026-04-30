import os
import random
import string
import uuid

import requests

from modelship.infer.infer_config import RawRequestProxy

_RAND_CHARS = string.ascii_lowercase + string.digits


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def rand_suffix(length: int = 5) -> str:
    return "".join(random.choices(_RAND_CHARS, k=length))


def base_request_id(raw_request: RawRequestProxy | None = None) -> str:
    """Return the request ID from a RawRequestProxy, or generate a new one."""
    if raw_request is not None and raw_request.request_id is not None:
        return raw_request.request_id
    return random_uuid()


def download(url: str, file_path: str, overwrite: bool = False):
    if os.path.isfile(file_path) is False:
        get_response = requests.get(url, stream=True)
        with open(file_path, "wb") as f:
            for chunk in get_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)


def cache_dir() -> str:
    path = os.environ.get("MSHIP_CACHE_DIR", "/.cache")
    os.makedirs(path, exist_ok=True)
    return path


def plugins_dir() -> str:
    path = f"{cache_dir()}/plugins"
    os.makedirs(path, exist_ok=True)
    return path
