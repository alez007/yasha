import contextvars
import json
import logging
import os

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

_configured = False


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        record.request_id = request_id_var.get(None)
        return True


class ModelshipJsonFormatter(logging.Formatter):
    def format(self, record):
        msg = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pid": record.process,
        }
        req_id = getattr(record, "request_id", None)
        if req_id:
            msg["request_id"] = req_id
        if record.exc_info and record.exc_info[1] is not None:
            msg["exception"] = self.formatException(record.exc_info)
        return json.dumps(msg)


class ModelshipTextFormatter(logging.Formatter):
    def format(self, record):
        ts = self.formatTime(record, self.datefmt)
        req_id = getattr(record, "request_id", None)
        req_part = f" [{req_id}]" if req_id else ""
        base = f"[{ts}] {record.levelname:<8} {record.name}{req_part} | {record.getMessage()}"
        if record.exc_info and record.exc_info[1] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


TRACE = 5
logging.addLevelName(TRACE, "TRACE")

_LIB_LOGGERS = ("ray", "ray.serve", "vllm", "transformers", "diffusers")

# Env vars that libraries check internally when creating their own loggers.
# Setting these ensures the level sticks even when a library re-configures
# its loggers after our configure_logging() call (e.g. vLLM's init_logger).
_LIB_ENV_VARS = {
    "RAY_LOG_LEVEL": "ray",
    "VLLM_LOGGING_LEVEL": "vllm",
    "TRANSFORMERS_VERBOSITY": "transformers",
}


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("MSHIP_LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("MSHIP_LOG_FORMAT", "text").lower()

    trace_mode = level_name == "TRACE"
    app_level = logging.DEBUG if trace_mode else getattr(logging, level_name, logging.INFO)
    lib_level = logging.DEBUG if trace_mode else logging.WARNING
    lib_level_name = logging.getLevelName(lib_level)

    root_logger = logging.getLogger("modelship")
    root_logger.setLevel(app_level)
    root_logger.propagate = False

    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(app_level)

    if log_format == "json":
        handler.setFormatter(ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    handler.addFilter(RequestIdFilter())
    root_logger.addHandler(handler)

    # Set library log levels via both the Python logger (immediate effect) and
    # the library's native env var (so the level sticks when the library
    # re-configures its own loggers later, e.g. vLLM's init_logger).
    for name in _LIB_LOGGERS:
        logging.getLogger(name).setLevel(lib_level)
    for env_var in _LIB_ENV_VARS:
        os.environ.setdefault(env_var, lib_level_name)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"modelship.{name}")
