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


class YashaJsonFormatter(logging.Formatter):
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


class YashaTextFormatter(logging.Formatter):
    def format(self, record):
        ts = self.formatTime(record, self.datefmt)
        req_id = getattr(record, "request_id", None)
        req_part = f" [{req_id}]" if req_id else ""
        base = f"[{ts}] {record.levelname:<8} {record.name}{req_part} | {record.getMessage()}"
        if record.exc_info and record.exc_info[1] is not None:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("YASHA_LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("YASHA_LOG_FORMAT", "text").lower()

    level = getattr(logging, level_name, logging.INFO)

    root_logger = logging.getLogger("yasha")
    root_logger.setLevel(level)
    root_logger.propagate = False

    if root_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setLevel(level)

    if log_format == "json":
        handler.setFormatter(YashaJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(YashaTextFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    handler.addFilter(RequestIdFilter())
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"yasha.{name}")
