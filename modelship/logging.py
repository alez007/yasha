import contextvars
import json
import logging
import os
import socket
from logging.handlers import SysLogHandler
from urllib.parse import urlparse

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

_LIB_LOGGERS = ("ray", "ray.serve", "vllm", "transformers", "diffusers", "llama_cpp")

# Env vars that libraries check internally when creating their own loggers.
# Setting these ensures the level sticks even when a library re-configures
# its loggers after our configure_logging() call (e.g. vLLM's init_logger).
_LIB_ENV_VARS = {
    "RAY_LOG_LEVEL": "ray",
    "VLLM_LOGGING_LEVEL": "vllm",
    "TRANSFORMERS_VERBOSITY": "transformers",
}


def _parse_syslog_target(target: str) -> SysLogHandler:
    """Parse a syslog URI and return a configured SysLogHandler.

    Supported formats:
        syslog://host:port       — UDP (default)
        syslog+tcp://host:port   — TCP
        syslog://host            — UDP, port 514
    """
    tcp = target.startswith("syslog+tcp://")
    url = target.replace("syslog+tcp://", "syslog://", 1)
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 514
    socktype = socket.SOCK_STREAM if tcp else socket.SOCK_DGRAM
    return SysLogHandler(address=(host, port), socktype=socktype)


def configure_logging() -> None:
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.environ.get("MSHIP_LOG_LEVEL", "INFO").upper()
    log_format = os.environ.get("MSHIP_LOG_FORMAT", "text").lower()
    log_target = os.environ.get("MSHIP_LOG_TARGET", "console").lower()

    levels = [TRACE, logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]
    app_level = TRACE if level_name == "TRACE" else getattr(logging, level_name, logging.INFO)
    lib_level = next((lv for lv in levels if lv > app_level), logging.CRITICAL)
    lib_level_name = logging.getLevelName(lib_level)

    root_logger = logging.getLogger("modelship")
    root_logger.setLevel(app_level)
    root_logger.propagate = False

    if root_logger.handlers:
        return

    handler = _parse_syslog_target(log_target) if log_target.startswith("syslog") else logging.StreamHandler()
    handler.setLevel(app_level)

    if log_format == "json":
        handler.setFormatter(ModelshipJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
    else:
        handler.setFormatter(ModelshipTextFormatter(datefmt="%Y-%m-%d %H:%M:%S"))

    handler.addFilter(RequestIdFilter())
    root_logger.addHandler(handler)

    # OpenTelemetry: add an OTLP log exporter as a second handler when configured.
    otel_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otel_endpoint:
        _setup_otel(root_logger, otel_endpoint, app_level)

    # Set library log levels via both the Python logger (immediate effect) and
    # the library's native env var (so the level sticks when the library
    # re-configures its own loggers later, e.g. vLLM's init_logger).
    for name in _LIB_LOGGERS:
        logging.getLogger(name).setLevel(lib_level)
    for env_var, lib_name in _LIB_ENV_VARS.items():
        val = lib_level_name.lower() if lib_name == "transformers" else lib_level_name
        os.environ.setdefault(env_var, val)


# pyright: reportMissingImports=false
def _setup_otel(root_logger: logging.Logger, endpoint: str, level: int) -> None:
    """Attach an OpenTelemetry log exporter to *root_logger*.

    Requires the ``opentelemetry-sdk`` and ``opentelemetry-exporter-otlp``
    packages (install via ``uv sync --extra otel``).
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.log_exporter import (
            OTLPLogExporter,
        )
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    except ImportError:
        root_logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry packages are not installed. "
            "Install with: uv sync --extra otel"
        )
        return

    resource = Resource.create({SERVICE_NAME: "modelship"})
    provider = LoggerProvider(resource=resource)
    exporter = OTLPLogExporter(endpoint=endpoint, insecure=not endpoint.startswith("https"))
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

    otel_handler = LoggingHandler(level=level, logger_provider=provider)
    root_logger.addHandler(otel_handler)

    # Enable Ray's native OTel tracing so spans from Ray workers are exported too.
    os.environ.setdefault("RAY_TRACING_ENABLED", "1")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"modelship.{name}")
