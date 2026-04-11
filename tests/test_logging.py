"""Tests for the yasha.logging module."""

import json
import logging
import os
from unittest.mock import patch

import pytest

from yasha.logging import (
    _LIB_ENV_VARS,
    _LIB_LOGGERS,
    RequestIdFilter,
    YashaJsonFormatter,
    YashaTextFormatter,
    configure_logging,
    get_logger,
    request_id_var,
)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Reset the yasha logger and _configured flag between tests."""
    import yasha.logging as yl

    yl._configured = False
    root = logging.getLogger("yasha")
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    root.propagate = True
    saved_lib_levels = {name: logging.getLogger(name).level for name in _LIB_LOGGERS}
    saved_env = {k: os.environ.get(k) for k in _LIB_ENV_VARS}
    token = request_id_var.set(None)
    yield
    request_id_var.reset(token)
    yl._configured = False
    root.handlers.clear()
    for name, lvl in saved_lib_levels.items():
        logging.getLogger(name).setLevel(lvl)
    for k, v in saved_env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


class TestGetLogger:
    def test_returns_yasha_prefixed_logger(self):
        log = get_logger("api")
        assert log.name == "yasha.api"

    def test_nested_name(self):
        log = get_logger("infer.vllm")
        assert log.name == "yasha.infer.vllm"


class TestConfigureLogging:
    def test_sets_up_handler(self):
        configure_logging()
        root = logging.getLogger("yasha")
        assert len(root.handlers) == 1
        assert root.level == logging.INFO
        assert root.propagate is False

    def test_idempotent(self):
        configure_logging()
        configure_logging()
        root = logging.getLogger("yasha")
        assert len(root.handlers) == 1

    @patch.dict(os.environ, {"YASHA_LOG_LEVEL": "DEBUG"})
    def test_respects_log_level_env(self):
        configure_logging()
        root = logging.getLogger("yasha")
        assert root.level == logging.DEBUG

    def test_lib_loggers_default_to_warning(self):
        configure_logging()
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING
        for env_var in _LIB_ENV_VARS:
            assert os.environ.get(env_var) == "WARNING"

    @patch.dict(os.environ, {"YASHA_LOG_LEVEL": "DEBUG"})
    def test_lib_loggers_stay_warning_at_debug(self):
        configure_logging()
        assert logging.getLogger("yasha").level == logging.DEBUG
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == logging.WARNING

    @patch.dict(os.environ, {"YASHA_LOG_LEVEL": "TRACE"})
    def test_trace_mode_enables_all_debug(self):
        configure_logging()
        assert logging.getLogger("yasha").level == logging.DEBUG
        for name in _LIB_LOGGERS:
            assert logging.getLogger(name).level == logging.DEBUG
        for env_var in _LIB_ENV_VARS:
            assert os.environ.get(env_var) == "DEBUG"

    @patch.dict(os.environ, {"YASHA_LOG_FORMAT": "json"})
    def test_json_format(self):
        configure_logging()
        root = logging.getLogger("yasha")
        handler = root.handlers[0]
        assert isinstance(handler.formatter, YashaJsonFormatter)

    def test_text_format_default(self):
        configure_logging()
        root = logging.getLogger("yasha")
        handler = root.handlers[0]
        assert isinstance(handler.formatter, YashaTextFormatter)


class TestRequestIdFilter:
    def test_injects_request_id(self):
        filt = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        request_id_var.set("abc-123")
        filt.filter(record)
        assert record.request_id == "abc-123"

    def test_none_when_not_set(self):
        filt = RequestIdFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        filt.filter(record)
        assert record.request_id is None


class TestYashaJsonFormatter:
    def test_produces_valid_json(self):
        formatter = YashaJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("yasha.api", logging.INFO, "", 0, "hello %s", ("world",), None)
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "yasha.api"
        assert parsed["message"] == "hello world"
        assert "pid" in parsed

    def test_includes_request_id(self):
        formatter = YashaJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("yasha.api", logging.INFO, "", 0, "test", (), None)
        record.request_id = "req-456"
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["request_id"] == "req-456"

    def test_excludes_request_id_when_none(self):
        formatter = YashaJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        record = logging.LogRecord("yasha.api", logging.INFO, "", 0, "test", (), None)
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "request_id" not in parsed

    def test_includes_exception(self):
        formatter = YashaJsonFormatter(datefmt="%Y-%m-%dT%H:%M:%S")
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord("yasha.api", logging.ERROR, "", 0, "error", (), sys.exc_info())
        record.request_id = None
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert "ValueError: boom" in parsed["exception"]


class TestYashaTextFormatter:
    def test_basic_format(self):
        formatter = YashaTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("yasha.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = None
        output = formatter.format(record)
        assert "INFO" in output
        assert "yasha.api" in output
        assert "hello" in output

    def test_includes_request_id(self):
        formatter = YashaTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("yasha.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = "req-789"
        output = formatter.format(record)
        assert "[req-789]" in output

    def test_excludes_request_id_when_none(self):
        formatter = YashaTextFormatter(datefmt="%Y-%m-%d %H:%M:%S")
        record = logging.LogRecord("yasha.api", logging.INFO, "", 0, "hello", (), None)
        record.request_id = None
        output = formatter.format(record)
        assert "None" not in output


class TestEndToEnd:
    def test_log_output_with_request_id(self, capsys):
        configure_logging()
        log = get_logger("test")
        request_id_var.set("e2e-test-id")
        log.info("end to end")
        captured = capsys.readouterr()
        assert "e2e-test-id" in captured.err
        assert "end to end" in captured.err

    @patch.dict(os.environ, {"YASHA_LOG_FORMAT": "json"})
    def test_json_log_output(self, capsys):
        configure_logging()
        log = get_logger("test")
        request_id_var.set("json-test-id")
        log.info("json test")
        captured = capsys.readouterr()
        parsed = json.loads(captured.err)
        assert parsed["request_id"] == "json-test-id"
        assert parsed["message"] == "json test"
