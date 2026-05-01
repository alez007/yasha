"""Tests for the cross-loader tool-calling toolkit."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from modelship.openai.tool_calling import (
    available_parsers,
    get_parser,
    register_parser,
    resolve_tools_for_request,
)
from modelship.openai.tool_calling.parsers import HermesToolCallParser, ParsedToolCalls, ToolCallParser


class TestRegistry:
    def test_default_registry_includes_hermes(self):
        assert "hermes" in available_parsers()

    def test_get_parser_returns_singleton(self):
        a = get_parser("hermes")
        b = get_parser("hermes")
        assert a is b

    def test_unknown_parser_raises_with_available_list(self):
        with pytest.raises(ValueError, match="hermes"):
            get_parser("does-not-exist")

    def test_register_parser_makes_it_findable(self):
        class Stub(ToolCallParser):
            name = "stub-test-parser"

            def parse(self, text: str) -> ParsedToolCalls:
                return ParsedToolCalls(content=text, tool_calls=[])

        register_parser(Stub())
        try:
            assert get_parser("stub-test-parser").name == "stub-test-parser"
        finally:
            # Clean up so other tests don't see the stub.
            from modelship.openai.tool_calling import registry

            registry._PARSERS.pop("stub-test-parser", None)


class TestHermesParser:
    parser = HermesToolCallParser()

    def test_no_tool_calls_returns_text_unchanged(self):
        result = self.parser.parse("just a regular response")
        assert result.tool_calls == []
        assert result.content == "just a regular response"
        assert result.has_tool_calls is False

    def test_single_tool_call(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].function.name == "get_weather"
        assert json.loads(result.tool_calls[0].function.arguments) == {"city": "Paris"}
        assert result.content is None

    def test_multiple_tool_calls(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {"x": 1}}</tool_call>'
            '<tool_call>{"name": "b", "arguments": {"y": 2}}</tool_call>'
        )
        result = self.parser.parse(text)
        assert [tc.function.name for tc in result.tool_calls] == ["a", "b"]

    def test_tool_call_with_residual_text(self):
        text = 'Sure, calling that.\n<tool_call>{"name": "ping", "arguments": {}}</tool_call>'
        result = self.parser.parse(text)
        assert len(result.tool_calls) == 1
        assert result.content == "Sure, calling that."

    def test_string_arguments_passed_through(self):
        text = '<tool_call>{"name": "x", "arguments": "{\\"a\\": 1}"}</tool_call>'
        result = self.parser.parse(text)
        assert result.tool_calls[0].function.arguments == '{"a": 1}'

    def test_object_arguments_serialized_to_json(self):
        text = '<tool_call>{"name": "x", "arguments": {"a": 1, "b": [2, 3]}}</tool_call>'
        result = self.parser.parse(text)
        assert json.loads(result.tool_calls[0].function.arguments) == {"a": 1, "b": [2, 3]}

    def test_malformed_json_block_drops_call_and_falls_back_to_content(self):
        text = "<tool_call>{not valid json}</tool_call>"
        result = self.parser.parse(text)
        assert result.tool_calls == []
        # Malformed block stays in content as-is.
        assert result.content == text

    def test_missing_name_drops_call(self):
        text = '<tool_call>{"arguments": {}}</tool_call>'
        result = self.parser.parse(text)
        assert result.tool_calls == []

    def test_empty_name_drops_call(self):
        text = '<tool_call>{"name": "", "arguments": {}}</tool_call>'
        result = self.parser.parse(text)
        assert result.tool_calls == []

    def test_each_tool_call_gets_unique_id(self):
        text = (
            '<tool_call>{"name": "a", "arguments": {}}</tool_call><tool_call>{"name": "b", "arguments": {}}</tool_call>'
        )
        result = self.parser.parse(text)
        assert result.tool_calls[0].id != result.tool_calls[1].id


class TestResolveToolsForRequest:
    tools: ClassVar = [
        {"type": "function", "function": {"name": "alpha"}},
        {"type": "function", "function": {"name": "beta"}},
    ]

    def test_no_tools_returns_none(self):
        assert resolve_tools_for_request(None, "auto") is None
        assert resolve_tools_for_request([], "auto") is None

    def test_auto_passes_through(self):
        assert resolve_tools_for_request(self.tools, "auto") == self.tools

    def test_unset_tool_choice_passes_through(self):
        assert resolve_tools_for_request(self.tools, None) == self.tools

    def test_none_suppresses_tools(self):
        assert resolve_tools_for_request(self.tools, "none") is None

    def test_required_passes_through(self):
        # We cannot strictly enforce a tool call without constrained decoding,
        # so "required" downgrades to "auto" semantics (with a logged warning).
        assert resolve_tools_for_request(self.tools, "required") == self.tools

    def test_specific_function_filters_to_that_tool(self):
        result = resolve_tools_for_request(self.tools, {"type": "function", "function": {"name": "beta"}})
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "beta"

    def test_unknown_function_falls_back_to_all(self):
        # If the named function isn't in the tools list, fall back to passing
        # them all through rather than emitting an empty list.
        result = resolve_tools_for_request(self.tools, {"type": "function", "function": {"name": "missing"}})
        assert result == self.tools

    def test_unrecognized_choice_falls_back_to_all(self):
        result = resolve_tools_for_request(self.tools, "weird-mode")
        assert result == self.tools
