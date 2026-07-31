"""Tests for modelship.openai.mcp.spec: parsing and policy for the Responses
``mcp`` tool type. No I/O — everything here is pure request-shape logic."""

import pytest

from modelship.openai.mcp.spec import (
    McpToolSpec,
    check_collisions,
    filter_tools,
    requires_approval,
    split_mcp_tools,
)
from modelship.openai.protocol.responses.schemas import McpListToolsTool
from modelship.openai.utils.responses import ResponsesApiError


def _tool(name: str, *, read_only: bool = False) -> McpListToolsTool:
    annotations = {"readOnlyHint": True} if read_only else None
    return McpListToolsTool(name=name, input_schema={"type": "object"}, annotations=annotations)


class TestSplitMcpTools:
    def test_no_tools_returns_empty(self):
        assert split_mcp_tools(None) == ([], [])
        assert split_mcp_tools([]) == ([], [])

    def test_function_tools_pass_through_untouched(self):
        fn = {"type": "function", "name": "f"}
        specs, other = split_mcp_tools([fn])
        assert specs == []
        assert other == [fn]

    def test_mcp_tool_parsed_into_spec(self):
        specs, other = split_mcp_tools(
            [{"type": "mcp", "server_label": "s", "server_url": "http://x", "headers": {"a": "b"}}]
        )
        assert other == []
        assert len(specs) == 1
        assert specs[0] == McpToolSpec(server_label="s", server_url="http://x", headers={"a": "b"})

    def test_missing_server_url_rejected(self):
        with pytest.raises(ResponsesApiError, match="server_url"):
            split_mcp_tools([{"type": "mcp", "server_label": "s"}])

    def test_missing_server_label_rejected(self):
        with pytest.raises(ResponsesApiError, match="server_label"):
            split_mcp_tools([{"type": "mcp", "server_url": "http://x"}])

    def test_duplicate_server_label_rejected(self):
        tools = [
            {"type": "mcp", "server_label": "s", "server_url": "http://a"},
            {"type": "mcp", "server_label": "s", "server_url": "http://b"},
        ]
        with pytest.raises(ResponsesApiError, match="duplicate"):
            split_mcp_tools(tools)

    def test_connector_id_rejected(self):
        with pytest.raises(ResponsesApiError, match="connector_id"):
            split_mcp_tools([{"type": "mcp", "server_label": "s", "connector_id": "c1"}])

    def test_non_string_server_url_rejected_as_responses_api_error(self):
        with pytest.raises(ResponsesApiError, match="invalid mcp tool"):
            split_mcp_tools([{"type": "mcp", "server_label": "s", "server_url": 123}])

    def test_non_dict_headers_rejected_as_responses_api_error(self):
        with pytest.raises(ResponsesApiError, match="invalid mcp tool"):
            split_mcp_tools(
                [{"type": "mcp", "server_label": "s", "server_url": "http://x", "headers": ["not", "a", "dict"]}]
            )


class TestPolicyShapesRejectedAtParseTime:
    """Rejected by split_mcp_tools, not by filter_tools/requires_approval — those first
    run mid-stream, where a raise can't become a 400."""

    @staticmethod
    def _split(**extra):
        return split_mcp_tools([{"type": "mcp", "server_label": "s", "server_url": "http://x", **extra}])

    @pytest.mark.parametrize("allowed", ["roll", 5, True])
    def test_unsupported_allowed_tools_shape(self, allowed):
        with pytest.raises(ResponsesApiError, match="allowed_tools"):
            self._split(allowed_tools=allowed)

    def test_allowed_tools_list_with_non_string_elements(self):
        with pytest.raises(ResponsesApiError, match="allowed_tools"):
            self._split(allowed_tools=["ok", 7])

    def test_allowed_tools_dict_with_string_tool_names(self):
        with pytest.raises(ResponsesApiError, match=r"allowed_tools\.tool_names"):
            self._split(allowed_tools={"tool_names": "roll"})

    @pytest.mark.parametrize("setting", ["sometimes", 5, ["always"]])
    def test_unsupported_require_approval_shape(self, setting):
        with pytest.raises(ResponsesApiError, match="require_approval"):
            self._split(require_approval=setting)

    def test_require_approval_non_dict_bucket(self):
        with pytest.raises(ResponsesApiError, match="require_approval"):
            self._split(require_approval={"never": "roll"})

    def test_require_approval_dict_with_string_tool_names(self):
        with pytest.raises(ResponsesApiError, match=r"require_approval\.never\.tool_names"):
            self._split(require_approval={"never": {"tool_names": "roll"}})

    def test_valid_shapes_still_accepted(self):
        specs, _ = self._split(
            allowed_tools={"tool_names": ["roll"], "read_only": True},
            require_approval={"never": {"tool_names": ["roll"]}, "always": {"read_only": False}},
        )
        assert len(specs) == 1
        specs, _ = self._split(allowed_tools=["roll"], require_approval="never")
        assert len(specs) == 1


class TestFilterTools:
    def test_no_allowed_tools_returns_all(self):
        spec = McpToolSpec(server_label="s", server_url="http://x")
        tools = [_tool("a"), _tool("b")]
        assert filter_tools(spec, tools) == tools

    def test_list_form_filters_by_name(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", allowed_tools=["a"])
        tools = [_tool("a"), _tool("b")]
        assert [t.name for t in filter_tools(spec, tools)] == ["a"]

    def test_dict_tool_names_form(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", allowed_tools={"tool_names": ["b"]})
        tools = [_tool("a"), _tool("b")]
        assert [t.name for t in filter_tools(spec, tools)] == ["b"]

    def test_dict_read_only_form(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", allowed_tools={"read_only": True})
        tools = [_tool("a", read_only=True), _tool("b", read_only=False)]
        assert [t.name for t in filter_tools(spec, tools)] == ["a"]

    def test_unsupported_shape_rejected(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", allowed_tools="bogus")
        with pytest.raises(ResponsesApiError):
            filter_tools(spec, [_tool("a")])

    def test_list_form_with_non_string_elements_rejected(self):
        # A list of non-strings would otherwise silently produce a set no tool.name
        # (always a str) can ever match, filtering everything out without a clear 400.
        spec = McpToolSpec(server_label="s", server_url="http://x", allowed_tools=[1, 2])
        with pytest.raises(ResponsesApiError, match="allowed_tools"):
            filter_tools(spec, [_tool("a")])

    def test_string_tool_names_rejected_not_treated_as_char_iterable(self):
        # A bare string is itself iterable, so set("roll") would silently become
        # {"r", "o", "l"} and change matching semantics instead of erroring.
        spec = McpToolSpec(server_label="s", server_url="http://x", allowed_tools={"tool_names": "roll"})
        with pytest.raises(ResponsesApiError, match="tool_names"):
            filter_tools(spec, [_tool("r"), _tool("roll")])


class TestRequiresApproval:
    def test_unset_defaults_to_always(self):
        spec = McpToolSpec(server_label="s", server_url="http://x")
        assert requires_approval(spec, _tool("a")) is True

    def test_explicit_always(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", require_approval="always")
        assert requires_approval(spec, _tool("a")) is True

    def test_explicit_never(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", require_approval="never")
        assert requires_approval(spec, _tool("a")) is False

    def test_dict_never_tool_names(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", require_approval={"never": {"tool_names": ["a"]}})
        assert requires_approval(spec, _tool("a")) is False
        assert requires_approval(spec, _tool("b")) is True

    def test_dict_always_tool_names_overrides_default_never_bucket(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", require_approval={"always": {"tool_names": ["a"]}})
        assert requires_approval(spec, _tool("a")) is True
        # Tools not named in either bucket still default to requiring approval.
        assert requires_approval(spec, _tool("b")) is True

    def test_dict_never_read_only(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", require_approval={"never": {"read_only": True}})
        assert requires_approval(spec, _tool("a", read_only=True)) is False
        assert requires_approval(spec, _tool("b", read_only=False)) is True

    def test_unsupported_shape_rejected(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", require_approval=123)
        with pytest.raises(ResponsesApiError):
            requires_approval(spec, _tool("a"))

    def test_non_dict_bucket_rejected(self):
        spec = McpToolSpec(server_label="s", server_url="http://x", require_approval={"never": "bogus"})
        with pytest.raises(ResponsesApiError):
            requires_approval(spec, _tool("a"))

    def test_string_tool_names_rejected_not_treated_as_substring_membership(self):
        # A bare string "roll" is itself iterable, so `"o" in "roll"` is True even
        # though "o" was never listed as a tool name — this must 400, not bypass approval.
        spec = McpToolSpec(server_label="s", server_url="http://x", require_approval={"never": {"tool_names": "roll"}})
        with pytest.raises(ResponsesApiError, match="tool_names"):
            requires_approval(spec, _tool("o"))


class TestCheckCollisions:
    def test_no_collision_is_fine(self):
        spec_a = McpToolSpec(server_label="a", server_url="http://a")
        spec_b = McpToolSpec(server_label="b", server_url="http://b")
        discovered = {"a": [_tool("x")], "b": [_tool("y")]}
        check_collisions([spec_a, spec_b], discovered, [])

    def test_two_servers_exposing_same_name_collide(self):
        spec_a = McpToolSpec(server_label="a", server_url="http://a")
        spec_b = McpToolSpec(server_label="b", server_url="http://b")
        discovered = {"a": [_tool("x")], "b": [_tool("x")]}
        with pytest.raises(ResponsesApiError, match="both"):
            check_collisions([spec_a, spec_b], discovered, [])

    def test_collision_with_client_tool(self):
        spec_a = McpToolSpec(server_label="a", server_url="http://a")
        discovered = {"a": [_tool("x")]}
        with pytest.raises(ResponsesApiError, match="client tool"):
            check_collisions([spec_a], discovered, [{"type": "function", "name": "x"}])
