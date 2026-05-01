"""Hermes-style ``<tool_call>{json}</tool_call>`` parser.

Used by Hermes-2-Pro, Qwen2.5-Instruct, and a large family of NousResearch /
community fine-tunes whose chat templates wrap each tool call in the literal
tags ``<tool_call>`` / ``</tool_call>`` around a JSON object of the shape
``{"name": "...", "arguments": {...}}``.
"""

from __future__ import annotations

import json
import re

from modelship.openai.protocol import FunctionCall, ToolCall
from modelship.openai.tool_calling.parsers.base import ParsedToolCalls, ToolCallParser

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class HermesToolCallParser(ToolCallParser):
    name = "hermes"

    def parse(self, text: str) -> ParsedToolCalls:
        matches = list(_TOOL_CALL_RE.finditer(text))
        if not matches:
            return ParsedToolCalls(content=text, tool_calls=[])

        tool_calls: list[ToolCall] = []
        for m in matches:
            try:
                payload = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                # Malformed block — leave the text in residual, skip this call.
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                continue
            arguments = payload.get("arguments", {})
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments)
            tool_calls.append(ToolCall(function=FunctionCall(name=name, arguments=arguments)))

        if not tool_calls:
            return ParsedToolCalls(content=text, tool_calls=[])

        residual = _TOOL_CALL_RE.sub("", text).strip()
        return ParsedToolCalls(content=residual or None, tool_calls=tool_calls)
