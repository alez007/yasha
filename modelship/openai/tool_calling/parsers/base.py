"""Base class for model-family-specific tool-call output parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from modelship.openai.protocol import ToolCall


@dataclass(frozen=True)
class ParsedToolCalls:
    """Result of running a parser over a model's raw chat-completion text.

    ``content`` carries the residual non-tool-call text once any tool-call
    markers are stripped. It is ``None`` when tool calls were extracted *and*
    the residual is empty, matching OpenAI's behavior of nulling ``content``
    alongside ``tool_calls``.
    """

    content: str | None
    tool_calls: list[ToolCall]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class ToolCallParser(ABC):
    """Convert a raw text generation into OpenAI-shape tool calls.

    Each subclass targets one model-family output convention (Hermes XML tags,
    Llama 3.1 ``<|python_tag|>``, Mistral ``[TOOL_CALLS]``, …). Implementations
    must be pure functions of the input text — no model state, no side effects
    — so the same parser can be reused across loaders, deployments, and tests.
    """

    name: str

    @abstractmethod
    def parse(self, text: str) -> ParsedToolCalls:
        """Extract tool calls and residual content from ``text``."""
