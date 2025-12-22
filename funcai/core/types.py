"""Core types for funcai."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """A tool call from the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """Result of executing a tool."""

    tool_call_id: str
    content: str
