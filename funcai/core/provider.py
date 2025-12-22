"""Provider abstraction for LLM APIs."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from kungfu import Result, Option, Nothing
from pydantic import BaseModel

from funcai.core.message import Message
from funcai.core.types import ToolCall

if TYPE_CHECKING:
    from funcai.agents.tool import Tool


@dataclass
class AIResponse[S]:
    """Response from LLM."""

    message: Message
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    parsed: Option[S] = field(default_factory=Nothing)
    meta: dict[str, Any] = field(default_factory=dict[str, Any])

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class ABCAIProvider[E](ABC):
    """Abstract LLM provider."""

    @abstractmethod
    async def send_messages[S: BaseModel](
        self,
        messages: list[Message],
        *,
        schema: Option[type[S]] = Nothing(),
        tools: list["Tool"] | None = None,
    ) -> Result[AIResponse[S], E]:
        """
        Send messages to LLM.

        Args:
            messages: Conversation history
            schema: Optional Pydantic model for structured output
            tools: Optional list of tools LLM can call

        Returns:
            AIResponse with message, tool_calls, and optional parsed output
        """
        ...
