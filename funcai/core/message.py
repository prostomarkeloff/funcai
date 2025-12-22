"""Message types for dialogues."""

from dataclasses import dataclass, field
from enum import StrEnum

from kungfu import Option, Nothing, Some, from_optional

from funcai.core.media import Media
from funcai.core.types import ToolCall


class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class Message:
    """A message in a dialogue."""

    role: Role
    text: Option[str] = field(default_factory=Nothing)
    media: list[Media] = field(default_factory=list[Media])

    tool_call_id: Option[str] = field(default_factory=Nothing)
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])

    def __post_init__(self) -> None:
        if self.role == Role.TOOL:
            match self.tool_call_id:
                case Nothing():
                    raise ValueError("Tool message requires tool_call_id")
                case _:
                    pass
            return
        # Assistant with tool calls is valid even without text
        if self.role == Role.ASSISTANT and self.tool_calls:
            return
        match self.text:
            case Nothing() if not self.media:
                raise ValueError("Message must contain text or media")
            case _:
                pass

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


def system(*, text: str) -> Message:
    return Message(role=Role.SYSTEM, text=Some(text))


def user(*, text: str | None = None, media: list[Media] | None = None) -> Message:
    return Message(role=Role.USER, text=from_optional(text), media=media or [])


def assistant(
    *,
    text: str | None = None,
    tool_calls: list[ToolCall] | None = None,
) -> Message:
    return Message(
        role=Role.ASSISTANT, text=from_optional(text), tool_calls=tool_calls or []
    )


def tool_result(*, tool_call_id: str, content: str) -> Message:
    return Message(role=Role.TOOL, text=Some(content), tool_call_id=Some(tool_call_id))
