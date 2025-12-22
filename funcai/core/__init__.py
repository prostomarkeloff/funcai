"""Core funcai primitives."""

from funcai.core.types import ToolCall, ToolResult
from funcai.core.media import Media, MediaType, InputFile, image, audio, video, file
from funcai.core.message import Message, Role, user, system, assistant, tool_result
from funcai.core.provider import ABCAIProvider, AIResponse
from funcai.core.dialogue import Dialogue

__all__ = [
    "ToolCall",
    "ToolResult",
    "Dialogue",
    "Message",
    "Role",
    "user",
    "system",
    "assistant",
    "tool_result",
    "Media",
    "MediaType",
    "InputFile",
    "image",
    "audio",
    "video",
    "file",
    "ABCAIProvider",
    "AIResponse",
]
