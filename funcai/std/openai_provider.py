import base64
import json
from dataclasses import dataclass, field
from typing import Any

from kungfu import Error, Result, Ok, Option, Nothing, Some, from_optional
from openai import AsyncOpenAI, APIError
from pydantic import BaseModel

from funcai.core.message import Message, Role, assistant
from funcai.core.media import Media, MediaType
from funcai.core.provider import ABCAIProvider, AIResponse
from funcai.core.types import ToolCall
from funcai.agents.tool import Tool


@dataclass(frozen=True)
class OpenAIError:
    message: str
    code: Option[str] = field(default_factory=Nothing)

    @classmethod
    def from_api_error(cls, e: APIError) -> "OpenAIError":
        return cls(message=str(e), code=from_optional(e.code))


def _audio_format(content_type: str) -> str:
    return content_type.split("/")[1] if "/" in content_type else content_type


def _media_to_content_part(media: Media) -> dict[str, Any]:
    b64 = base64.b64encode(media.file.file).decode("utf-8")

    match media.media_type:
        case MediaType.IMAGE:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:{media.file.content_type};base64,{b64}"},
            }
        case MediaType.AUDIO:
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": b64,
                    "format": _audio_format(media.file.content_type),
                },
            }
        case MediaType.VIDEO:
            raise RuntimeError("Video not supported. Extract frames as images.")
        case MediaType.FILE:
            raise RuntimeError("Files not supported. Use Assistants API.")


def _tool_to_openai(tool: Tool) -> dict[str, Any]:
    schema = tool.parameters.model_json_schema()
    schema["additionalProperties"] = False

    if "properties" in schema:
        schema["required"] = list(schema.get("properties", {}).keys())

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "strict": True,
            "parameters": schema,
        },
    }


def _message_to_openai(message: Message) -> dict[str, Any]:
    """Convert funcai Message to OpenAI format."""
    # Tool result
    if message.role == Role.TOOL:
        return {
            "role": "tool",
            "tool_call_id": message.tool_call_id.unwrap_or_none(),
            "content": message.text.unwrap_or(""),
        }

    # Assistant with tool calls
    if message.role == Role.ASSISTANT and message.has_tool_calls:
        return {
            "role": "assistant",
            "content": message.text.unwrap_or_none(),
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in message.tool_calls
            ],
        }

    # Simple text
    if not message.media:
        return {"role": message.role.value, "content": message.text.unwrap_or("")}

    # Multimodal
    content: list[dict[str, Any]] = []
    match message.text:
        case Some(text):
            content.append({"type": "text", "text": text})
        case _:
            pass
    for media in message.media:
        content.append(_media_to_content_part(media))

    return {"role": message.role.value, "content": content}


class OpenAIProvider(ABCAIProvider[OpenAIError]):
    """OpenAI Chat Completions provider."""

    def __init__(
        self,
        model: str,
        api_key: Option[str] = Nothing(),
        temperature: Option[float] = Nothing(),
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key.unwrap_or_none())
        self.model = model
        self.temperature = temperature

    async def send_messages[S: BaseModel](
        self,
        messages: list[Message],
        *,
        schema: Option[type[S]] = Nothing(),
        tools: list[Tool] | None = None,
    ) -> Result[AIResponse[S], OpenAIError]:
        openai_messages = [_message_to_openai(m) for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
        }

        match self.temperature:
            case Some(temp):
                kwargs["temperature"] = temp
            case _:
                pass

        match schema:
            case Some(s):
                kwargs["response_format"] = s
            case _:
                pass

        if tools:
            kwargs["tools"] = [_tool_to_openai(t) for t in tools]

        try:
            response = await self.client.chat.completions.parse(**kwargs)
        except APIError as e:
            return Error(OpenAIError.from_api_error(e))

        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls into core types
        tool_calls: list[ToolCall] = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )

        # Message includes tool_calls for conversation history
        response_msg = assistant(text=msg.content, tool_calls=tool_calls)

        parsed: Option[S] = Nothing()
        match schema:
            case Some(_) if msg.parsed is not None:
                parsed = Some(msg.parsed)
            case _:
                pass

        meta: dict[str, Any] = {
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens
                if response.usage
                else None,
                "completion_tokens": response.usage.completion_tokens
                if response.usage
                else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "finish_reason": choice.finish_reason,
        }

        return Ok(
            AIResponse(
                message=response_msg, tool_calls=tool_calls, parsed=parsed, meta=meta
            )
        )
