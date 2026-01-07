"""OpenAI Chat Completions provider with multimodal support."""

import base64
import json
from dataclasses import dataclass, field
from typing import Any, Literal

from kungfu import Error, Result, Ok, Option, Nothing, Some, from_optional
from openai import AsyncOpenAI, APIError
from pydantic import BaseModel

from funcai.core.message import Message, Role
from funcai.core.media import Media, MediaType, audio as audio_media
from funcai.core.provider import ABCAIProvider, AIResponse
from funcai.core.types import ToolCall
from funcai.agents.tool import Tool


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

type Modality = Literal["text", "audio"]
type AudioVoice = Literal["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
type AudioFormat = Literal["wav", "mp3", "flac", "opus", "pcm16"]


@dataclass(frozen=True)
class OpenAIError:
    """Error from OpenAI API."""
    message: str
    code: Option[str] = field(default_factory=Nothing)

    def __str__(self) -> str:
        match self.code:
            case Some(c):
                return f"OpenAIError[{c}]: {self.message}"
            case Nothing():
                return f"OpenAIError: {self.message}"

    @classmethod
    def from_api_error(cls, e: APIError) -> "OpenAIError":
        return cls(message=str(e), code=from_optional(e.code))


@dataclass(frozen=True)
class AudioConfig:
    """Configuration for audio output."""
    voice: AudioVoice = "alloy"
    format: AudioFormat = "mp3"


# ─────────────────────────────────────────────────────────────────────────────
# Converters
# ─────────────────────────────────────────────────────────────────────────────

def _audio_format(content_type: str) -> str:
    return content_type.split("/")[1] if "/" in content_type else content_type


def _media_to_content_part(media: Media) -> dict[str, Any]:
    """Convert Media to OpenAI content part for input."""
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

    # NOTE: OpenAI strict mode requires ALL properties to be in 'required',
    # even those with default values. This is weird but it's how strict works.
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

    # Multimodal input
    content: list[dict[str, Any]] = []
    match message.text:
        case Some(text):
            content.append({"type": "text", "text": text})
        case _:
            pass
    for media in message.media:
        content.append(_media_to_content_part(media))

    return {"role": message.role.value, "content": content}


def _parse_audio_output(audio_data: Any, format: AudioFormat) -> Media:
    """Parse audio from OpenAI response."""
    audio_bytes = base64.b64decode(audio_data.data)
    content_type = f"audio/{format}"
    return audio_media(file=audio_bytes, content_type=content_type)


# ─────────────────────────────────────────────────────────────────────────────
# Provider
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider(ABCAIProvider[OpenAIError]):
    """
    OpenAI Chat Completions provider with multimodal support.

    Supports:
    - Text input/output
    - Image input (GPT-4o vision)
    - Audio input (GPT-4o audio)
    - Audio output (GPT-4o-audio-preview)

    Examples:
        # Text only
        >>> gpt = OpenAIProvider(model="gpt-4o")

        # With audio output
        >>> gpt_audio = OpenAIProvider(
        ...     model="gpt-4o-audio-preview",
        ...     modalities=["text", "audio"],
        ...     audio=AudioConfig(voice="alloy", format="mp3"),
        ... )
    """

    def __init__(
        self,
        model: str,
        api_key: Option[str] = Nothing(),
        temperature: Option[float] = Nothing(),
        modalities: list[Modality] | None = None,
        audio: AudioConfig | None = None,
    ) -> None:
        self.client = AsyncOpenAI(api_key=api_key.unwrap_or_none())
        self.model = model
        self.temperature = temperature
        self.modalities = modalities
        self.audio_config = audio

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

        # Temperature
        match self.temperature:
            case Some(temp):
                kwargs["temperature"] = temp
            case _:
                pass

        # Structured output
        match schema:
            case Some(s):
                kwargs["response_format"] = s
            case _:
                pass

        # Tools
        if tools:
            kwargs["tools"] = [_tool_to_openai(t) for t in tools]

        # Multimodal output: modalities + audio config
        if self.modalities:
            kwargs["modalities"] = self.modalities

            if self.audio_config and "audio" in self.modalities:
                kwargs["audio"] = {
                    "voice": self.audio_config.voice,
                    "format": self.audio_config.format,
                }

        # API call
        try:
            response = await self.client.chat.completions.parse(**kwargs)
        except APIError as e:
            return Error(OpenAIError.from_api_error(e))

        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls
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

        # Parse response text
        response_text: str | None = msg.content

        # Parse audio output if present
        response_media: list[Media] = []
        audio_transcript: str | None = None

        if msg.audio is not None:
            audio_format = self.audio_config.format if self.audio_config else "mp3"
            response_media.append(_parse_audio_output(msg.audio, audio_format))
            # Audio transcript can be used as text
            if msg.audio.transcript:
                audio_transcript = msg.audio.transcript

        # Use audio transcript if no text content
        final_text = response_text or audio_transcript

        # Build response message
        response_msg = Message(
            role=Role.ASSISTANT,
            text=from_optional(final_text),
            media=response_media,
            tool_calls=tool_calls,
        )

        # Parse structured output
        parsed: Option[S] = Nothing()
        match schema:
            case Some(_) if msg.parsed is not None:
                parsed = Some(msg.parsed)
            case _:
                pass

        # Metadata
        meta: dict[str, Any] = {
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
            "finish_reason": choice.finish_reason,
        }

        # Add audio metadata if present
        if msg.audio:
            meta["audio_id"] = msg.audio.id if hasattr(msg.audio, "id") else None

        return Ok(
            AIResponse(
                message=response_msg,
                tool_calls=tool_calls,
                parsed=parsed,
                meta=meta,
            )
        )
