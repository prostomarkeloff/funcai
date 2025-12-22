"""Media types for multimodal messages."""

from dataclasses import dataclass, field
from enum import StrEnum

from kungfu import Option, Nothing


class MediaType(StrEnum):
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"


@dataclass
class InputFile:
    file: bytes
    content_type: str
    file_name: Option[str] = field(default_factory=Nothing)


@dataclass
class Media:
    media_type: MediaType
    file: InputFile


def image(
    file: bytes,
    content_type: str = "image/png",
    file_name: Option[str] = Nothing(),
) -> Media:
    return Media(
        media_type=MediaType.IMAGE,
        file=InputFile(file=file, content_type=content_type, file_name=file_name),
    )


def audio(
    file: bytes,
    content_type: str = "audio/mpeg",
    file_name: Option[str] = Nothing(),
) -> Media:
    return Media(
        media_type=MediaType.AUDIO,
        file=InputFile(file=file, content_type=content_type, file_name=file_name),
    )


def video(
    file: bytes,
    content_type: str = "video/mp4",
    file_name: Option[str] = Nothing(),
) -> Media:
    return Media(
        media_type=MediaType.VIDEO,
        file=InputFile(file=file, content_type=content_type, file_name=file_name),
    )


def file(
    file: bytes,
    content_type: str = "application/octet-stream",
    file_name: Option[str] = Nothing(),
) -> Media:
    return Media(
        media_type=MediaType.FILE,
        file=InputFile(file=file, content_type=content_type, file_name=file_name),
    )
