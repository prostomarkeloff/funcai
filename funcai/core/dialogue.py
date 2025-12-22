from typing import overload
import typing

from kungfu import LazyCoroResult, Ok, Error, Result, Nothing, Some
from pydantic import BaseModel

from funcai.core.message import Message
from funcai.core.provider import ABCAIProvider, AIResponse


class Dialogue:
    """
    A conversation as a list of messages.

    >>> dialogue = Dialogue([
    ...     system(text="You are helpful."),
    ...     user(text="Hello!"),
    ... ])
    >>> response = await dialogue.interpret(provider)
    """

    def __init__(self, messages: list[Message] | None = None):
        self.messages = messages or []

    @overload
    def interpret[E](
        self,
        provider: ABCAIProvider[E],
    ) -> LazyCoroResult[AIResponse[None], E]: ...

    @overload
    def interpret[E, S: BaseModel](
        self,
        provider: ABCAIProvider[E],
        schema: type[S] | Nothing,
    ) -> LazyCoroResult[S, E | str]: ...

    def interpret[E, S: BaseModel](
        self,
        provider: ABCAIProvider[E],
        schema: type[S] | Nothing = Nothing(),
    ) -> LazyCoroResult[AIResponse[None], E] | LazyCoroResult[S, E | str]:
        match schema:
            case Nothing():
                return LazyCoroResult[AIResponse[None], E](
                    lambda: typing.cast(
                        typing.Coroutine[None, None, Result[AIResponse[None], E]],
                        provider.send_messages(self.messages, schema=Nothing()),
                    )
                )
            case s:

                async def interpret_structured() -> Result[S, E | str]:
                    result = await provider.send_messages(self.messages, schema=Some(s))
                    match result:
                        case Ok(response):
                            match response.parsed:
                                case Nothing():
                                    return Error("No parsed response")
                                case Some(parsed):
                                    return Ok(parsed)
                        case Error(e):
                            return Error(e)

                return LazyCoroResult(interpret_structured)

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def pop(self) -> Message:
        return self.messages.pop()

    def copy(self) -> "Dialogue":
        return Dialogue(self.messages.copy())
