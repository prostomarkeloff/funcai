"""
Domain-specific combinators for funcai.

Generic combinators (parallel, fallback, timeout, batch, etc.) are now in
the `combinators` package: https://github.com/prostomarkeloff/combinators.py

This module contains funcai-specific combinators that work with Dialogue
and AI providers.
"""

from typing import Callable

from kungfu import Error, LazyCoroResult, Ok, Result
from pydantic import BaseModel

from funcai.core import ABCAIProvider, Dialogue, Message, assistant, system, user

type Interp[T, E] = LazyCoroResult[T, E]
type Condition[T] = Callable[[T], bool]
type Selector[T, K] = Callable[[T], K]
type Route[T, R, E] = Callable[[T], Interp[R, E]]


def when[T, R, E](
    interp: Interp[T, E],
    condition: Condition[T],
    then: Route[T, R, E],
    otherwise: Route[T, R, E],
) -> Interp[R, E]:
    """
    Conditional branching.

    >>> result = await when(
    ...     dialogue.interpret(provider, Analysis),
    ...     condition=lambda a: a.confidence > 0.8,
    ...     then=lambda a: proceed(a),
    ...     otherwise=lambda a: clarify(a),
    ... )
    """

    async def run() -> Result[R, E]:
        match await interp:
            case Ok(value):
                route = then if condition(value) else otherwise
                return await route(value)
            case Error(e):
                return Error(e)

    return LazyCoroResult(run)


def refine[T: BaseModel, E](
    dialogue: Dialogue,
    provider: ABCAIProvider[E],
    schema: type[T],
    until: Condition[T],
    feedback: Callable[[T], str],
    max_rounds: int = 3,
) -> Interp[T, E | str]:
    """
    Iteratively refine until condition is met.

    >>> essay = await refine(
    ...     dialogue,
    ...     provider,
    ...     Essay,
    ...     until=lambda e: e.word_count >= 500,
    ...     feedback=lambda e: f"Too short ({e.word_count} words). Expand.",
    ...     max_rounds=3,
    ... )
    """

    async def run() -> Result[T, E | str]:
        d = Dialogue(dialogue.messages.copy())

        for _ in range(max_rounds):
            match await d.interpret(provider, schema):
                case Ok(value):
                    if until(value):
                        return Ok(value)
                    d.append(assistant(text=str(value)))
                    d.append(user(text=feedback(value)))
                case Error(e):
                    return Error(e)

        return Error(f"Condition not met after {max_rounds} rounds")

    return LazyCoroResult(run)


def with_context(
    dialogue: Dialogue, context: list[str], separator: str = "\n---\n"
) -> Dialogue:
    """
    Inject context documents into dialogue.

    >>> enriched = with_context(dialogue, docs)
    >>> result = await enriched.interpret(provider, Answer)
    """
    context_msg = system(text=f"Context:\n{separator.join(context)}")
    return Dialogue([context_msg] + dialogue.messages)


def prepend(dialogue: Dialogue, *messages: Message) -> Dialogue:
    """Prepend messages to dialogue."""
    return Dialogue(list(messages) + dialogue.messages)


def append(dialogue: Dialogue, *messages: Message) -> Dialogue:
    """Append messages to dialogue."""
    return Dialogue(dialogue.messages + list(messages))


__all__ = [
    # Type aliases
    "Interp",
    "Condition",
    "Selector",
    "Route",
    # Domain-specific combinators
    "when",
    "refine",
    "with_context",
    "prepend",
    "append",
]
