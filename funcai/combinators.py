import asyncio
from typing import Callable, Awaitable

from kungfu import LazyCoroResult, Ok, Error, Result, Option, Nothing, Some
from pydantic import BaseModel

from funcai.core import Dialogue, Message, user, assistant, system, ABCAIProvider


type Interp[T, E] = LazyCoroResult[T, E]
type Condition[T] = Callable[[T], bool]
type Selector[T, K] = Callable[[T], K]
type Route[T, R, E] = Callable[[T], Interp[R, E]]


def parallel[T, E](*interps: Interp[T, E]) -> Interp[list[T], E]:
    """
    Run interpretations in parallel, collect all results.
    Fails on first error.

    >>> results = await parallel(
    ...     dialogue1.interpret(provider, Story),
    ...     dialogue2.interpret(provider, Story),
    ...     dialogue3.interpret(provider, Story),
    ... )
    """

    async def run() -> Result[list[T], E]:
        results = await asyncio.gather(*[i() for i in interps])
        values: list[T] = []
        for r in results:
            match r:
                case Ok(v):
                    values.append(v)
                case Error(e):
                    return Error(e)
        return Ok(values)

    return LazyCoroResult(run)


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


def vote[T: BaseModel, E](
    candidates: list[Interp[T, E]],
    judge: Callable[[list[T]], Awaitable[T]],
) -> Interp[T, E]:
    """
    Generate multiple candidates, let judge pick the best.

    >>> best = await vote(
    ...     candidates=[
    ...         dialogue.interpret(provider, Story) for _ in range(3)
    ...     ],
    ...     judge=lambda stories: pick_best(stories, provider),
    ... )
    """

    async def run() -> Result[T, E]:
        match await parallel(*candidates):
            case Ok(values):
                winner = await judge(values)
                return Ok(winner)
            case Error(e):
                return Error(e)

    return LazyCoroResult(run)


def best_of[T, E](
    interp: Interp[T, E],
    n: int,
    key: Callable[[T], float],
) -> Interp[T, E]:
    """
    Run n times, pick best by key function.

    >>> result = await best_of(
    ...     dialogue.interpret(provider, Analysis),
    ...     n=3,
    ...     key=lambda a: a.confidence,
    ... )
    """

    async def run() -> Result[T, E]:
        copies = [interp for _ in range(n)]
        match await parallel(*copies):
            case Ok(values):
                best = max(values, key=key)
                return Ok(best)
            case Error(e):
                return Error(e)

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


def tap[T, E](
    interp: Interp[T, E],
    effect: Callable[[T], None],
) -> Interp[T, E]:
    """
    Perform side effect without changing value.

    >>> result = await tap(
    ...     dialogue.interpret(provider, Answer),
    ...     effect=lambda a: print(f"Got answer: {a}"),
    ... )
    """
    return interp.map(lambda v: (effect(v), v)[1])


class TimeoutError(Exception):
    """Raised when interpretation times out."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        super().__init__(f"Timed out after {seconds}s")


def fallback[T, E](
    primary: Interp[T, E],
    secondary: Interp[T, E],
) -> Interp[T, E]:
    """
    Use secondary if primary fails.

    >>> result = await fallback(
    ...     dialogue.interpret(gpt4, Answer),
    ...     dialogue.interpret(gpt35, Answer),
    ... )
    """

    async def run() -> Result[T, E]:
        match await primary:
            case Ok(v):
                return Ok(v)
            case Error(_):
                return await secondary

    return LazyCoroResult(run)


def fallback_chain[T, E](*interps: Interp[T, E]) -> Interp[T, E]:
    """
    Try each interpretation in order until one succeeds.

    >>> result = await fallback_chain(
    ...     dialogue.interpret(gpt4, Answer),
    ...     dialogue.interpret(claude, Answer),
    ...     dialogue.interpret(gemini, Answer),
    ... )
    """

    async def run() -> Result[T, E]:
        last_error: Option[E] = Nothing()
        for interp in interps:
            match await interp:
                case Ok(v):
                    return Ok(v)
                case Error(e):
                    last_error = Some(e)
        return Error(last_error.unwrap())  # type: ignore

    return LazyCoroResult(run)


def timeout[T, E](
    interp: Interp[T, E],
    seconds: float,
) -> Interp[T, E | TimeoutError]:
    """
    Fail if interpretation takes too long.

    >>> result = await timeout(
    ...     dialogue.interpret(provider, Answer),
    ...     seconds=30.0,
    ... )
    """

    async def run() -> Result[T, E | TimeoutError]:
        try:
            return await asyncio.wait_for(interp(), timeout=seconds)  # type: ignore
        except asyncio.TimeoutError:
            return Error(TimeoutError(seconds))

    return LazyCoroResult(run)


def batch[A, T, E](
    items: list[A],
    handler: Callable[[A], Interp[T, E]],
    concurrency: int = 5,
) -> Interp[list[T], E]:
    """
    Process items with limited parallelism.

    >>> results = await batch(
    ...     items=documents,
    ...     handler=lambda doc: summarize(doc, provider),
    ...     concurrency=3,
    ... )
    """

    async def run() -> Result[list[T], E]:
        semaphore = asyncio.Semaphore(concurrency)
        results: list[T] = []
        errors: list[E] = []

        async def process(item: A) -> None:
            async with semaphore:
                match await handler(item):
                    case Ok(v):
                        results.append(v)
                    case Error(e):
                        errors.append(e)

        await asyncio.gather(*[process(item) for item in items])

        if errors:
            return Error(errors[0])
        return Ok(results)

    return LazyCoroResult(run)


def batch_all[A, T, E](
    items: list[A],
    handler: Callable[[A], Interp[T, E]],
    concurrency: int = 5,
) -> Interp[list[Result[T, E]], None]:
    """
    Process all items, collect both successes and failures.

    >>> results = await batch_all(
    ...     items=documents,
    ...     handler=lambda doc: summarize(doc, provider),
    ...     concurrency=3,
    ... )
    >>> for r in results.unwrap():
    ...     match r:
    ...         case Ok(summary): print(summary)
    ...         case Error(e): print(f"Failed: {e}")
    """

    async def run() -> Result[list[Result[T, E]], None]:
        semaphore = asyncio.Semaphore(concurrency)
        results: list[Result[T, E]] = [None] * len(items)  # type: ignore

        async def process(idx: int, item: A) -> None:
            async with semaphore:
                results[idx] = await handler(item)

        await asyncio.gather(*[process(i, item) for i, item in enumerate(items)])
        return Ok(results)

    return LazyCoroResult(run)
