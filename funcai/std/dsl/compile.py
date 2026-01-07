"""
Compiler: Op → LazyCoroResult[Any, E]

This is the interpreter for the DSL — it traverses the AST
and builds the actual computation using funcai primitives.

Type safety is handled at the AI[T] level, not here.
"""

from __future__ import annotations

from typing import Any

from kungfu import LazyCoroResult, Result, Ok, Error
from combinators import fallback_chain, parallel, timeout, batch, best_of, tap

from funcai.core.provider import ABCAIProvider
from funcai.combinators import when, refine
from funcai.agents.agent import agent as agent_fn

from funcai.std.dsl.op import (
    Op,
    Ask,
    Agent,
    Pure,
    Map,
    Then,
    Parallel,
    Batch,
    Fallback,
    Timeout,
    MapErr,
    When,
    Tap,
    Refine,
    BestOf,
    Ensure,
)


def compile(
    op: Op,
    provider: ABCAIProvider[Any],
) -> LazyCoroResult[Any, Any]:
    """
    Compile an Op AST to an executable LazyCoroResult.

    Type information is erased at this level — the AI[T] wrapper
    maintains type safety through phantom types.
    """
    match op:
        # ─── Core LLM Operations ───────────────────────────────────────────

        case Ask(dialogue, schema):
            return dialogue.interpret(provider, schema)

        case Agent(dialogue, tools, schema, max_steps):
            if schema is not None:
                return agent_fn(dialogue, provider, tools, max_steps, schema=schema)
            return agent_fn(dialogue, provider, tools, max_steps)

        # ─── Functor / Monad ───────────────────────────────────────────────

        case Pure(value):
            return LazyCoroResult.pure(value)

        case Map(inner, f):
            return compile(inner, provider).map(f)

        case Then(inner, f):
            return compile(inner, provider).then(
                lambda t: compile(f(t), provider)
            )

        # ─── Parallelism ───────────────────────────────────────────────────

        case Parallel(ops):
            compiled = [compile(o, provider) for o in ops]
            return parallel(*compiled)

        case Batch(items, handler, concurrency):
            return batch(
                items,
                lambda item: compile(handler(item), provider),
                concurrency=concurrency,
            )

        # ─── Error Handling ────────────────────────────────────────────────

        case Fallback(primary, fallbacks):
            compiled_primary = compile(primary, provider)
            compiled_fallbacks = [compile(fb, provider) for fb in fallbacks]
            return fallback_chain(compiled_primary, *compiled_fallbacks)

        case Timeout(inner, seconds):
            return timeout(compile(inner, provider), seconds=seconds)

        case MapErr(inner, f):
            return compile(inner, provider).map_err(f)

        # ─── Control Flow ──────────────────────────────────────────────────

        case When(inner, condition, then, otherwise):
            return when(
                compile(inner, provider),
                condition,
                lambda t: compile(then(t), provider),
                lambda t: compile(otherwise(t), provider),
            )

        case Tap(inner, effect):
            return tap(compile(inner, provider), effect=effect)

        # ─── Refinement / Selection ────────────────────────────────────────

        case Refine(dialogue, schema, until, feedback, max_rounds):
            return refine(
                dialogue, provider, schema, until, feedback, max_rounds
            )

        case BestOf(inner, n, key):
            return best_of(compile(inner, provider), n=n, key=key)

        case Ensure(inner, check, error):
            async def run() -> Result[Any, Any]:
                result = await compile(inner, provider)
                match result:
                    case Ok(value):
                        if check(value):
                            return Ok(value)
                        return Error(error)
                    case Error(e):
                        return Error(e)

            return LazyCoroResult(run)


__all__ = ("compile",)
