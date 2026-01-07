"""
AI[T] — Type-safe fluent builder for the funcai DSL.

Type safety is achieved through phantom types:
- AI[T] wraps an untyped Op
- T tracks the result type at compile time
- Each method preserves/transforms T correctly
"""

from __future__ import annotations

from typing import Callable, overload, cast

from pydantic import BaseModel
from kungfu import LazyCoroResult

from funcai.core.dialogue import Dialogue
from funcai.core.provider import ABCAIProvider
from funcai.agents.tool import Tool
from funcai.agents.abc import AgentResponse

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
    BestOf,
    Ensure,
    Refine,
)


class AI[T]:
    """
    Type-safe builder for AI programs.

    T is a phantom type — it exists only at type-checking time.
    The underlying Op is untyped, but AI[T] ensures type safety
    through its methods.

    Example:
        program: AI[Summary] = (
            AI.ask(dialogue, Analysis)
            .when(
                condition=lambda a: a.confidence > 0.8,
                then=lambda a: AI.ask(followup(a), Summary),
                otherwise=lambda _: AI.pure(default_summary),
            )
            .timeout(30.0)
        )

        result = await program.compile(provider)
    """

    __slots__ = ("_op",)

    def __init__(self, op: Op) -> None:
        self._op = op

    # ═══════════════════════════════════════════════════════════════════════════
    # Constructors
    # ═══════════════════════════════════════════════════════════════════════════

    @staticmethod
    def pure[U](value: U) -> AI[U]:
        """Lift a pure value into AI."""
        return AI(Pure(value))

    @staticmethod
    def ask[S: BaseModel](dialogue: Dialogue, schema: type[S]) -> AI[S]:
        """Simple LLM call with structured output."""
        return AI(Ask(dialogue, schema))

    @overload
    @staticmethod
    def agent(
        dialogue: Dialogue,
        tools: list[Tool],
        *,
        max_steps: int = 10,
    ) -> AI[AgentResponse[None]]: ...

    @overload
    @staticmethod
    def agent[S: BaseModel](
        dialogue: Dialogue,
        tools: list[Tool],
        *,
        schema: type[S],
        max_steps: int = 10,
    ) -> AI[AgentResponse[S]]: ...

    @staticmethod
    def agent[S: BaseModel](
        dialogue: Dialogue,
        tools: list[Tool],
        *,
        schema: type[S] | None = None,
        max_steps: int = 10,
    ) -> AI[AgentResponse[S]] | AI[AgentResponse[None]]:
        """Agent loop with tools."""
        return AI(Agent(dialogue, tools, schema, max_steps))

    @staticmethod
    def parallel[U](*programs: AI[U]) -> AI[list[U]]:
        """Run programs in parallel, collect all results."""
        return AI(Parallel([p._op for p in programs]))

    @staticmethod
    def batch[A, U](
        items: list[A],
        handler: Callable[[A], AI[U]],
        concurrency: int = 5,
    ) -> AI[list[U]]:
        """Process items with bounded parallelism."""
        return AI(Batch(items, lambda a: handler(a)._op, concurrency))

    @staticmethod
    def refine[S: BaseModel](
        dialogue: Dialogue,
        schema: type[S],
        *,
        until: Callable[[S], bool],
        feedback: Callable[[S], str],
        max_rounds: int = 3,
    ) -> AI[S]:
        """Iteratively refine until condition is met."""
        return AI(Refine(dialogue, schema, until, feedback, max_rounds))

    # ═══════════════════════════════════════════════════════════════════════════
    # Functor / Monad
    # ═══════════════════════════════════════════════════════════════════════════

    def map[U](self, f: Callable[[T], U]) -> AI[U]:
        """Transform the result."""
        return AI(Map(self._op, f))

    def then[U](self, f: Callable[[T], AI[U]]) -> AI[U]:
        """Sequence computations — flatMap / monadic bind."""
        return AI(Then(self._op, lambda t: f(t)._op))

    def map_err(self, f: Callable[[object], object]) -> AI[T]:
        """Transform error type."""
        return AI(MapErr(self._op, f))

    # ═══════════════════════════════════════════════════════════════════════════
    # Error Handling
    # ═══════════════════════════════════════════════════════════════════════════

    def fallback(self, *others: AI[T]) -> AI[T]:
        """Try self, fall back to others on error."""
        return AI(Fallback(self._op, [o._op for o in others]))

    def timeout(self, seconds: float) -> AI[T]:
        """Fail if computation takes too long."""
        return AI(Timeout(self._op, seconds))

    # ═══════════════════════════════════════════════════════════════════════════
    # Control Flow
    # ═══════════════════════════════════════════════════════════════════════════

    def when[R](
        self,
        condition: Callable[[T], bool],
        then: Callable[[T], AI[R]],
        otherwise: Callable[[T], AI[R]],
    ) -> AI[R]:
        """Conditional branching based on result."""
        return AI(When(
            self._op,
            condition,
            lambda t: then(t)._op,
            lambda t: otherwise(t)._op,
        ))

    def tap(self, effect: Callable[[T], None]) -> AI[T]:
        """Perform side effect without changing result."""
        return AI(Tap(self._op, effect))

    # ═══════════════════════════════════════════════════════════════════════════
    # Refinement / Selection
    # ═══════════════════════════════════════════════════════════════════════════

    def best_of(self, n: int, key: Callable[[T], float]) -> AI[T]:
        """Run N times, pick best by key function."""
        return AI(BestOf(self._op, n, key))

    def ensure(self, check: Callable[[T], bool], error: str) -> AI[T]:
        """Validate result, fail if check returns False."""
        return AI(Ensure(self._op, check, error))

    # ═══════════════════════════════════════════════════════════════════════════
    # Execution & Introspection
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def op(self) -> Op:
        """Access raw AST for analysis or custom compilation."""
        return self._op

    def compile[E](self, provider: ABCAIProvider[E]) -> LazyCoroResult[T, E | str]:
        """
        Compile AST to executable LazyCoroResult.
        
        NOTE: Why cast is needed here:
        
        The compile() function returns LazyCoroResult[Any, Any] because:
        1. Op is an untyped union — Python's type system cannot express
           higher-kinded types (HKT) required for typed AST.
        2. Each Op node (Ask, Map, etc.) would need its own generic parameter,
           but then Op union would lose type info due to invariance.
        3. Making Op generic creates "Type X is not assignable to Y" errors
           in pattern matching because Python generics are invariant.
        
        The phantom type T in AI[T] tracks the correct type through the builder
        methods (ask returns AI[S], map[U] returns AI[U], etc.). This cast
        restores that type information after compilation.
        
        This is the standard "phantom type" pattern in languages without HKT.
        Haskell/Scala solve this with proper HKT, Python requires this bridge.
        """
        from funcai.std.dsl.compile import compile as compile_op
        return cast(LazyCoroResult[T, E | str], compile_op(self._op, provider))


__all__ = ("AI",)
