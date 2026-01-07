"""
Op — AST nodes for the funcai DSL.

Each Op represents a computation. Type information is tracked
at the AI[T] level (phantom types), not in the Op itself.

This design choice:
- Avoids complex generic constraints that Python's type system can't express
- Keeps Op as simple immutable data structures
- Type safety is enforced at the builder level (AI[T])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any

from pydantic import BaseModel

from funcai.core.dialogue import Dialogue
from funcai.agents.tool import Tool


# ═══════════════════════════════════════════════════════════════════════════════
# Core LLM Operations
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Ask:
    """Simple LLM call with structured output."""
    dialogue: Dialogue
    schema: type[BaseModel]


@dataclass(frozen=True)
class Agent:
    """Agent loop with tools."""
    dialogue: Dialogue
    tools: list[Tool]
    schema: type[BaseModel] | None = None
    max_steps: int = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Functor / Monad
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Pure:
    """Lift a pure value into Op."""
    value: Any


@dataclass(frozen=True)
class Map:
    """Functor map — transform the result."""
    inner: Op
    f: Callable[[Any], Any]


@dataclass(frozen=True)
class Then:
    """Monadic bind — sequence computations."""
    inner: Op
    f: Callable[[Any], Op]


# ═══════════════════════════════════════════════════════════════════════════════
# Parallelism
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Parallel:
    """Run operations in parallel, collect all results."""
    ops: list[Op]


@dataclass(frozen=True)
class Batch:
    """Process items with bounded parallelism."""
    items: list[Any]
    handler: Callable[[Any], Op]
    concurrency: int = 5


# ═══════════════════════════════════════════════════════════════════════════════
# Error Handling
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Fallback:
    """Try primary, fall back to alternatives on error."""
    primary: Op
    fallbacks: list[Op]


@dataclass(frozen=True)
class Timeout:
    """Fail if computation takes too long."""
    inner: Op
    seconds: float


@dataclass(frozen=True)
class MapErr:
    """Transform error type."""
    inner: Op
    f: Callable[[Any], Any]


# ═══════════════════════════════════════════════════════════════════════════════
# Control Flow
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class When:
    """Conditional branching based on result."""
    inner: Op
    condition: Callable[[Any], bool]
    then: Callable[[Any], Op]
    otherwise: Callable[[Any], Op]


@dataclass(frozen=True)
class Tap:
    """Perform side effect without changing result."""
    inner: Op
    effect: Callable[[Any], None]


# ═══════════════════════════════════════════════════════════════════════════════
# Refinement / Selection
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Refine:
    """Iteratively refine until condition is met."""
    dialogue: Dialogue
    schema: type[BaseModel]
    until: Callable[[Any], bool]
    feedback: Callable[[Any], str]
    max_rounds: int = 3


@dataclass(frozen=True)
class BestOf:
    """Run N times, pick best by key function."""
    inner: Op
    n: int
    key: Callable[[Any], float]


@dataclass(frozen=True)
class Ensure:
    """Validate result, fail if check returns False."""
    inner: Op
    check: Callable[[Any], bool]
    error: str


# ═══════════════════════════════════════════════════════════════════════════════
# Op — Union of all operations (for pattern matching)
# ═══════════════════════════════════════════════════════════════════════════════

type Op = (
    Ask
    | Agent
    | Pure
    | Map
    | Then
    | Parallel
    | Batch
    | Fallback
    | Timeout
    | MapErr
    | When
    | Tap
    | Refine
    | BestOf
    | Ensure
)


__all__ = (
    "Op",
    "Ask",
    "Agent",
    "Pure",
    "Map",
    "Then",
    "Parallel",
    "Batch",
    "Fallback",
    "Timeout",
    "MapErr",
    "When",
    "Tap",
    "Refine",
    "BestOf",
    "Ensure",
)
