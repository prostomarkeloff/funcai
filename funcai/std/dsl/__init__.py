"""
Typed DSL for funcai — AST-based orchestration layer.

Usage:
    from funcai.std.dsl import AI, Op, compile, analyze

    # Build program as AST
    program = (
        AI.ask(dialogue, Analysis)
        .when(
            condition=lambda a: a.confidence > 0.8,
            then=lambda a: AI.ask(followup(a), Summary),
            otherwise=lambda a: AI.agent(clarify(a), tools),
        )
        .timeout(30.0)
        .fallback(AI.pure(default_answer))
    )

    # Analyze before running
    stats = analyze(program.op)
    print(f"Max LLM calls: {stats.llm_calls}")

    # Compile and run
    result = await program.compile(provider)
"""

from funcai.std.dsl.op import Op
from funcai.std.dsl.builder import AI
from funcai.std.dsl.compile import compile
from funcai.std.dsl.analyze import (
    analyze,
    Stats,
    Bounds,
    fmt,
    FmtStyle,
    warnings,
    validate,
    is_valid,
    Warning,
    Severity,
)

__all__ = (
    # Types
    "Op",
    "AI",
    "Stats",
    "Bounds",
    "Warning",
    "Severity",
    "FmtStyle",
    # Functions
    "compile",
    "analyze",
    "fmt",
    "warnings",
    "validate",
    "is_valid",
)

