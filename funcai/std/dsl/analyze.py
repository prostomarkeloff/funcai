"""
Static analysis for the funcai DSL.

Analyze an Op AST without executing it to get bounds and metadata.
Useful for cost estimation, debugging, policy verification, and optimization.

Features:
- Bounds estimation (min/max LLM calls)
- Complexity scoring
- Validation (detect anti-patterns)
- Warnings (potential issues)
- Multiple output formats (tree, compact, mermaid)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

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


# ═══════════════════════════════════════════════════════════════════════════════
# Stats — Static Analysis Result
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Bounds:
    """
    Min/max bounds for a numeric value.

    min: best case (e.g., first fallback succeeds, when condition is cheap)
    max: worst case (e.g., all fallbacks fail except last)
    """
    min: int
    max: int

    def __add__(self, other: Bounds) -> Bounds:
        return Bounds(self.min + other.min, self.max + other.max)

    def __mul__(self, n: int) -> Bounds:
        return Bounds(self.min * n, self.max * n)

    @staticmethod
    def zero() -> Bounds:
        return Bounds(0, 0)

    @staticmethod
    def exact(n: int) -> Bounds:
        return Bounds(n, n)

    @staticmethod
    def range(min_val: int, max_val: int) -> Bounds:
        return Bounds(min_val, max_val)

    def max_with(self, other: Bounds) -> Bounds:
        """Take maximum of both bounds (for parallel branches)."""
        return Bounds(max(self.min, other.min), max(self.max, other.max))

    def sum_with(self, other: Bounds) -> Bounds:
        """Sum bounds (for sequential operations)."""
        return self + other

    def __repr__(self) -> str:
        if self.min == self.max:
            return f"{self.min}"
        return f"{self.min}..{self.max}"


@dataclass(frozen=True)
class Stats:
    """
    Static analysis result — bounds on resource usage.
    
    All values track both best-case (min) and worst-case (max) scenarios.
    """
    # LLM interaction bounds
    llm_calls: Bounds

    # Parallelism
    parallel_branches: int
    max_concurrency: int  # for batch operations

    # Tool usage
    tools_used: frozenset[str]

    # Control flow features
    has_timeout: bool
    has_fallback: bool
    has_branching: bool  # When nodes
    has_refinement: bool  # Refine nodes
    has_agent: bool

    # Agent-specific
    max_agent_steps: int

    # Complexity metrics
    depth: int  # AST nesting depth
    node_count: int  # total nodes in AST

    # Refinement-specific
    max_refine_rounds: int

    @property
    def is_deterministic(self) -> bool:
        """
        Returns True if the program has deterministic LLM call count.

        NOTE: "Deterministic" здесь означает что количество LLM-вызовов
        известно заранее (min == max), а не что результат детерминирован.
        Сами LLM-вызовы по определению недетерминированы.
        """
        return self.llm_calls.min == self.llm_calls.max

    @property
    def may_timeout(self) -> bool:
        """Returns True if any timeout is present."""
        return self.has_timeout

    @property
    def complexity_score(self) -> float:
        """
        Compute a complexity score for the program.

        Higher score = more complex execution pattern.
        Factors: LLM calls, parallelism, control flow, nesting.

        NOTE: Это эвристическая метрика для сравнения программ между собой,
        не абсолютная мера сложности. Коэффициенты подобраны эмпирически.
        """
        score = 0.0

        # Base: average LLM calls
        score += (self.llm_calls.min + self.llm_calls.max) / 2

        # Parallelism penalty (coordination overhead)
        score += self.parallel_branches * 0.5

        # Control flow penalty
        if self.has_branching:
            score += 2.0
        if self.has_fallback:
            score += 1.5
        if self.has_refinement:
            score += self.max_refine_rounds * 0.5

        # Agent penalty (complex loops)
        if self.has_agent:
            score += self.max_agent_steps * 0.3

        # Depth penalty (nested operations)
        score += self.depth * 0.2

        return round(score, 2)

    def pretty(self) -> str:
        """One-line summary for quick inspection."""
        parts = [
            f"LLM:{self.llm_calls}",
            f"∥:{self.parallel_branches}",
        ]
        if self.tools_used:
            parts.append(f"tools:{len(self.tools_used)}")
        if self.has_timeout:
            parts.append("⏱")
        if self.has_fallback:
            parts.append("↩")
        if self.has_branching:
            parts.append("⑂")
        if self.has_agent:
            parts.append(f"🤖(≤{self.max_agent_steps})")
        parts.append(f"⚙{self.complexity_score:.1f}")
        return " | ".join(parts)


def _empty_stats(depth: int = 0) -> Stats:
    """Create empty stats with given depth."""
    return Stats(
        llm_calls=Bounds.zero(),
        parallel_branches=1,
        max_concurrency=1,
        tools_used=frozenset(),
        has_timeout=False,
        has_fallback=False,
        has_branching=False,
        has_refinement=False,
        has_agent=False,
        max_agent_steps=0,
        depth=depth,
        node_count=1,
        max_refine_rounds=0,
    )


def _merge_stats(stats: list[Stats], mode: str = "sum") -> Stats:
    """
    Merge multiple stats together.

    mode:
        "sum" - для последовательных операций (llm_calls складываются)
        "max" - для fallback (берём худший случай)
        "parallel" - для parallel ops (llm_calls складываются, branches = len)
    """
    if not stats:
        return _empty_stats()

    if mode == "sum":
        llm_calls = Bounds.zero()
        for s in stats:
            llm_calls = llm_calls + s.llm_calls
    elif mode == "max":
        # Для fallback: min = 1 попытка, max = все попытки
        llm_calls = Bounds(
            min=stats[0].llm_calls.min,  # best case: first succeeds
            max=sum(s.llm_calls.max for s in stats),  # worst: all fail except last
        )
    else:  # parallel
        llm_calls = Bounds(
            min=sum(s.llm_calls.min for s in stats),
            max=sum(s.llm_calls.max for s in stats),
        )

    merged_tools: frozenset[str] = frozenset[str]().union(*(s.tools_used for s in stats))
    return Stats(
        llm_calls=llm_calls,
        parallel_branches=max(s.parallel_branches for s in stats) if mode != "parallel" else len(stats),
        max_concurrency=max(s.max_concurrency for s in stats),
        tools_used=merged_tools,
        has_timeout=any(s.has_timeout for s in stats),
        has_fallback=any(s.has_fallback for s in stats) or mode == "max",
        has_branching=any(s.has_branching for s in stats),
        has_refinement=any(s.has_refinement for s in stats),
        has_agent=any(s.has_agent for s in stats),
        max_agent_steps=max(s.max_agent_steps for s in stats),
        depth=max(s.depth for s in stats),
        node_count=sum(s.node_count for s in stats) + 1,
        max_refine_rounds=max(s.max_refine_rounds for s in stats),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Analyze — Main Analysis Function
# ═══════════════════════════════════════════════════════════════════════════════


def analyze(op: Op, depth: int = 0) -> Stats:
    """
    Analyze an Op AST without executing it.
    
    Returns bounds on resource usage (min/max for LLM calls, etc.)

    Args:
        op: The Op AST to analyze
        depth: Current nesting depth (internal use)

    Returns:
        Stats with min/max bounds and metadata
    """
    match op:
        # ─── Core LLM Operations ───────────────────────────────────────────
        
        case Ask():
            return Stats(
                llm_calls=Bounds.exact(1),
                parallel_branches=1,
                max_concurrency=1,
                tools_used=frozenset(),
                has_timeout=False,
                has_fallback=False,
                has_branching=False,
                has_refinement=False,
                has_agent=False,
                max_agent_steps=0,
                depth=depth,
                node_count=1,
                max_refine_rounds=0,
            )
        
        case Agent(tools=tools, max_steps=max_steps):
            tool_names: frozenset[str] = frozenset(t.name for t in tools)
            return Stats(
                llm_calls=Bounds.range(1, max_steps + 1),  # min=1 (immediate answer), max=all steps+final
                parallel_branches=1,
                max_concurrency=1,
                tools_used=tool_names,
                has_timeout=False,
                has_fallback=False,
                has_branching=False,
                has_refinement=False,
                has_agent=True,
                max_agent_steps=max_steps,
                depth=depth,
                node_count=1,
                max_refine_rounds=0,
            )
        
        # ─── Functor / Monad ───────────────────────────────────────────────
        
        case Pure():
            return _empty_stats(depth)

        case Map(inner=inner) | Tap(inner=inner) | MapErr(inner=inner):
            s = analyze(inner, depth + 1)
            return Stats(
                llm_calls=s.llm_calls,
                parallel_branches=s.parallel_branches,
                max_concurrency=s.max_concurrency,
                tools_used=s.tools_used,
                has_timeout=s.has_timeout,
                has_fallback=s.has_fallback,
                has_branching=s.has_branching,
                has_refinement=s.has_refinement,
                has_agent=s.has_agent,
                max_agent_steps=s.max_agent_steps,
                depth=s.depth,
                node_count=s.node_count + 1,
                max_refine_rounds=s.max_refine_rounds,
            )

        case Then(inner=inner):
            s = analyze(inner, depth + 1)
            # NOTE: Then represents monadic bind — the continuation (f) creates
            # new Op at runtime. We can't statically know what it will be.
            # Conservative estimate: at least 1 more LLM call possible.
            return Stats(
                llm_calls=Bounds(s.llm_calls.min, s.llm_calls.max + 1),
                parallel_branches=s.parallel_branches,
                max_concurrency=s.max_concurrency,
                tools_used=s.tools_used,
                has_timeout=s.has_timeout,
                has_fallback=s.has_fallback,
                has_branching=s.has_branching,
                has_refinement=s.has_refinement,
                has_agent=s.has_agent,
                max_agent_steps=s.max_agent_steps,
                depth=s.depth,
                node_count=s.node_count + 1,
                max_refine_rounds=s.max_refine_rounds,
            )
        
        # ─── Parallelism ───────────────────────────────────────────────────
        
        case Parallel(ops=ops):
            if not ops:
                return _empty_stats(depth)
            
            stats = [analyze(o, depth + 1) for o in ops]
            merged = _merge_stats(stats, mode="parallel")
            return Stats(
                llm_calls=merged.llm_calls,
                parallel_branches=len(ops),
                max_concurrency=len(ops),
                tools_used=merged.tools_used,
                has_timeout=merged.has_timeout,
                has_fallback=merged.has_fallback,
                has_branching=merged.has_branching,
                has_refinement=merged.has_refinement,
                has_agent=merged.has_agent,
                max_agent_steps=merged.max_agent_steps,
                depth=merged.depth,
                node_count=merged.node_count,
                max_refine_rounds=merged.max_refine_rounds,
            )
        
        case Batch(items=items, handler=handler, concurrency=concurrency):
            if not items:
                return _empty_stats(depth)
            
            # Analyze one sample to understand the pattern
            sample_op = handler(items[0])
            s = analyze(sample_op, depth + 1)
            n = len(items)
            
            return Stats(
                llm_calls=s.llm_calls * n,
                parallel_branches=min(n, concurrency),
                max_concurrency=concurrency,
                tools_used=s.tools_used,
                has_timeout=s.has_timeout,
                has_fallback=s.has_fallback,
                has_branching=s.has_branching,
                has_refinement=s.has_refinement,
                has_agent=s.has_agent,
                max_agent_steps=s.max_agent_steps,
                depth=s.depth,
                node_count=s.node_count + 1,
                max_refine_rounds=s.max_refine_rounds,
            )
        
        # ─── Error Handling ────────────────────────────────────────────────
        
        case Fallback(primary=primary, fallbacks=fallbacks):
            primary_s = analyze(primary, depth + 1)
            fallback_stats = [analyze(fb, depth + 1) for fb in fallbacks]

            all_stats = [primary_s, *fallback_stats]
            merged = _merge_stats(all_stats, mode="max")

            return Stats(
                llm_calls=merged.llm_calls,
                parallel_branches=merged.parallel_branches,
                max_concurrency=merged.max_concurrency,
                tools_used=merged.tools_used,
                has_timeout=merged.has_timeout,
                has_fallback=True,
                has_branching=merged.has_branching,
                has_refinement=merged.has_refinement,
                has_agent=merged.has_agent,
                max_agent_steps=merged.max_agent_steps,
                depth=merged.depth,
                node_count=merged.node_count,
                max_refine_rounds=merged.max_refine_rounds,
            )
        
        case Timeout(inner=inner):
            s = analyze(inner, depth + 1)
            return Stats(
                llm_calls=s.llm_calls,
                parallel_branches=s.parallel_branches,
                max_concurrency=s.max_concurrency,
                tools_used=s.tools_used,
                has_timeout=True,
                has_fallback=s.has_fallback,
                has_branching=s.has_branching,
                has_refinement=s.has_refinement,
                has_agent=s.has_agent,
                max_agent_steps=s.max_agent_steps,
                depth=s.depth,
                node_count=s.node_count + 1,
                max_refine_rounds=s.max_refine_rounds,
            )
        
        # ─── Control Flow ──────────────────────────────────────────────────
        
        case When(inner=inner):
            s_inner = analyze(inner, depth + 1)
            # NOTE: We can't analyze then/otherwise branches statically because
            # they're functions that create Op at runtime. Conservative: +1 max.
            return Stats(
                llm_calls=Bounds(s_inner.llm_calls.min, s_inner.llm_calls.max + 1),
                parallel_branches=s_inner.parallel_branches,
                max_concurrency=s_inner.max_concurrency,
                tools_used=s_inner.tools_used,
                has_timeout=s_inner.has_timeout,
                has_fallback=s_inner.has_fallback,
                has_branching=True,
                has_refinement=s_inner.has_refinement,
                has_agent=s_inner.has_agent,
                max_agent_steps=s_inner.max_agent_steps,
                depth=s_inner.depth,
                node_count=s_inner.node_count + 1,
                max_refine_rounds=s_inner.max_refine_rounds,
            )
        
        # ─── Refinement / Selection ────────────────────────────────────────
        
        case Refine(max_rounds=max_rounds):
            return Stats(
                llm_calls=Bounds.range(1, max_rounds),  # min=1 if immediately satisfied
                parallel_branches=1,
                max_concurrency=1,
                tools_used=frozenset(),
                has_timeout=False,
                has_fallback=False,
                has_branching=False,
                has_refinement=True,
                has_agent=False,
                max_agent_steps=0,
                depth=depth,
                node_count=1,
                max_refine_rounds=max_rounds,
            )
        
        case BestOf(inner=inner, n=n):
            s = analyze(inner, depth + 1)
            return Stats(
                llm_calls=s.llm_calls * n,
                parallel_branches=n,
                max_concurrency=n,
                tools_used=s.tools_used,
                has_timeout=s.has_timeout,
                has_fallback=s.has_fallback,
                has_branching=s.has_branching,
                has_refinement=s.has_refinement,
                has_agent=s.has_agent,
                max_agent_steps=s.max_agent_steps,
                depth=s.depth,
                node_count=s.node_count + 1,
                max_refine_rounds=s.max_refine_rounds,
            )
        
        case Ensure(inner=inner):
            s = analyze(inner, depth + 1)
            return Stats(
                llm_calls=s.llm_calls,
                parallel_branches=s.parallel_branches,
                max_concurrency=s.max_concurrency,
                tools_used=s.tools_used,
                has_timeout=s.has_timeout,
                has_fallback=s.has_fallback,
                has_branching=s.has_branching,
                has_refinement=s.has_refinement,
                has_agent=s.has_agent,
                max_agent_steps=s.max_agent_steps,
                depth=s.depth,
                node_count=s.node_count + 1,
                max_refine_rounds=s.max_refine_rounds,
            )

    # NOTE: Этот код недостижим если все Op-типы обработаны выше.
    # Pyright strict mode требует exhaustive match, но type alias Op
    # гарантирует полноту. Оставляем для будущих расширений.
    raise ValueError(f"Unknown Op type: {type(op)}")  # pragma: no cover


# ═══════════════════════════════════════════════════════════════════════════════
# Warnings — Detect Potential Issues
# ═══════════════════════════════════════════════════════════════════════════════


class Severity(Enum):
    """Warning severity levels."""
    HINT = auto()     # suggestion, not a problem
    WARNING = auto()  # potential issue
    ERROR = auto()    # likely bug


@dataclass(frozen=True)
class Warning:
    """A warning about potential issues in the Op AST."""
    severity: Severity
    code: str
    message: str
    location: str  # path in AST, e.g. "Fallback.primary.Agent"

    def __str__(self) -> str:
        prefix = {
            Severity.HINT: "💡",
            Severity.WARNING: "⚠️",
            Severity.ERROR: "❌",
        }[self.severity]
        return f"{prefix} [{self.code}] {self.message} (at {self.location})"


def warnings(op: Op, path: str = "root") -> list[Warning]:
    """
    Analyze Op AST for potential issues and anti-patterns.

    Returns list of warnings sorted by severity (errors first).
    """
    result: list[Warning] = []

    def warn(severity: Severity, code: str, message: str) -> None:
        result.append(Warning(severity, code, message, path))

    match op:
        case Agent(tools=tools, max_steps=max_steps):
            if not tools:
                warn(Severity.WARNING, "W001", "Agent has no tools — consider using Ask instead")
            if max_steps > 50:
                warn(Severity.HINT, "H001", f"High max_steps ({max_steps}) may cause long runs")
            if max_steps < 2:
                warn(Severity.WARNING, "W002", f"max_steps={max_steps} is very low for an agent")

        case Parallel(ops=ops):
            if not ops:
                warn(Severity.ERROR, "E001", "Empty Parallel — no operations to run")
            elif len(ops) > 20:
                warn(Severity.WARNING, "W003", f"Many parallel branches ({len(ops)}) may hit rate limits")

            for i, child in enumerate(ops):
                result.extend(warnings(child, f"{path}.Parallel[{i}]"))

        case Batch(items=items, concurrency=concurrency):
            if not items:
                warn(Severity.HINT, "H002", "Empty Batch — nothing to process")
            if concurrency > 50:
                warn(Severity.WARNING, "W004", f"High concurrency ({concurrency}) may hit rate limits")

        case Fallback(primary=primary, fallbacks=fallbacks):
            if not fallbacks:
                warn(Severity.WARNING, "W005", "Fallback with no fallbacks — just use the primary")

            result.extend(warnings(primary, f"{path}.Fallback.primary"))
            for i, fb in enumerate(fallbacks):
                result.extend(warnings(fb, f"{path}.Fallback[{i}]"))

        case Timeout(inner=inner, seconds=seconds):
            if seconds < 1.0:
                warn(Severity.WARNING, "W006", f"Very short timeout ({seconds}s) for LLM call")
            if seconds > 300.0:
                warn(Severity.HINT, "H003", f"Long timeout ({seconds}s) — consider shorter with fallback")

            result.extend(warnings(inner, f"{path}.Timeout"))

        case Refine(max_rounds=max_rounds):
            if max_rounds < 2:
                warn(Severity.HINT, "H004", f"Refine with max_rounds={max_rounds} — consider Ask instead")
            if max_rounds > 10:
                warn(Severity.WARNING, "W007", f"High max_rounds ({max_rounds}) may be expensive")

        case BestOf(inner=inner, n=n):
            if n < 2:
                warn(Severity.WARNING, "W008", f"BestOf with n={n} — should be at least 2")
            if n > 10:
                warn(Severity.WARNING, "W009", f"High BestOf n={n} is expensive")

            result.extend(warnings(inner, f"{path}.BestOf"))

        case Map(inner=inner):
            result.extend(warnings(inner, f"{path}.Map"))

        case Then(inner=inner):
            result.extend(warnings(inner, f"{path}.Then"))

        case When(inner=inner):
            result.extend(warnings(inner, f"{path}.When"))

        case Tap(inner=inner):
            result.extend(warnings(inner, f"{path}.Tap"))

        case MapErr(inner=inner):
            result.extend(warnings(inner, f"{path}.MapErr"))

        case Ensure(inner=inner, error=error):
            if not error.strip():
                warn(Severity.WARNING, "W010", "Ensure with empty error message")
            result.extend(warnings(inner, f"{path}.Ensure"))

        case Ask() | Pure():
            pass  # leaf nodes, no issues

    # Sort by severity (errors first)
    result.sort(key=lambda w: w.severity.value, reverse=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Fmt — Pretty Printing
# ═══════════════════════════════════════════════════════════════════════════════


class FmtStyle(Enum):
    """Output style for fmt()."""
    TREE = auto()      # indented tree (default)
    COMPACT = auto()   # one-line summary
    MERMAID = auto()   # mermaid flowchart


def fmt(op: Op, style: FmtStyle = FmtStyle.TREE, indent: int = 0) -> str:
    """
    Pretty-print an Op AST for debugging.

    Args:
        op: The Op to format
        style: Output style (TREE, COMPACT, or MERMAID)
        indent: Current indentation level (internal use)
    """
    if style == FmtStyle.COMPACT:
        return _fmt_compact(op)
    elif style == FmtStyle.MERMAID:
        return _fmt_mermaid(op)
    else:
        return _fmt_tree(op, indent)


def _fmt_tree(op: Op, indent: int = 0) -> str:
    """Pretty-print as indented tree."""
    prefix = "  " * indent
    
    match op:
        case Ask(schema=schema):
            return f"{prefix}Ask({schema.__name__})"
        
        case Agent(tools=tools, schema=schema, max_steps=max_steps):
            tool_names = [t.name for t in tools]
            schema_name = schema.__name__ if schema else "None"
            return f"{prefix}Agent(tools={tool_names}, schema={schema_name}, max_steps={max_steps})"
        
        case Pure(value=value):
            val_repr = repr(value)
            if len(val_repr) > 50:
                val_repr = val_repr[:47] + "..."
            return f"{prefix}Pure({val_repr})"
        
        case Map(inner=inner):
            return f"{prefix}Map\n{_fmt_tree(inner, indent + 1)}"
        
        case Then(inner=inner):
            return f"{prefix}Then\n{_fmt_tree(inner, indent + 1)}"
        
        case Parallel(ops=ops):
            if not ops:
                return f"{prefix}Parallel[]"
            children = "\n".join(_fmt_tree(o, indent + 1) for o in ops)
            return f"{prefix}Parallel[\n{children}\n{prefix}]"
        
        case Batch(items=items, concurrency=concurrency):
            return f"{prefix}Batch(n={len(items)}, concurrency={concurrency})"
        
        case Fallback(primary=primary, fallbacks=fallbacks):
            primary_str = _fmt_tree(primary, indent + 1)
            fallback_strs = "\n".join(_fmt_tree(fb, indent + 1) for fb in fallbacks)
            return f"{prefix}Fallback\n{prefix}  primary:\n{primary_str}\n{prefix}  fallbacks:\n{fallback_strs}"
        
        case Timeout(inner=inner, seconds=seconds):
            return f"{prefix}Timeout({seconds}s)\n{_fmt_tree(inner, indent + 1)}"
        
        case When(inner=inner):
            return f"{prefix}When\n{_fmt_tree(inner, indent + 1)}"
        
        case Tap(inner=inner):
            return f"{prefix}Tap\n{_fmt_tree(inner, indent + 1)}"

        case MapErr(inner=inner):
            return f"{prefix}MapErr\n{_fmt_tree(inner, indent + 1)}"
        
        case Refine(schema=schema, max_rounds=max_rounds):
            return f"{prefix}Refine({schema.__name__}, max_rounds={max_rounds})"
        
        case BestOf(inner=inner, n=n):
            return f"{prefix}BestOf(n={n})\n{_fmt_tree(inner, indent + 1)}"
        
        case Ensure(inner=inner, error=error):
            err_repr = repr(error)
            if len(err_repr) > 30:
                err_repr = err_repr[:27] + "..."
            return f"{prefix}Ensure({err_repr})\n{_fmt_tree(inner, indent + 1)}"

    return f"{prefix}{type(op).__name__}(...)"  # pragma: no cover


def _fmt_compact(op: Op) -> str:
    """One-line summary format."""
    match op:
        case Ask(schema=schema):
            return f"ask({schema.__name__})"

        case Agent(tools=tools, max_steps=max_steps):
            return f"agent({len(tools)} tools, ≤{max_steps})"

        case Pure():
            return "pure"

        case Map(inner=inner):
            return f"map({_fmt_compact(inner)})"

        case Then(inner=inner):
            return f"then({_fmt_compact(inner)})"

        case Parallel(ops=ops):
            return f"parallel({len(ops)})"

        case Batch(items=items, concurrency=concurrency):
            return f"batch({len(items)}, c={concurrency})"

        case Fallback(primary=primary, fallbacks=fallbacks):
            return f"fallback({_fmt_compact(primary)}, +{len(fallbacks)})"

        case Timeout(inner=inner, seconds=seconds):
            return f"timeout({_fmt_compact(inner)}, {seconds}s)"

        case When(inner=inner):
            return f"when({_fmt_compact(inner)})"

        case Tap(inner=inner):
            return f"tap({_fmt_compact(inner)})"

        case MapErr(inner=inner):
            return f"mapErr({_fmt_compact(inner)})"

        case Refine(max_rounds=max_rounds):
            return f"refine(≤{max_rounds})"

        case BestOf(inner=inner, n=n):
            return f"bestOf({_fmt_compact(inner)}, n={n})"

        case Ensure(inner=inner):
            return f"ensure({_fmt_compact(inner)})"

    return type(op).__name__  # pragma: no cover


def _fmt_mermaid(op: Op) -> str:
    """Generate Mermaid flowchart syntax."""
    lines: list[str] = ["flowchart TD"]
    counter = [0]  # mutable counter for unique IDs

    def node_id() -> str:
        counter[0] += 1
        return f"n{counter[0]}"

    def add_node(op: Op, parent_id: str | None = None) -> str:
        nid = node_id()

        match op:
            case Ask(schema=schema):
                lines.append(f'    {nid}["Ask({schema.__name__})"]')

            case Agent(tools=tools, max_steps=max_steps):
                lines.append(f'    {nid}["Agent({len(tools)} tools, ≤{max_steps})"]')

            case Pure():
                lines.append(f'    {nid}["Pure"]')

            case Map(inner=inner):
                lines.append(f'    {nid}["Map"]')
                child_id = add_node(inner, nid)
                lines.append(f'    {nid} --> {child_id}')

            case Then(inner=inner):
                lines.append(f'    {nid}["Then"]')
                child_id = add_node(inner, nid)
                lines.append(f'    {nid} --> {child_id}')

            case Parallel(ops=ops):
                lines.append(f'    {nid}{{"Parallel({len(ops)})"}}')
                for child in ops:
                    child_id = add_node(child, nid)
                    lines.append(f'    {nid} --> {child_id}')

            case Batch(items=items, concurrency=concurrency):
                lines.append(f'    {nid}["Batch({len(items)}, c={concurrency})"]')

            case Fallback(primary=primary, fallbacks=fallbacks):
                lines.append(f'    {nid}{{"Fallback"}}')
                primary_id = add_node(primary, nid)
                lines.append(f'    {nid} -->|primary| {primary_id}')
                for i, fb in enumerate(fallbacks):
                    fb_id = add_node(fb, nid)
                    lines.append(f'    {nid} -->|fallback{i+1}| {fb_id}')

            case Timeout(inner=inner, seconds=seconds):
                lines.append(f'    {nid}["Timeout({seconds}s)"]')
                child_id = add_node(inner, nid)
                lines.append(f'    {nid} --> {child_id}')

            case When(inner=inner):
                lines.append(f'    {nid}{{"When"}}')
                child_id = add_node(inner, nid)
                lines.append(f'    {nid} --> {child_id}')

            case Tap(inner=inner):
                lines.append(f'    {nid}["Tap"]')
                child_id = add_node(inner, nid)
                lines.append(f'    {nid} --> {child_id}')

            case MapErr(inner=inner):
                lines.append(f'    {nid}["MapErr"]')
                child_id = add_node(inner, nid)
                lines.append(f'    {nid} --> {child_id}')

            case Refine(max_rounds=max_rounds):
                lines.append(f'    {nid}["Refine(≤{max_rounds})"]')

            case BestOf(inner=inner, n=n):
                lines.append(f'    {nid}["BestOf(n={n})"]')
                child_id = add_node(inner, nid)
                lines.append(f'    {nid} --> {child_id}')

            case Ensure(inner=inner):
                lines.append(f'    {nid}["Ensure"]')
                child_id = add_node(inner, nid)
                lines.append(f'    {nid} --> {child_id}')

            case _:  # pragma: no cover
                lines.append(f'    {nid}["{type(op).__name__}"]')

        if parent_id:
            pass  # connection handled by parent

        return nid

    add_node(op)
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Validate — Check for Errors
# ═══════════════════════════════════════════════════════════════════════════════


def validate(op: Op) -> list[Warning]:
    """
    Validate an Op AST and return only errors (not hints/warnings).

    Use this for CI/pre-commit checks.
    """
    all_warnings = warnings(op)
    return [w for w in all_warnings if w.severity == Severity.ERROR]


def is_valid(op: Op) -> bool:
    """Returns True if Op has no errors."""
    return len(validate(op)) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Exports
# ═══════════════════════════════════════════════════════════════════════════════


__all__ = (
    # Stats
    "Stats",
    "Bounds",
    # Analysis
    "analyze",
    "warnings",
    "validate",
    "is_valid",
    # Formatting
    "fmt",
    "FmtStyle",
    # Types
    "Warning",
    "Severity",
)
