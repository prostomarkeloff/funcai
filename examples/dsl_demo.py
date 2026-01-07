"""
AI Judge: Multi-agent debate with consensus.

Three AI personas argue a case, then a judge synthesizes the verdict.
Static analysis reveals the execution structure before any API call.
"""

from pydantic import BaseModel
from funcai import Dialogue, message
from funcai.std.dsl import AI, analyze, fmt


class Argument(BaseModel):
    position: str
    confidence: float


class Verdict(BaseModel):
    decision: str
    reasoning: str


def persona(name: str, stance: str, topic: str) -> AI[Argument]:
    return AI.ask(
        Dialogue([
            message.system(text=f"You are {name}. Stance: {stance}. Be persuasive."),
            message.user(text=topic),
        ]),
        Argument,
    )


def synthesize(args: list[Argument]) -> AI[Verdict]:
    summary = "\n".join(f"- {a.position} (conf: {a.confidence})" for a in args)
    return AI.ask(
        Dialogue([
            message.system(text="You are an impartial judge. Weigh all arguments."),
            message.user(text=f"Arguments:\n{summary}\n\nDeliver verdict."),
        ]),
        Verdict,
    )


MISTRIAL = Verdict(decision="Mistrial", reasoning="Court could not decide.")


def judge(topic: str) -> AI[Verdict]:
    return (
        AI.parallel(
            persona("Optimist", "Progress is good", topic),
            persona("Skeptic", "Evidence over hype", topic),
            persona("Pragmatist", "Balance tradeoffs", topic),
        )
        .best_of(2, key=lambda args: sum(a.confidence for a in args))
        .then(synthesize)
        .ensure(lambda v: len(v.reasoning) > 20, "Needs reasoning")
        .timeout(60.0)
        .fallback(AI.pure(MISTRIAL))
    )


if __name__ == "__main__":
    from funcai.std.dsl import warnings, FmtStyle
    
    case = judge("Should AI be open-source by default?")
    s = analyze(case.op)
    
    # Quick summary with new .pretty() method
    print(f"📊 Stats: {s.pretty()}\n")
    
    # Detailed stats
    print(f"   LLM calls: {s.llm_calls} (min..max)")
    print(f"   Parallel branches: {s.parallel_branches}")
    print(f"   Complexity score: {s.complexity_score}")
    print(f"   AST nodes: {s.node_count}")
    print(f"   Deterministic: {s.is_deterministic}")
    print()
    
    # Warnings detection
    warns = warnings(case.op)
    if warns:
        print("⚠️ Warnings:")
        for w in warns:
            print(f"   {w}")
        print()
    
    # Tree format (default)
    print("🌳 AST Tree:")
    print(fmt(case.op))
    print()
    
    # Compact format
    print(f"📝 Compact: {fmt(case.op, FmtStyle.COMPACT)}")
    print()
    
    # Mermaid diagram
    print("📈 Mermaid diagram:")
    print(fmt(case.op, FmtStyle.MERMAID))
