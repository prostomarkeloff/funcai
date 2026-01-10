<h1 align="center">funcai</h1>

<p align="center">
  <em><strong>「 intelligence is a function 」</strong></em>
</p>

<p align="center">
  Compose LLM interactions like functions, not inheritance chains.
</p>

---

```python
from funcai import Dialogue, message, agent, tool, OpenAIProvider
from kungfu import Ok, Error

provider = OpenAIProvider(model="gpt-4o")

@tool("Evaluate a mathematical expression")
def calculate(expression: str) -> float:
    return eval(expression)

dialogue = Dialogue([
    message.system(text="You're a calculator."),
    message.user(text="What is 2**10 + 156?")
])

match await agent(dialogue, provider, tools=[calculate]):
    case Ok(response):
        print(response.message.text.unwrap())
    case Error(e):
        print(f"Failed: {e}")
```

`agent` returns `LazyCoroResult[AgentResponse, AgentError]` — lazy, typed, composable.

---

## Why funcai?

- **Lazy by design.** Nothing executes until you `await`. Compose first, run later.
- **Errors as values.** `Result[T, E]` instead of try/except spaghetti.
- **Two composition styles.** Direct combinators or typed DSL with static analysis.
- **Analyzable.** Know LLM call bounds *before* execution.
- **Easy to hack.** `ABCAIProvider` and `ABCAgent` — swap providers, build custom loops.
- **Zero magic.** No callbacks, no middleware, no runtime introspection.
- **Python 3.14+** with native generics.

## Installation

```bash
uv add git+https://github.com/prostomarkeloff/funcai.git
```

---

## Two Ways to Compose

funcai offers two complementary approaches:

### 1. Direct Combinators — Simple & Explicit

Use [combinators.py](https://github.com/prostomarkeloff/combinators.py) for straightforward composition:

```python
from combinators import flow, fallback, parallel, timeout, batch
from funcai import Dialogue, message, agent

# Fluent pipeline with retry + timeout
result = await (
    flow(dialogue.interpret(provider, Analysis))
    .retry(times=3, delay_seconds=0.5)
    .timeout(seconds=30.0)
    .compile()
)

# Parallel execution
results = await parallel(
    d1.interpret(provider, Summary),
    d2.interpret(provider, Summary),
)

# Fallback chain
result = await fallback(
    agent(dialogue, gpt4, tools=[search]),
    dialogue.interpret(claude, Answer),
)
```

### 2. Typed DSL — Analyze Before Execute

Build programs as AST, analyze statically, then compile:

```python
from funcai.std.dsl import AI, analyze, fmt

# Build program as typed AST
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

# Analyze BEFORE running — know your costs
stats = analyze(program.op)
print(f"LLM calls: {stats.llm_calls}")      # e.g., "1..3"
print(f"Has timeout: {stats.has_timeout}")   # True
print(f"Complexity: {stats.complexity_score}")

# Pretty-print the AST
print(fmt(program.op))

# Compile and execute
result = await program.compile(provider)
```

---

## The DSL — Programs as Data

The DSL represents AI workflows as an Abstract Syntax Tree (AST). This unlocks:

### Static Analysis

```python
from funcai.std.dsl import AI, analyze, warnings, fmt, FmtStyle

program = (
    AI.parallel(
        AI.ask(d1, Summary),
        AI.ask(d2, Summary),
        AI.ask(d3, Summary),
    )
    .best_of(2, key=lambda xs: sum(len(s.text) for s in xs))
    .timeout(60.0)
    .fallback(AI.pure([]))
)

# Get bounds
stats = analyze(program.op)
print(stats.pretty())
# → "LLM:6..6 | ∥:3 | ⏱ | ↩ | ⚙3.5"

# Detect issues
for w in warnings(program.op):
    print(w)
# → ⚠️ [W003] Many parallel branches (3) may hit rate limits

# Visualize as Mermaid diagram
print(fmt(program.op, FmtStyle.MERMAID))
```

### Type-Safe Builder

```python
from funcai.std.dsl import AI
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float

class Summary(BaseModel):
    text: str
    word_count: int

# Types flow through the chain
program: AI[Summary] = (
    AI.ask(dialogue, Analysis)              # AI[Analysis]
    .then(lambda a: AI.ask(followup(a), Summary))  # AI[Summary]
    .map(lambda s: s.text)                  # AI[str] — oops, type error!
)  # ❌ pyright catches: AI[str] != AI[Summary]
```

### DSL Operations

```python
from funcai.std.dsl import AI

# Core constructors
AI.pure(value)                    # Lift value
AI.ask(dialogue, Schema)          # Single LLM call
AI.agent(dialogue, tools)         # ReAct agent loop
AI.agent(dialogue, tools, schema=T)  # Agent with structured output

# Composition
program.map(f)                    # Transform result
program.then(f)                   # Monadic bind (flatMap)
program.map_err(f)                # Transform error

# Parallelism
AI.parallel(p1, p2, p3)           # Run all, collect results
AI.batch(items, handler, concurrency=5)  # Bounded parallelism

# Error handling
program.fallback(alt1, alt2)      # Try alternatives on failure
program.timeout(30.0)             # Fail after N seconds

# Control flow
program.when(cond, then, otherwise)  # Conditional branching
program.tap(effect)               # Side effect without changing result

# Refinement
AI.refine(dialogue, Schema, until=pred, feedback=fn)  # Iterative improvement
program.best_of(n=3, key=score_fn)  # Run N times, pick best
program.ensure(check=pred, error="msg")  # Validate result
```

---

## Combinators

funcai uses [combinators.py](https://github.com/prostomarkeloff/combinators.py) for generic async combinators, plus domain-specific combinators for dialogue manipulation:

```python
# Generic (from combinators.py)
from combinators import (
    flow,           # Fluent pipeline builder
    parallel,       # Run concurrently
    fallback,       # Try secondary on failure
    fallback_chain, # Chain of fallbacks
    timeout,        # Time limit
    batch,          # Bounded parallelism
    best_of,        # Run N, pick best
    tap,            # Side effect
)

# Domain-specific (funcai)
from funcai.combinators import (
    refine,         # Iterative refinement with LLM feedback
    when,           # Conditional branching for Interp
    append,         # Add messages to Dialogue
    prepend,        # Prepend messages
    with_context,   # Inject context documents
)
```

### Examples

```python
from combinators import flow, fallback, batch
from funcai.combinators import refine, append

# Fluent pipeline
result = await (
    flow(dialogue.interpret(provider, Answer))
    .retry(times=3, delay_seconds=0.2)
    .timeout(seconds=30.0)
    .compile()
)

# Iterative refinement
essay = await refine(
    dialogue, provider, Essay,
    until=lambda e: e.word_count >= 500,
    feedback=lambda e: f"Too short ({e.word_count}). Expand.",
    max_rounds=3,
)

# Batch processing
summaries = await batch(
    documents,
    handler=lambda doc: summarize(doc, provider),
    concurrency=5,
)

# Dialogue manipulation
extended = append(dialogue, message.user(text="Be concise."))
```

---

## Tools & Agents

Tools are functions with a description for the LLM:

```python
from funcai import tool, agent

@tool("Search the knowledge base")
def search(query: str, top_k: int = 5) -> list[str]:
    return kb.search(query, top_k)

@tool("Evaluate math expressions")
def calculate(expression: str) -> float:
    return eval(expression)
```

`agent` runs the ReAct loop (Reason + Act):

```python
# Default ReActAgent
result = await agent(dialogue, provider, tools=[search, calculate])

# With structured output
result = await agent(dialogue, provider, tools=[search], schema=Answer)

# With custom agent
my_agent = TreeOfThoughtsAgent(provider=provider, tools=[search])
result = await agent(dialogue, use=my_agent)
```

---

## Extensibility

### Custom Providers

```python
from funcai.core import ABCAIProvider, AIResponse, Message
from kungfu import Result, Option, Nothing

class ClaudeProvider(ABCAIProvider[ClaudeError]):
    async def send_messages[S: BaseModel](
        self,
        messages: list[Message],
        *,
        schema: Option[type[S]] = Nothing(),
        tools: list[Tool] | None = None,
    ) -> Result[AIResponse[S], ClaudeError]:
        ...
```

### Custom Agents

```python
from funcai.agents import ABCAgent, AgentResponse, AgentError

class TreeOfThoughtsAgent[E, S](ABCAgent[E, S]):
    def run(self, dialogue: Dialogue) -> LazyCoroResult[AgentResponse[S], AgentError]:
        ...
```

---

## Result Types

funcai uses [kungfu](https://github.com/timoniq/kungfu):

```python
from kungfu import Result, Ok, Error, Option, Some, Nothing

match result:
    case Ok(response):
        match response.message.text:
            case Some(text): print(text)
            case Nothing(): print("(no text)")
    case Error(e):
        print(f"Failed: {e}")
```

---

## vs LangChain

| LangChain | funcai |
|-----------|--------|
| `Chain.invoke(input)` | `dialogue.interpret(provider)` |
| `CallbackHandler.on_*()` | `.map()` / `tap()` |
| `Memory.save_context()` | Immutable `Dialogue` |
| `AgentExecutor.run()` | `agent(dialogue, provider, tools)` |
| Exception handling | `Result[T, E]` |
| Config objects | Function composition |
| Runtime introspection | Static AST analysis |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         funcai                              │
├─────────────────────────────────────────────────────────────┤
│  Dialogue → interpret() → LazyCoroResult[T, E]              │
│           → agent()     → LazyCoroResult[AgentResponse, E]  │
├─────────────────────────────────────────────────────────────┤
│  DSL: AI[T] → analyze() → Stats                             │
│             → compile() → LazyCoroResult[T, E]              │
├─────────────────────────────────────────────────────────────┤
│  Combinators: parallel, fallback, timeout, batch, ...       │
│               (from combinators.py)                         │
└─────────────────────────────────────────────────────────────┘
```

Category-theoretically:

- `Dialogue.interpret()` is a functor: `Dial → Comp`
- `LazyCoroResult[T, E]` is the Kleisli category for `Result`
- `.then()` is monadic bind: `m a → (a → m b) → m b`
- DSL `Op` is a free monad over LLM operations
- `analyze()` is an algebra (catamorphism) over `Op`

---

## Quick Start Example

```python
import asyncio
from funcai import Dialogue, message, agent, tool
from funcai.combinators import append
from funcai.std.providers.openai import OpenAIProvider
from kungfu import Ok, Error

provider = OpenAIProvider(model="gpt-4o")

@tool("Store an item")
def store(name: str, value: str) -> str:
    items[name] = value
    return f"Stored '{name}'"

@tool("Retrieve an item")  
def get(name: str) -> str | None:
    return items.get(name)

items: dict[str, str] = {}

async def main():
    dialogue = Dialogue([
        message.system(text="You're a key-value store. Use tools.")
    ])
    
    query = append(dialogue, message.user(text="Store 'hello' = 'world'"))
    
    match await agent(query, provider, tools=[store, get]):
        case Ok(r): print(r.message.text.unwrap())
        case Error(e): print(f"Error: {e}")

asyncio.run(main())
```

---

**Requirements:** Python 3.14+, [kungfu](https://github.com/timoniq/kungfu), [combinators.py](https://github.com/prostomarkeloff/combinators.py)

**Author:** [@prostomarkeloff](https://github.com/prostomarkeloff)

**License:** MIT
