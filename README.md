# funcai

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
    message.user(text="What is 2^10 + 156?")
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
- **Combinators.** `parallel`, `fallback`, `refine`, `batch` — complex flows from simple pieces.
- **Easy to hack.** `ABCAIProvider` and `ABCAgent` — swap providers, build custom loops.
- **Zero magic.** No callbacks, no middleware, no runtime introspection.
- **Python 3.14+** with native generics.

## Installation

```bash
uv add https://github.com/prostomarkeloff/funcai.git
```

## The Core Idea

Everything in funcai is built from one primitive: **Dialogue**.

- `dialogue.interpret(provider)` — simple interpreter. One request, one response.
- `agent(dialogue, provider, tools)` — complex interpreter. ReAct loop with tool calls.
- `combinators.refine(dialogue, ...)` — iterative interpreter. Loops until condition met.

They all take a `Dialogue`, compose operations, return `LazyCoroResult`. The difference is complexity of interpretation, not the abstraction. Mix and match freely:

```python
result = await combinators.fallback(
    agent(dialogue, gpt4, tools=[search]),   # complex interpreter
    dialogue.interpret(claude, Answer),       # simple interpreter as fallback
)
```

## Dialogue & Interpretation

A `Dialogue` is just a list of messages. `interpret` is the simplest interpreter:

```python
from funcai import Dialogue, message, combinators
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float

dialogue = Dialogue([
    message.system(text="Analyze sentiment."),
    message.user(text="This product is amazing!"),
])

# Without schema → AIResponse
response = await dialogue.interpret(provider)

# With schema → validated Pydantic model
analysis = await dialogue.interpret(provider, Analysis)

# Pure operations
extended = combinators.append(dialogue, message.user(text="Be concise."))
```

Composition via `.map()` and `.then()`:

```python
result = (
    dialogue.interpret(provider, Analysis)
    .map(lambda a: f"{a.sentiment} ({a.confidence:.0%})")
    .map_err(lambda e: f"API error: {e}")
)
```

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

`agent` is a complex interpreter — it runs the ReAct loop (Reason + Act), executing tools until the LLM decides it's done:

```python
# Default ReActAgent
result = await agent(dialogue, provider, tools=[search, calculate])

# With structured output — final answer parsed into schema
result = await agent(dialogue, provider, tools=[search], schema=Answer)

# With custom agent instance
my_agent = TreeOfThoughtsAgent(provider=provider, tools=[search], max_steps=20)
result = await agent(dialogue, use=my_agent)
```

Same `Dialogue` in, `LazyCoroResult` out. Just a more sophisticated interpretation.

## Combinators

Higher-order functions that compose interpretations:

```python
from funcai import combinators

# Run concurrently
results = await combinators.parallel(
    d1.interpret(provider, Summary),
    d2.interpret(provider, Summary),
)

# Fallback on failure
result = await combinators.fallback(
    dialogue.interpret(gpt4, Answer),
    dialogue.interpret(claude, Answer),
)

# Conditional branching
result = await combinators.when(
    dialogue.interpret(provider, Analysis),
    condition=lambda a: a.confidence > 0.8,
    then=lambda a: proceed(a),
    otherwise=lambda a: clarify(a),
)

# Iterate until satisfied
essay = await combinators.refine(
    dialogue, provider, Essay,
    until=lambda e: e.word_count >= 500,
    feedback=lambda e: f"Too short ({e.word_count}). Expand.",
)

# Pick best from n runs
best = await combinators.best_of(
    dialogue.interpret(provider, Analysis),
    n=3,
    key=lambda a: a.confidence,
)

# Bounded parallelism
summaries = await combinators.batch(
    items=documents,
    handler=lambda doc: summarize(doc),
    concurrency=5,
)

# Timeout
result = await combinators.timeout(
    dialogue.interpret(provider, Answer),
    seconds=30.0,
)
```

All return `LazyCoroResult`. Chain infinitely.

## Extensibility

Implement `ABCAIProvider` to add new LLM backends:

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
        # Your implementation
        ...
```

Implement `ABCAgent` for custom agent loops, then pass an instance via `use`:

```python
from funcai.agents import ABCAgent, AgentResponse, AgentError
from kungfu import LazyCoroResult

class TreeOfThoughtsAgent[E, S](ABCAgent[E, S]):
    provider: ABCAIProvider[E]
    tools: list[Tool]
    max_steps: int = 10
    schema: Option[type[S]] = Nothing()

    def run(self, dialogue: Dialogue) -> LazyCoroResult[AgentResponse[S], AgentError]:
        # Your implementation
        ...

# Use it
my_agent = TreeOfThoughtsAgent(provider=provider, tools=[search], max_steps=20)
result = await agent(dialogue, use=my_agent)
```

## Result Types

funcai uses [kungfu](https://github.com/timoniq/kungfu):

```python
from kungfu import Result, Ok, Error, Option, Some, Nothing

# Result[T, E] = Ok[T] | Error[E]
# Option[T] = Some[T] | Nothing

match result:
    case Ok(response):
        match response.message.text:
            case Some(text): print(text)
            case Nothing(): print("(no text)")
    case Error(e):
        print(f"Failed: {e}")
```

## vs LangChain

| LangChain | funcai |
|-----------|--------|
| `Chain.invoke(input)` | `dialogue.interpret(provider)` |
| `CallbackHandler.on_*()` | `.map()` / `combinators.tap()` |
| `Memory.save_context()` | Immutable `Dialogue` |
| `AgentExecutor.run()` | `agent(dialogue, provider, tools)` |
| Exception handling | `Result[T, E]` |
| Config objects | Function composition |

## For the Curious

The architecture is simple: `Dialogue` is the AST, interpreters evaluate it.

- `interpret` — trivial interpreter (single LLM call)
- `agent` — recursive interpreter (ReAct loop)
- `refine` — iterative interpreter (feedback loop)

Category-theoretically:

- `Dialogue.interpret()` is a functor: `Dial → Comp`
- `LazyCoroResult[T, E]` is the Kleisli category for `Result`
- `.then()` is monadic bind: `m a → (a → m b) → m b`
- Combinators are natural transformations

If you know interpreters or category theory, you already know funcai.

## Real Example

Interactive provisor — stores and retrieves items via LLM:

```python
import asyncio
from funcai import Dialogue, message, combinators, agent, tool
from funcai.std.openai_provider import OpenAIProvider
from kungfu import Ok, Error

provider = OpenAIProvider(model="gpt-4o")

items: dict[str, str] = {}

@tool("Get item from storage by name, returns None if not exists")
def retrieve(item_name: str) -> str | None:
    return items.get(item_name)

@tool("Store an item with given name and description")
def put(item_name: str, description: str) -> str:
    items[item_name] = description
    return f"Stored '{item_name}'"

def new_dialogue() -> Dialogue:
    return Dialogue([
        message.system(text="You're a provisor. Store and retrieve items using tools.")
    ])

async def ask(dialogue: Dialogue, prompt: str) -> str:
    extended = combinators.append(dialogue, message.user(text=prompt))
    result = await agent(extended, provider, tools=[put, retrieve])
    match result:
        case Ok(r):
            return r.message.text.unwrap_or("Done")
        case Error(e):
            return f"Error: {e.message}"

async def main():
    dialogue = new_dialogue()
    print("Provisor ready. Type 'quit' to exit.\n")
    
    while True:
        prompt = input("> ")
        if prompt.lower() == "quit":
            break
        answer = await ask(dialogue, prompt)
        print(f">> {answer}\n")

asyncio.run(main())
```

```
> Store a red apple with description "fresh from the garden"
>> Stored 'red apple'

> What do we have stored about apples?
>> The red apple is fresh from the garden.

> quit
```

---

**Requirements:** Python 3.14+, [kungfu](https://github.com/timoniq/kungfu)

**Author:** [@prostomarkeloff](https://github.com/prostomarkeloff)

**License:** MIT
