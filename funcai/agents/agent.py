from typing import Any, overload, cast

from kungfu import LazyCoroResult, Nothing, Some
from pydantic import BaseModel

from funcai.core.dialogue import Dialogue
from funcai.core.provider import ABCAIProvider
from funcai.agents.abc import ABCAgent, AgentResponse, AgentError
from funcai.agents.tool import Tool
from funcai.std.react_agent import ReActAgent


@overload
def agent[E, S: BaseModel](
    dialogue: Dialogue,
    *,
    use: ABCAgent[E, S],
) -> LazyCoroResult[AgentResponse[S] | AgentResponse[None], AgentError]: ...


@overload
def agent[E](
    dialogue: Dialogue,
    provider: ABCAIProvider[E],
    tools: list[Tool],
    max_steps: int = ...,
) -> LazyCoroResult[AgentResponse[None], AgentError]: ...


@overload
def agent[E, S: BaseModel](
    dialogue: Dialogue,
    provider: ABCAIProvider[E],
    tools: list[Tool],
    max_steps: int = ...,
    *,
    schema: type[S],
) -> LazyCoroResult[AgentResponse[S], AgentError]: ...


def agent[E, S: BaseModel](
    dialogue: Dialogue,
    provider: ABCAIProvider[E] | None = None,
    tools: list[Tool] | None = None,
    max_steps: int = 10,
    *,
    schema: type[S] | Nothing = Nothing(),
    use: ABCAgent[Any, Any] | None = None,
) -> LazyCoroResult[AgentResponse[S] | AgentResponse[None], AgentError]:
    """
    Run agent loop with tools.

    With default ReActAgent:
        >>> result = await agent(dialogue, provider, tools=[search])
        >>> result.unwrap().message.text

    With structured output:
        >>> result = await agent(dialogue, provider, tools=[search], schema=Answer)
        >>> result.unwrap().parsed  # → Answer instance

    With custom agent instance:
        >>> my_agent = TreeOfThoughtsAgent(provider=provider, tools=[search], ...)
        >>> result = await agent(dialogue, use=my_agent)

    Args:
        dialogue: The conversation to process
        provider: LLM provider (not needed if `use` is provided)
        tools: List of tools (not needed if `use` is provided)
        max_steps: Maximum iterations (not needed if `use` is provided)
        schema: Optional Pydantic model for structured output
        use: Pre-configured agent instance (if provided, ignores provider/tools/max_steps)
    """
    if use is not None:
        # Use pre-configured agent instance
        result = use.run(dialogue)
        return cast(
            LazyCoroResult[AgentResponse[S] | AgentResponse[None], AgentError], result
        )

    # Create ReActAgent with provided params
    if provider is None:
        raise ValueError("provider is required when not using a pre-configured agent")
    if tools is None:
        raise ValueError("tools is required when not using a pre-configured agent")

    agent_instance = ReActAgent(
        provider=provider,
        tools=tools,
        max_steps=max_steps,
        schema=Some(schema) if schema else schema,
    )
    result = agent_instance.run(dialogue)
    return result


__all__ = ["agent"]
