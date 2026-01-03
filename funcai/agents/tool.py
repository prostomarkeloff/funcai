"""Tool definition and @tool decorator."""

import inspect
from dataclasses import dataclass
from typing import Any, get_type_hints, overload, Callable

from pydantic import BaseModel, create_model
from kungfu import Option, Nothing, Some, from_optional


@dataclass(frozen=True)
class Tool:
    """A tool that can be called by an LLM."""

    name: str
    description: str
    parameters: type[BaseModel]
    fn: Callable[..., Any]
    return_type: Option[type[Any]]

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool with given arguments."""
        if inspect.iscoroutinefunction(self.fn):
            return await self.fn(**kwargs)
        return self.fn(**kwargs)


def _make_tool(fn: Callable[..., Any], description: str) -> Tool:
    """Internal: create Tool from function and description."""
    hints = get_type_hints(fn)
    return_type = from_optional(hints.pop("return", None))

    sig = inspect.signature(fn)
    fields: dict[str, Any] = {}

    for name, param in sig.parameters.items():
        annotation = hints.get(name, str)
        if param.default is inspect.Parameter.empty:
            fields[name] = (annotation, ...)
        else:
            fields[name] = (annotation, param.default)

    Parameters = create_model(f"{fn.__name__}_params", **fields)

    return Tool(
        name=fn.__name__,
        description=description,
        parameters=Parameters,
        fn=fn,
        return_type=return_type,
    )


@overload
def tool(description: str) -> Callable[[Callable[..., Any]], Tool]: ...


@overload
def tool(description: Callable[..., Any]) -> Tool: ...


def tool(
    description: str | Callable[..., Any],
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """
    Decorator to create a Tool from a function.

    The description is what the LLM reads to understand the tool.
    Type hints are used to generate the parameter schema via Pydantic.

    Usage with description (recommended):
        >>> @tool("Search the knowledge base for relevant documents")
        ... def search(query: str, top_k: int = 5) -> list[str]:
        ...     return retriever.search(query, top_k)

    Usage without description (uses empty string):
        >>> @tool
        ... def calculate(expression: str) -> float:
        ...     return eval(expression)
    """
    # Called as @tool (without parentheses) — description is actually the function
    if callable(description):
        return _make_tool(description, "")

    # Called as @tool("description") — return decorator
    def decorator(fn: Callable[..., Any]) -> Tool:
        return _make_tool(fn, description)

    return decorator
