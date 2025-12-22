"""Abstract agent interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from kungfu import LazyCoroResult

from funcai.core.message import Message
from funcai.core.dialogue import Dialogue
from funcai.core.types import ToolCall


@dataclass(frozen=True)
class AgentError:
    """Error during agent execution."""

    message: str


@dataclass
class AgentResponse[S]:
    """
    Final response from agent.

    Without schema: S = None, parsed = None
    With schema: S = YourModel, parsed = YourModel instance
    """

    message: Message
    parsed: S
    tool_calls_made: list[ToolCall] = field(default_factory=list[ToolCall])
    steps: int = 0


class ABCAgent[E, S](ABC):
    """
    Abstract agent interface.

    Generic parameters:
        E — provider error type
        S — schema type for parsed output (None if no schema)

    Implementations:
        - ReActAgent (reason + act) — default
        - Custom agent loops
    """

    @abstractmethod
    def run(
        self,
        dialogue: Dialogue,
    ) -> LazyCoroResult[AgentResponse[S] | AgentResponse[None], AgentError]:
        """
        Run agent on dialogue.

        Returns LazyCoroResult for composition with .then()/.map()
        """
        ...
