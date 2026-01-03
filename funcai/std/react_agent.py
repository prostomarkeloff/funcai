"""ReAct agent — default agent implementation via composition."""

import json
from dataclasses import dataclass, field
import typing

from kungfu import Result, Ok, Error, LazyCoroResult, Option, Nothing, Some
from pydantic import BaseModel

from funcai.core.message import Message, assistant, tool_result
from funcai.core.dialogue import Dialogue
from funcai.core.provider import ABCAIProvider, AIResponse
from funcai.core.types import ToolCall
from funcai.agents.abc import ABCAgent, AgentResponse, AgentError
from funcai.agents.tool import Tool


def _make_error_coro[E](e: Error[E]):
    async def _error_coro():
        return e

    return _error_coro


def _make_assistant_message[S](response: AIResponse[S]) -> Message:
    """Create assistant message from response."""
    return assistant(
        text=response.message.text.unwrap_or_none(),
        tool_calls=list(response.tool_calls),
    )


def _serialize_value(value: typing.Any) -> typing.Any:
    """Recursively prepare value for JSON serialization."""

    match value:
        case BaseModel():
            return value.model_dump()
        case list():
            return [_serialize_value(item) for item in value]
        case dict():
            return {k: _serialize_value(v) for k, v in value.items()}
        case _:
            return value


def _serialize_tool_result(result: typing.Any) -> str:
    """Serialize tool result to string for LLM."""

    match result:
        case None:
            return "null"
        case str():
            return result
        case BaseModel():
            return result.model_dump_json()
        case _:
            return json.dumps(_serialize_value(result))


def _execute_tool_call(tools_map: dict[str, Tool]):
    """Curried tool executor."""

    async def execute(call: ToolCall) -> Message:
        tool_obj = tools_map.get(call.name)
        if tool_obj is None:
            return tool_result(
                tool_call_id=call.id, content=f"Error: Unknown tool {call.name}"
            )
        try:
            result = await tool_obj.execute(**call.arguments)
            content = _serialize_tool_result(result)
        except Exception as e:
            content = f"Error: {e}"
        return tool_result(tool_call_id=call.id, content=content)

    return execute


async def _execute_all_tools(
    calls: list[ToolCall],
    tools_map: dict[str, Tool],
) -> list[Message]:
    """Execute all tool calls, return tool_result messages."""
    executor = _execute_tool_call(tools_map)
    return [await executor(call) for call in calls]


def _build_final_response[S](
    message: Message,
    parsed: S,
    tool_calls_made: list[ToolCall],
    steps: int,
) -> AgentResponse[S]:
    """Construct final AgentResponse."""
    return AgentResponse(
        message=message,
        parsed=parsed,
        tool_calls_made=tool_calls_made,
        steps=steps,
    )


@dataclass(frozen=True)
class StepState:
    """Immutable state for one step of the loop."""

    messages: tuple[Message, ...]
    tool_calls_made: tuple[ToolCall, ...]
    step: int

    def with_messages(self, *new_msgs: Message) -> "StepState":
        return StepState(
            messages=self.messages + tuple(new_msgs),
            tool_calls_made=self.tool_calls_made,
            step=self.step,
        )

    def with_tool_calls(self, calls: list[ToolCall]) -> "StepState":
        return StepState(
            messages=self.messages,
            tool_calls_made=self.tool_calls_made + tuple(calls),
            step=self.step,
        )

    def next_step(self) -> "StepState":
        return StepState(
            messages=self.messages,
            tool_calls_made=self.tool_calls_made,
            step=self.step + 1,
        )


async def _run_step[E, S: BaseModel](
    state: StepState,
    provider: ABCAIProvider[E],
    tools: list[Tool],
    tools_map: dict[str, Tool],
    schema: Option[type[S]],
) -> Result[AgentResponse[S] | AgentResponse[None] | StepState, AgentError]:
    """
    Execute one step of agent loop.

    Returns:
        Ok(AgentResponse[S]) — done with schema
        Ok(AgentResponse[None]) — done without schema
        Ok(StepState) — continue, need more steps
        Error(AgentError) — failed
    """
    result = typing.cast(
        Result[AIResponse[None], E],
        await provider.send_messages(list(state.messages), tools=tools),
    )

    match result:
        case Error(e):
            return Error(AgentError(str(e)))

        case Ok(response) if not response.has_tool_calls:
            # Final answer — optionally parse with schema
            match schema:
                case Some(s):
                    final = await provider.send_messages(
                        list(state.messages) + [response.message],
                        schema=Some(s),
                    )
                    match final:
                        case Ok(r):
                            match r.parsed:
                                case Some(parsed):
                                    resp: AgentResponse[S] = _build_final_response(
                                        r.message,
                                        parsed,
                                        list(state.tool_calls_made),
                                        state.step + 1,
                                    )
                                    return Ok(resp)
                                case Nothing():
                                    return Error(AgentError("No parsed response"))
                        case Error(e):
                            return Error(AgentError(str(e)))
                case Nothing():
                    resp_none: AgentResponse[None] = _build_final_response(
                        response.message,
                        None,
                        list(state.tool_calls_made),
                        state.step + 1,
                    )
                    return Ok(resp_none)

        case Ok(response):
            # Has tool calls — execute and continue
            assistant_msg = _make_assistant_message(response)
            tool_messages = await _execute_all_tools(response.tool_calls, tools_map)

            new_state = (
                state.with_messages(assistant_msg, *tool_messages)
                .with_tool_calls(response.tool_calls)
                .next_step()
            )
            return Ok(new_state)


def _loop[E, S: BaseModel](
    state: StepState,
    provider: ABCAIProvider[E],
    tools: list[Tool],
    tools_map: dict[str, Tool],
    schema: Option[type[S]],
    max_steps: int,
) -> LazyCoroResult[AgentResponse[S] | AgentResponse[None], AgentError]:
    """Recursive loop via composition."""

    def continue_or_done(
        result: AgentResponse[S] | AgentResponse[None] | StepState,
    ) -> LazyCoroResult[AgentResponse[S] | AgentResponse[None], AgentError]:
        match result:
            case AgentResponse() as response:
                # Done — use pure to lift value into LazyCoroResult
                return LazyCoroResult.pure(response)
            case StepState() as next_state if next_state.step >= max_steps:
                # Max steps exceeded
                return LazyCoroResult[
                    AgentResponse[S] | AgentResponse[None], AgentError
                ](
                    _make_error_coro(
                        Error(AgentError(f"Max steps ({max_steps}) exceeded"))
                    )
                )
            case StepState() as next_state:
                # Continue recursively
                return _loop(next_state, provider, tools, tools_map, schema, max_steps)

    return LazyCoroResult(
        lambda: _run_step(state, provider, tools, tools_map, schema)
    ).then(continue_or_done)


@dataclass
class ReActAgent[E, S: BaseModel](ABCAgent[E, S]):
    """
    ReAct agent — Reason + Act loop via pure composition.

    Generic parameters:
        E — provider error type
        S — schema type (or None if no schema)

    Flow:
        1. Send messages to LLM
        2. If LLM returns tool_calls → execute tools → loop
        3. If no tool_calls → return response
        4. Optionally parse final response into schema
    """

    provider: ABCAIProvider[E]
    tools: list[Tool]
    max_steps: int = 10
    schema: Option[type[S]] = field(default_factory=Nothing)

    def run(
        self, dialogue: Dialogue
    ) -> LazyCoroResult[AgentResponse[S] | AgentResponse[None], AgentError]:
        """Run agent — returns lazy computation for composition."""
        initial_state = StepState(
            messages=tuple(dialogue.messages),
            tool_calls_made=(),
            step=0,
        )
        tools_map = {t.name: t for t in self.tools}
        return _loop(
            initial_state,
            self.provider,
            self.tools,
            tools_map,
            self.schema,
            self.max_steps,
        )


__all__ = ["ReActAgent"]
