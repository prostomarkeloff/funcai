import asyncio

from kungfu import Error, Ok
from funcai import Dialogue, agent, message, tool
from funcai.std.providers.openai import OpenAIProvider

provider = OpenAIProvider(model="gpt-4o")


@tool("Evaluate a mathematical expression")
def calculate(expression: str) -> float:
    return eval(expression)


async def main():
    dialogue = Dialogue(
        [
            message.system(text="You're a helpful calculator."),
            message.user(text="What is 2**10 + 156?"),
        ]
    )

    result = await agent(dialogue, provider, tools=[calculate])

    match result:
        case Ok(response):
            print(response.message.text.unwrap())
        case Error(e):
            print(f"Failed: {e}")


asyncio.run(main())
