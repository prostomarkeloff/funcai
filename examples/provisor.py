import asyncio

from funcai import Dialogue, message, agent, tool
from funcai.combinators import append
from funcai.std.providers.openai import OpenAIProvider
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
    return Dialogue(
        [
            message.system(
                text="You're a provisor. Store and retrieve items using tools."
            )
        ]
    )


async def ask(dialogue: Dialogue, prompt: str) -> str:
    extended = append(dialogue, message.user(text=prompt))
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
        try:
            prompt = input("> ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if prompt.lower() == "quit":
            break

        answer = await ask(dialogue, prompt)
        print(f">> {answer}\n")


asyncio.run(main())
