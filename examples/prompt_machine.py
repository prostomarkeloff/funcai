import asyncio

from kungfu import Error, Nothing, Ok, Option
from funcai import Dialogue, message, combinators
from funcai.std.openai_provider import OpenAIProvider

DEFAULT_PROVIDER = OpenAIProvider(model="gpt-5")


def new_dialogue() -> Dialogue:
    return Dialogue(
        [message.system(text="You're prompt machine designed to answer to any prompts")]
    )


def new_prompt(d: Dialogue, prompt: str) -> Dialogue:
    return combinators.append(d, message.user(text=prompt))


async def ask(d: Dialogue, prompt: str) -> str:
    text: Option[str] = Nothing()
    match await new_prompt(d, prompt).interpret(DEFAULT_PROVIDER):
        case Ok(answer):
            text = answer.message.text
        case Error(_):
            pass

    return text.unwrap_or("Can't answer to your question")


async def main():
    print("Prompt!")
    dialogue = new_dialogue()

    try:
        while prompt := input(">"):
            answer = await ask(dialogue, prompt)
            print(f">> {answer}")
    except KeyboardInterrupt:
        print("Goodbye!")


asyncio.run(main())
