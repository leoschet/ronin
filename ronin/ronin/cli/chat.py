import json

import click
from haystack.nodes import PromptNode
from loguru import logger

from ronin.assistants import AssistantRegister, ProactiveChatAssistant
from ronin.cli import coroutine
from ronin.llm import LLMProvider, get_llm
from ronin.prompts.templates import SystemPromptTemplate


@click.command(name="chat", help="Chat with Ronin assistants.")
@coroutine
@click.option(
    "--assistant",
    "-a",
    "assistant_id",
    default="base-chat-assistant",
    help="ID of the assistant you want to chat with.",
)
@click.option(
    "--llm",
    "llm_provider",
    default="openai",
    help="ID of the LLM provider you want. Either `azure` or `openai`.",
)
@click.option(
    "--message",
    "-m",
    "first_message",
    help="Initial message to send to assistant.",
)
@click.option(
    "--system-message",
    default="",
    help="System message to use. This overrides the existing system message.",
)
@click.option(
    "--max-length",
    default=100,
    help="Maximum length of the response.",
)
@click.option(
    "--interactive",
    "-i",
    is_flag=True,
    default=False,
    help="Whether to keep the conversation open or not.",
)
@click.option(
    "--output-path",
    "-o",
    help="Path where to save .json file with output.",
)
async def chat(
    assistant_id: str,
    llm_provider: str,
    first_message: str,
    interactive: bool,
    system_message: str,
    max_length: int,
    output_path: str,
):
    llm: PromptNode = get_llm(
        LLMProvider(llm_provider),
        max_length=max_length,
    )

    logger.debug(f"Loading {assistant_id}.")
    Assistant = AssistantRegister.get(assistant_id)

    assistant_kwargs = {}
    if system_message:
        logger.debug("Building system message.")
        assistant_kwargs["chat_system_prompt"] = SystemPromptTemplate.from_str(
            system_message
        )

    try:
        logger.info(f"Starting {assistant_id} Assistant.")
        assistant = Assistant(chat_node=llm, **assistant_kwargs)
    except TypeError:
        logger.exception(
            f"Could not instantiate {assistant_id} Assistant. "
            f"Make sure you have passed all required arguments."
        )
        return

    if isinstance(assistant, ProactiveChatAssistant):
        response = assistant.proactively_send_message()
        print(f"Assistant:\n{response['content']}\n")

    logger.info("Initializing chat:")
    message = first_message
    if not message:
        message = click.prompt("User", type=str)
        print()
    else:
        print(f"User:\n{message}\n")

    while message != "exit":
        response = assistant.chat(message)
        print(f"Assistant:\n{response['content']}\n")

        if interactive:
            message = click.prompt("User", type=str)
            print()
        else:
            break

    if output_path:
        with open(output_path, "w") as f:
            json.dump(assistant.history, f, indent=4)
