import json

import click
from loguru import logger

from ronin.assistants import AssistantRegister, ProactiveChatAssistant
from ronin.cli import coroutine
from ronin.llm.base import LLM, LLMProvider
from ronin.llm.loader import create_llm
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
    "--model-name",
    default=None,
    help="Name of model to use. This overrides the respective env variable.",
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
    default=None,
    type=int,
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
    model_name: str,
    first_message: str,
    interactive: bool,
    system_message: str,
    max_length: int | None,
    output_path: str,
):
    llm: LLM = create_llm(
        LLMProvider(llm_provider),
        max_length=max_length,
        access_kwargs={
            "model_name_or_path": model_name,
        },
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
        assistant = Assistant(llm=llm, **assistant_kwargs)
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
