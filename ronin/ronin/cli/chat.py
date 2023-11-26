import json

import click
from haystack.nodes import PromptModel, PromptNode
from loguru import logger

from ronin.assistants.base import ChatAssistant, ProactiveChatAssistant
from ronin.assistants.coach import ConversationDesigner
from ronin.cli import coroutine
from ronin.config import settings
from ronin.prompts.templates import SystemPromptTemplate


@click.command(name="chat", help="Chat with the assistant.")
@coroutine
@click.option(
    "--message",
    "-m",
    "first_message",
    help="Initial message to send to assistant.",
)
@click.option(
    "--system-message",
    default="You are a helpful assistant.",
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
    first_message: str,
    interactive: bool,
    system_message: str,
    max_length: int,
    output_path: str,
):
    logger.info("Connecting to OpenAI.")
    prompt_azure_openai = PromptModel(
        model_name_or_path=settings.azure_openai_chatgpt_deployment,
        api_key=settings.azure_openai_api_key,
        model_kwargs={
            "api_version": settings.azure_openai_api_version,
            "azure_base_url": settings.azure_openai_endpoint,
            "azure_deployment_name": settings.azure_openai_chatgpt_deployment,
        },
        max_length=max_length,
    )
    openai_node = PromptNode(prompt_azure_openai)

    logger.info("Starting Assistant.")
    assistant = ConversationDesigner(
        chat_node=openai_node,
        # chat_system_prompt=SystemPromptTemplate.from_str(system_message)
    )

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
