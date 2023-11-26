import functools

import attrs
from loguru import logger

from ronin.assistants.chat import ProactiveChatAssistant
from ronin.prompts.templates import (
    AssistantMessageTemplate,
    SystemPromptTemplate,
    UserMessageTemplate,
)


@attrs.define
class ConversationDesigner(ProactiveChatAssistant):
    """Conversation Designer assistant."""

    chat_system_prompt: SystemPromptTemplate = attrs.field(
        factory=functools.partial(
            SystemPromptTemplate.from_str,
            prompt=(
                "You are an experienced UX designer, specialized in designing "
                "conversational experiences with virtual assistants. You have a "
                "background in coaching and can design a virtual assistant that "
                "helps people achieve their goals.\n"
                "In this role, you are responsible for:\n"
                "- Defining what information the user shall provide for briefing "
                "the assistant;\n"
                "- Proposing the assistant's personality;\n"
                "- Designing the conversation flow;\n"
                "- Proposing integrations with other systems;\n"
                "You are conversing with a software engineer that will implement "
                "the assistant along with any other integration you propose.\n"
                "You have the creative freedom to challange the status quo and propose "
                "new ideas, and to think of the best way to build the perfect "
                "assistant."
            ),
        )
    )
    chat_user_prompt: UserMessageTemplate = attrs.field(
        factory=UserMessageTemplate.with_dummy_template, repr=False
    )
    chat_assistant_prompt: AssistantMessageTemplate = attrs.field(
        factory=AssistantMessageTemplate.with_dummy_template, repr=False
    )

    auto_prime: bool = True

    def prime(self):
        logger.debug("Priming assistant to helpful state.")
        prime_response = self.chat(
            (
                "In three sentences, what are the top 3 principles and concepts that "
                "make a great virtual assistant?"
            )
        )
        logger.debug(f"Prime response: {prime_response}")
