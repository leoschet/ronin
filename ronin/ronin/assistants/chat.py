import attrs
from haystack.nodes import PromptNode
from loguru import logger

from ronin.prompts.defaults import DEFAULT_PROACTIVE_MESSAGE_TRIGGER
from ronin.prompts.templates import (
    AssistantMessageTemplate,
    SystemPromptTemplate,
    UserMessageTemplate,
)
from ronin.typing_mixin import ChatMessage


# XXX: It would be nice to have this be a Haystack Node.
# https://github.com/deepset-ai/haystack/issues/5442
@attrs.define(kw_only=True)
class ChatAssistant:
    chat_node: PromptNode
    chat_system_prompt: SystemPromptTemplate
    chat_user_prompt: UserMessageTemplate = attrs.field(
        factory=UserMessageTemplate.with_dummy_template, repr=False
    )
    chat_assistant_prompt: AssistantMessageTemplate = attrs.field(
        factory=AssistantMessageTemplate.with_dummy_template, repr=False
    )

    chat_system_kwargs: dict = attrs.field(factory=dict, repr=False)
    auto_prime: bool = False

    history: list[ChatMessage] = attrs.field(factory=list, repr=False, init=False)

    def __attrs_post_init__(self):
        if self.auto_prime:
            self.prime()

    def build_chat_system_message(self, **kwargs) -> ChatMessage:
        """Build system's chat message."""
        return self.chat_system_prompt.fill(**kwargs)

    def build_user_chat_message(self, message: str, **kwargs) -> ChatMessage:
        """Build the user's chat message."""
        params = kwargs
        params[self.chat_user_prompt.message_prompt_parameter] = message
        return self.chat_user_prompt.fill(**params)

    def build_assistant_chat_message(self, message: str, **kwargs) -> ChatMessage:
        """Build the assistant's chat message."""
        params = kwargs
        params[self.chat_assistant_prompt.message_prompt_parameter] = message
        return self.chat_assistant_prompt.fill(**params)

    def prime(self):
        """Prime the assistant to a certain state."""
        return

    def chat(
        self,
        message: str,
        user_message_kwargs: dict = dict(),
        system_message_kwargs: dict = dict(),
    ) -> ChatMessage:
        """Chat with the assistant."""

        if not system_message_kwargs:
            system_message_kwargs = self.chat_system_kwargs

        user_message = self.build_user_chat_message(
            message=message, **user_message_kwargs
        )

        self.history.append(user_message)
        messages = [
            self.build_chat_system_message(**system_message_kwargs)
        ] + self.history

        return self._chat(messages=messages)

    def _chat(self, messages: list[ChatMessage]) -> ChatMessage:
        # XXX: Haystack does not support passing chat messages to the PromptNode.run
        # method, and thus we can't use the Pipeline class
        answer = self.chat_node(messages)[0]

        assistant_message = self.build_assistant_chat_message(message=answer)
        self.history.append(assistant_message)
        return assistant_message


@attrs.define(kw_only=True)
class ProactiveChatAssistant(ChatAssistant):
    """A chat assistant that proactively sends messages to the user."""

    proactive_message_trigger: UserMessageTemplate = DEFAULT_PROACTIVE_MESSAGE_TRIGGER

    def proactively_send_message(self, **kwargs) -> ChatMessage:
        """Proactively send a message to the user."""
        logger.debug("Assistant is proactively sending a message.")
        messages = [
            self.build_chat_system_message(),
            self.proactive_message_trigger.fill(**kwargs),
        ]

        return self._chat(messages=messages)
