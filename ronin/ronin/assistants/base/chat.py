import attrs
from loguru import logger

from ronin.llm.base import LLM
from ronin.prompts.defaults import DEFAULT_PROACTIVE_MESSAGE_TRIGGER
from ronin.prompts.templates import (
    AssistantMessageTemplate,
    SystemPromptTemplate,
    UserMessageTemplate,
)
from ronin.typing_mixin import ChatMessage

from .registry import AssistantRegister


# XXX: It would be nice to have this be a Haystack Node.
# https://github.com/deepset-ai/haystack/issues/5442
@AssistantRegister.register("base-chat-assistant")
@attrs.define(kw_only=True)
class ChatAssistant:
    """Base chat assistant class.

    Attributes
    ----------
    llm : LLM
        The Ronin LLM that the assistant uses to chat.
    chat_system_prompt : SystemPromptTemplate
        System's chat message template.
    chat_user_prompt : UserMessageTemplate
        User's chat message template.
        By default, this is a dummy template that passes the message through.
    chat_assistant_prompt : AssistantMessageTemplate
        Assistant's chat message template.
        By default, this is a dummy template that passes the message through.
    chat_system_kwargs : dict, optional
        Keyword arguments passed to the system's chat message template.
    priming_message : str, optional
        The message sent to the assistant to prime it to a certain state.
    auto_prime : bool, optional
        Whether to prime the assistant to a certain state when the assistant is
        instantiated.
    history : list[ChatMessage]
        The chat history.
    """

    llm: LLM
    chat_system_prompt: SystemPromptTemplate
    chat_user_prompt: UserMessageTemplate = attrs.field(
        factory=UserMessageTemplate.with_dummy_template, repr=False
    )
    chat_assistant_prompt: AssistantMessageTemplate = attrs.field(
        factory=AssistantMessageTemplate.with_dummy_template, repr=False
    )

    chat_system_kwargs: dict = attrs.field(factory=dict, repr=False)

    priming_message: str | None = None
    auto_prime: bool = False

    history: list[ChatMessage] = attrs.field(factory=list, repr=False)

    def __attrs_post_init__(self):
        self.try_auto_prime()

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

    def reset_history(self):
        """Reset the chat history."""
        self.history = []
        self.try_auto_prime()

    def try_auto_prime(self):
        """Try to auto-prime the assistant."""
        if self.auto_prime and self.priming_message:
            self.prime()

    def prime(self) -> ChatMessage:
        """Prime the assistant to a certain state."""
        if not self.priming_message:
            logger.warning(f"No priming message set for assistant {type(self)}.")
            return

        logger.debug("Priming assistant to helpful state.")
        messages = [
            # XXX: Should priming happen with or without the system message?
            self.build_chat_system_message(),
            UserMessageTemplate.from_str(prompt=self.priming_message).fill(),
        ]

        prime_response = self._chat(messages=messages)
        logger.debug(f"Prime response: {prime_response}")
        return prime_response

    def chat(
        self,
        message: str,
        user_message_kwargs: dict | None = None,
        system_message_kwargs: dict | None = None,
    ) -> ChatMessage:
        """Chat with an assistant."""
        if not user_message_kwargs:
            user_message_kwargs = {}

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
        answer = self.llm(messages)

        assistant_message = self.build_assistant_chat_message(message=answer)
        self.history.append(assistant_message)
        return assistant_message


@AssistantRegister.register("base-proactive-assistant")
@attrs.define(kw_only=True)
class ProactiveChatAssistant(ChatAssistant):
    """A chat assistant that proactively sends messages to the user.

    Attributes
    ----------
    proactive_message_trigger : UserMessageTemplate
        The message that triggers the assistant to proactively send a message.
    """

    proactive_message_trigger: UserMessageTemplate = DEFAULT_PROACTIVE_MESSAGE_TRIGGER

    def proactively_send_message(self, **kwargs) -> ChatMessage:
        """Proactively send a message to the user."""
        logger.debug("Assistant is proactively sending a message.")
        messages = [
            self.build_chat_system_message(),
            self.proactive_message_trigger.fill(**kwargs),
        ]

        return self._chat(messages=messages)
