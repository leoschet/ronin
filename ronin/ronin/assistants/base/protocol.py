from typing import TypeVar

from haystack.nodes import PromptNode
from typing_extensions import Protocol

from ronin.prompts.templates import (
    AssistantMessageTemplate,
    SystemPromptTemplate,
    UserMessageTemplate,
)
from ronin.typing_mixin import ChatMessage

GenericChatAssistant = TypeVar("GenericChatAssistant", bound="ChatAssistantProtocol")


class ChatAssistantProtocol(Protocol):
    """Base chat assistant class.

    Attributes
    ----------
    chat_node : PromptNode
        The PromptNode that the assistant uses to chat.
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

    chat_node: PromptNode
    chat_system_prompt: SystemPromptTemplate
    chat_user_prompt: UserMessageTemplate
    chat_assistant_prompt: AssistantMessageTemplate

    chat_system_kwargs: dict

    priming_message: str | None
    auto_prime: bool

    history: list[ChatMessage]

    def build_chat_system_message(self, **kwargs) -> ChatMessage:
        """Build system's chat message."""
        ...

    def build_user_chat_message(self, message: str, **kwargs) -> ChatMessage:
        """Build the user's chat message."""
        ...

    def build_assistant_chat_message(self, message: str, **kwargs) -> ChatMessage:
        """Build the assistant's chat message."""
        ...

    def prime(self) -> ChatMessage:
        """Prime the assistant to a certain state."""
        ...

    def chat(
        self,
        message: str,
        user_message_kwargs: dict | None = None,
        system_message_kwargs: dict | None = None,
    ) -> ChatMessage:
        """Chat with an assistant."""
        ...


# XXX: This could become a protocol in the future.
class ChatAssistantBuilderProtocol(Protocol[GenericChatAssistant]):
    def build(self) -> GenericChatAssistant:
        ...
