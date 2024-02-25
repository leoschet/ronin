from abc import ABC, abstractmethod
from typing import Self

import attrs
from haystack.nodes import PromptTemplate

from ronin.typing_mixin import ChatMessage


@attrs.define
class BaseChatPromptTemplate(ABC):
    prompt: PromptTemplate
    role_field_name: str = "role"
    message_field_name: str = "content"

    @classmethod
    def from_str(cls, prompt: str, **kwargs) -> Self:
        return cls(prompt=PromptTemplate(prompt=prompt), **kwargs)

    @abstractmethod
    def fill(self, *args, **kwargs) -> ChatMessage:
        raise NotImplementedError

    def _fill(self, role: str, *args, **kwargs) -> ChatMessage:
        return {
            self.role_field_name: role,
            self.message_field_name: list(self.prompt.fill(*args, **kwargs))[0],
        }


@attrs.define
class BaseChatMessageTemplate(BaseChatPromptTemplate, ABC):
    message_prompt_parameter: str = "message"

    @classmethod
    def with_dummy_template(cls, **kwargs) -> Self:
        return cls(
            prompt=PromptTemplate(prompt="{message}"),
            message_prompt_parameter="message",
            **kwargs,
        )


@attrs.define
class SystemPromptTemplate(BaseChatPromptTemplate):
    def fill(self, *args, **kwargs):
        return self._fill(role="system", *args, **kwargs)


@attrs.define
class UserMessageTemplate(BaseChatMessageTemplate):
    def fill(self, *args, **kwargs):
        return self._fill(role="user", *args, **kwargs)


@attrs.define
class AssistantMessageTemplate(BaseChatMessageTemplate):
    def fill(self, *args, **kwargs):
        return self._fill(role="assistant", *args, **kwargs)
