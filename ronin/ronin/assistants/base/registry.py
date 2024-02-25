from enum import Enum
from typing import ClassVar, MutableMapping

import attrs
from loguru import logger

from .protocol import ChatAssistantBuilderProtocol, ChatAssistantProtocol


class AssistantRegisteredDataType(Enum):
    """Type of data that can be registered with the AssistantRegister."""

    ASSISTANT = "assistant"
    BUILDER = "builder"
    # PROMPT = "prompt"


@attrs.define
class AssistantRegisteredDataCollection:
    """Data collection for the AssistantRegister.

    We do not attempt to validate whether all registered data is complete.
    Information gets filled on the go, and the Register is responsible for
    throwing errors if the data is incomplete.
    """

    name: str
    class_: type[ChatAssistantProtocol] | None = None
    builder: type[ChatAssistantBuilderProtocol] | None = None
    # In the future we can extend it to include prompt calsses too.
    # prompt: type["Prompt"]


class AssistantRegister:
    """Implements the Register pattern for assistants.

    It's important that the assistant class extends from `ChatAssistant`.
    ```
    @AssistantRegister.register("my-chat-assistant")
    class MyChatAssistant(ChatAssistant)
    ```

    You can retrieve the registered assistant class by using the `get` method:
    ```
    MyChatAssistant = AssistantRegister.get("my-chat-assistant")
    ```
    """

    _registry: ClassVar[MutableMapping[str, AssistantRegisteredDataCollection]] = dict()

    @classmethod
    def register(
        cls,
        name: str,
        register_as: AssistantRegisteredDataType = AssistantRegisteredDataType.ASSISTANT,
    ):
        """Decorator for registering assistants."""

        def _register(assistant_class: type[ChatAssistantProtocol]):
            if name not in cls._registry:
                cls._registry[name] = AssistantRegisteredDataCollection(name=name)

            if register_as == AssistantRegisteredDataType.ASSISTANT:
                cls._registry[name].class_ = assistant_class
            elif register_as == AssistantRegisteredDataType.BUILDER:
                cls._registry[name].builder = assistant_class

            return assistant_class

        return _register

    @classmethod
    def get(
        cls,
        name: str,
        info: AssistantRegisteredDataType = AssistantRegisteredDataType.ASSISTANT,
    ) -> type[ChatAssistantProtocol] | type[ChatAssistantBuilderProtocol]:
        """Retrieve an Assistant's class from its name."""
        if name not in cls._registry:
            error_msg = (
                f"Could not find assistant registered with name '{name}'. "
                f"Available are: {list(cls._registry.keys())}"
            )
            logger.error(error_msg)
            raise KeyError(error_msg)

        retrieved_data = None
        if info == AssistantRegisteredDataType.ASSISTANT:
            retrieved_data = cls._registry[name].class_
        elif info == AssistantRegisteredDataType.BUILDER:
            retrieved_data = cls._registry[name].builder

        if retrieved_data is None:
            error_msg = (
                f"Could not find a registered `{info.value}` registered "
                f"for assistant '{name}'. "
            )
            logger.error(error_msg)
            raise KeyError(error_msg)

        return retrieved_data
