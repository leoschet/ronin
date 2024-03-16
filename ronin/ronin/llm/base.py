from abc import ABC
from enum import Enum
from typing import Self

from loguru import logger

from ronin.typing_mixin import ChatMessage


class LLMProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"


class LLM(ABC):
    @classmethod
    def create(
        cls,
        llm_provider: LLMProvider,
        max_length: int | None = 700,
        access_kwargs: dict | None = None,
        model_kwargs: dict | None = None,
    ) -> Self:
        access_kwargs = access_kwargs or {}
        model_kwargs = model_kwargs or {}

        if llm_provider == LLMProvider.OPENAI:
            logger.info("Connecting to OpenAI.")
            return cls.create_openai_llm(max_length, access_kwargs, model_kwargs)
        elif llm_provider == LLMProvider.AZURE:
            logger.info("Connecting to Azure.")
            return cls.create_azure_llm(max_length, access_kwargs, model_kwargs)

    @classmethod
    def create_openai_llm(
        cls,
        max_length: int,
        access_kwargs: dict,
        model_kwargs: dict,
    ) -> Self:
        raise NotImplementedError

    @classmethod
    def create_azure_llm(
        cls,
        max_length: int,
        access_kwargs: dict,
        model_kwargs: dict,
    ) -> Self:
        raise NotImplementedError

    def chat(self, messages: list[ChatMessage]) -> str:
        raise NotImplementedError
