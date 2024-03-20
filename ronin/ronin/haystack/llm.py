from typing import Self

import attrs

from ronin.config import settings
from ronin.dependencies import haystack
from ronin.llm.base import LLM
from ronin.typing_mixin import ChatMessage


@attrs.define
class HaystackLLM(LLM):
    llm: "haystack.PromptNode"

    @classmethod
    def create_openai_llm(
        cls,
        max_length: int,
        access_kwargs: dict,
        model_kwargs: dict,
    ) -> Self:
        llm = haystack.PromptNode(
            haystack.PromptModel(
                model_name_or_path=access_kwargs.get("model_name_or_path")
                or settings.openai_model_name,
                api_key=access_kwargs.get("api_key") or settings.openai_api_key,
                max_length=max_length,
                model_kwargs=model_kwargs,
            )
        )

        return cls(llm)

    @classmethod
    def create_azure_llm(
        cls,
        max_length: int,
        access_kwargs: dict,
        model_kwargs: dict,
    ) -> Self:
        model_kwargs.update(
            {
                "api_version": access_kwargs.get("api_version")
                or settings.azure_openai_api_version,
                "azure_base_url": access_kwargs.get("azure_base_url")
                or settings.azure_openai_endpoint,
                "azure_deployment_name": access_kwargs.get("azure_deployment_name")
                or settings.azure_openai_chatgpt_deployment,
            }
        )

        llm = haystack.PromptNode(
            haystack.PromptModel(
                model_name_or_path=access_kwargs.get("model_name_or_path")
                or settings.azure_openai_chatgpt_deployment,
                api_key=access_kwargs.get("api_key") or settings.azure_openai_api_key,
                model_kwargs=model_kwargs,
                max_length=max_length,
            )
        )

        return cls(llm)

    def chat(self, messages: list[ChatMessage]) -> str:
        # XXX: Haystack does not support passing chat messages to the PromptNode.run
        # method, and thus we can't use the Pipeline class
        return self.llm(messages)[0]
