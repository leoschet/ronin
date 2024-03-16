from typing import Self

import attrs

from ronin.config import settings
from ronin.dependencies import langchain
from ronin.llm.base import LLM


@attrs.define
class LangchainLLM(LLM):
    llm: "langchain.BaseLanguageModel"

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

        streaming = model_kwargs.pop("streaming", False)
        temperature = model_kwargs.pop("temperature", 0)

        if "seed" not in model_kwargs:
            model_kwargs["seed"] = 0

        llm = langchain.AzureChatOpenAI(
            azure_endpoint=access_kwargs.get("azure_base_url")
            or settings.azure_openai_endpoint,
            openai_api_version=access_kwargs.get("api_version")
            or settings.azure_openai_api_version,
            deployment_name=access_kwargs.get("azure_deployment_name")
            or settings.azure_openai_chatgpt_deployment,
            openai_api_key=access_kwargs.get("api_key")
            or settings.azure_openai_api_key,
            streaming=streaming,
            temperature=temperature,
            model_kwargs=model_kwargs,
        )

        return cls(llm)
