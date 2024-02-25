from enum import Enum

from haystack.nodes import PromptModel, PromptNode
from loguru import logger

from ronin.config import settings


class LLMProvider(str, Enum):
    OPENAI = "openai"
    AZURE = "azure"


def get_llm(
    llm_provider: LLMProvider,
    max_length: int | None = 700,
    access_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
) -> PromptNode:
    access_kwargs = access_kwargs or {}
    model_kwargs = model_kwargs or {}

    llm: PromptModel
    if llm_provider == LLMProvider.OPENAI:
        logger.info("Connecting to OpenAI.")
        llm = _get_openai_llm(max_length, access_kwargs, model_kwargs)
    elif llm_provider == LLMProvider.AZURE:
        logger.info("Connecting to Azure.")
        llm = _get_azure_llm(max_length, access_kwargs, model_kwargs)
    return PromptNode(llm)


def _get_openai_llm(
    max_length: int | None,
    access_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
) -> PromptModel:
    return PromptModel(
        model_name_or_path=access_kwargs.get("model_name_or_path")
        or settings.openai_model_name,
        api_key=access_kwargs.get("api_key") or settings.openai_api_key,
        max_length=max_length,
        model_kwargs=model_kwargs,
    )


def _get_azure_llm(
    max_length: int | None,
    access_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
) -> PromptModel:
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

    return PromptModel(
        model_name_or_path=access_kwargs.get("model_name_or_path")
        or settings.azure_openai_chatgpt_deployment,
        api_key=access_kwargs.get("api_key") or settings.azure_openai_api_key,
        model_kwargs=model_kwargs,
        max_length=max_length,
    )
