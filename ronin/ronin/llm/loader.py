from loguru import logger

from ronin.dependencies import (
    _HAYSTACK_INTEGRATION_AVAILABLE,
    _LANGCHAIN_INTEGRATION_AVAILABLE,
)
from ronin.haystack.llm import HaystackLLM
from ronin.langchain.llm import LangchainLLM

from .base import LLM, LLMProvider


def create_llm(
    llm_provider: LLMProvider,
    max_length: int | None = 700,
    access_kwargs: dict | None = None,
    model_kwargs: dict | None = None,
) -> LLM:
    if _LANGCHAIN_INTEGRATION_AVAILABLE and _HAYSTACK_INTEGRATION_AVAILABLE:
        logger.warning("Multiple LLM frameworks found. Is this expected?")

    ConcreteLLM: LLM | None = None
    if _LANGCHAIN_INTEGRATION_AVAILABLE:
        logger.info("Loading LLM using LangChain.")
        ConcreteLLM = LangchainLLM
    elif _HAYSTACK_INTEGRATION_AVAILABLE:
        logger.info("Loading LLM using Haystack.")
        ConcreteLLM = HaystackLLM

    if ConcreteLLM is None:
        raise ModuleNotFoundError(
            "At least one LLM framework must be installed to use Ronin."
        )

    return ConcreteLLM.create(
        llm_provider,
        max_length,
        access_kwargs,
        model_kwargs,
    )
