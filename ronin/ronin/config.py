"""Pydantic settings for Ronin."""

from dotenv import load_dotenv
from pydantic import BaseSettings, root_validator


class Settings(BaseSettings):
    """Pydantic settings for Ronin."""

    # OpenAI
    azure_openai_service: str
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str
    azure_openai_api_key: str
    azure_embeddings_deployment: str
    azure_openai_chatgpt_deployment: str

    # XXX: When updating pydantic to v2, check: https://stackoverflow.com/a/76301965/7454638
    @root_validator
    def _set_azure_openai_endpoint(cls, values):
        values[
            "azure_openai_endpoint"
        ] = f"https://{values['azure_openai_service']}.openai.azure.com/"
        return values


load_dotenv()
settings = Settings()
