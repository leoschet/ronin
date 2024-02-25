"""Pydantic settings for Ronin."""

from dotenv import load_dotenv
from pydantic import BaseSettings, root_validator


class Settings(BaseSettings):
    """Pydantic settings for Ronin."""

    # TODO: Field validation
    env: str = "local"
    
    # OpenAI
    openai_model_name: str | None = None
    openai_api_key: str | None = None

    # Azure OpenAI
    azure_openai_service: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_api_key: str | None = None
    azure_embeddings_deployment: str | None = None
    azure_openai_chatgpt_deployment: str | None = None

    # XXX: When updating pydantic to v2, check: https://stackoverflow.com/a/76301965/7454638
    @root_validator
    def _set_azure_openai_endpoint(cls, values):
        if values["azure_openai_service"] is not None:
            values[
                "azure_openai_endpoint"
            ] = f"https://{values['azure_openai_service']}.openai.azure.com/"
        return values


load_dotenv()
settings = Settings()
