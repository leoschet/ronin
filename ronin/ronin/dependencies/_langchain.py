# flake8: noqa: F401

"""Langchain integration with Ronin.

LangChain is an optional dependency of Ronin. If not installed, this module
raises an error when imported.

This module is primarily supposed to be imported through the lazy import mechanism
in `ronin.dependencies`.

This simplifies the loading of optional dependencies, at the cost of concentrating
all langchain imports in this module. This is to workaround the main limitation of
`ronin.dependencies`, and that langchain does not make all of its functionality 
available through the root module.

Functionality that completely depends on langchain can be implemented as part of
this module.
"""


try:
    import langchain as _
    import langchain_community as _  # type: ignore # noqa: F811
    import langchain_openai as _  # type: ignore # noqa: F811
    from langchain.schema import Document
    from langchain.schema.language_model import BaseLanguageModel
    from langchain.chat_models import AzureChatOpenAI
    from langchain_openai import OpenAI
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "LangChain is not installed. To install it run `poetry install --with langchain`."
    ) from e
