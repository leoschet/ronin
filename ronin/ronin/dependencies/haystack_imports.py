# flake8: noqa: F401

"""Haystack integration with Ronin.

Haystack is an optional dependency of Ronin. If not installed, this module
raises an error when imported.

This module is primarily supposed to be imported through the lazy import mechanism
in `ronin.dependencies`.

This simplifies the loading of optional dependencies, at the cost of concentrating
all haystack imports in this module. This is to workaround the main limitation of
`ronin.dependencies`, and that haystack does not make all of its functionality 
available through the root module.

Functionality that completely depends on haystack can be implemented as part of
this module.
"""


try:
    import haystack as _
    from haystack.nodes import PromptModel, PromptNode, PromptTemplate
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Haystack is not installed. To install it run `poetry install --with haystack`."
    ) from e
