"""Manage optional dependencies for Ronin.

Main functionality of this module is to provide lazy imports of optional dependencies.
Effectively, this means errors are only raised when we try to use an optional dependency
that is not installed. This allows for classes to implement optional features without
being aware whether the dependency is installed or not.

The main limitation of this module is that it targets a specific module of the dependency.
For example, langchain has many sub-modules that need to be imported separately (e.g. 
`langchain.schema` and `langchain.prompts`). To have them all loaded here, we would need
several separate entries in `__all__`.

This whole module is based on the mechanism implemented for the polars library.
See https://github.com/pola-rs/polars/blob/main/py-polars/polars/dependencies.py

Note from Polars' mechanism
---------------------------
We do NOT register this module with `sys.modules` so as not to cause
confusion in the global environment. This way we have a valid proxy
module for our own use, but it lives *exclusively* within ~polars~ Ronin.
"""

from __future__ import annotations

import re
import sys
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, Any, ClassVar


class _LazyModule(ModuleType):
    """
    Module that can act both as a lazy-loader and as a proxy.

    Notes
    -----
    We do NOT register this module with `sys.modules` so as not to cause
    confusion in the global environment. This way we have a valid proxy
    module for our own use, but it lives *exclusively* within polars.

    """

    __lazy__ = True

    _mod_pfx: ClassVar[dict[str, str]] = {
        "numpy": "np.",
        "pandas": "pd.",
        "pyarrow": "pa.",
    }

    def __init__(
        self,
        module_name: str,
        *,
        module_available: bool,
    ) -> None:
        """
        Initialise lazy-loading proxy module.

        Parameters
        ----------
        module_name : str
            the name of the module to lazy-load (if available).

        module_available : bool
            indicate if the referenced module is actually available (we will proxy it
            in both cases, but raise a helpful error when invoked if it doesn't exist).

        """
        self._module_available = module_available
        self._module_name = module_name
        self._globals = globals()
        super().__init__(module_name)

    def _import(self) -> ModuleType:
        # import the referenced module, replacing the proxy in this module's globals
        module = import_module(self.__name__)
        self._globals[self._module_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, attr: Any) -> Any:
        # have "hasattr('__wrapped__')" return False without triggering import
        # (it's for decorators, not modules, but keeps "make doctest" happy)
        if attr == "__wrapped__":
            raise AttributeError(
                f"{self._module_name!r} object has no attribute {attr!r}"
            )

        # accessing the proxy module's attributes triggers import of the real thing
        if self._module_available:
            # import the module and return the requested attribute
            module = self._import()
            return getattr(module, attr)

        # user has not installed the proxied/lazy module
        elif attr == "__name__":
            return self._module_name
        elif re.match(r"^__\w+__$", attr) and attr != "__version__":
            # allow some minimal introspection on private module
            # attrs to avoid unnecessary error-handling elsewhere
            return None
        else:
            # all other attribute access raises a helpful exception
            pfx = self._mod_pfx.get(self._module_name, "")
            raise ModuleNotFoundError(
                f"{pfx}{attr} requires {self._module_name!r} module to be installed"
            ) from None


def _lazy_import(module_name: str) -> tuple[ModuleType, bool]:
    """
    Lazy import the given module; avoids up-front import costs.

    Parameters
    ----------
    module_name : str
        name of the module to import, eg: "pyarrow".

    Notes
    -----
    If the requested module is not available (eg: has not been installed), a proxy
    module is created in its place, which raises an exception on any attribute
    access. This allows for import and use as normal, without requiring explicit
    guard conditions - if the module is never used, no exception occurs; if it
    is, then a helpful exception is raised.

    Returns
    -------
    tuple of (Module, bool)
        A lazy-loading module and a boolean indicating if the requested/underlying
        module exists (if not, the returned module is a proxy).

    """
    # check if module is LOADED
    if module_name in sys.modules:
        return sys.modules[module_name], True

    # check if module is AVAILABLE
    try:
        module_spec = find_spec(module_name)
        module_available = not (module_spec is None or module_spec.loader is None)
    except ModuleNotFoundError:
        module_available = False

    # create lazy/proxy module that imports the real one on first use
    # (or raises an explanatory ModuleNotFoundError if not available)
    return (
        _LazyModule(
            module_name=module_name,
            module_available=module_available,
        ),
        module_available,
    )


_PANDAS_INTEGRATION_AVAILABLE = True
_LANGCHAIN_INTEGRATION_AVAILABLE = True
_HAYSTACK_INTEGRATION_AVAILABLE = True

# Handle static type checking
# https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING
if TYPE_CHECKING:
    from ronin.dependencies import _haystack as haystack
    from ronin.dependencies import _langchain as langchain
    # import polars
else:
    # polars, _PANDAS_INTEGRATION_AVAILABLE = _lazy_import("polars")
    langchain, _ = _lazy_import("ronin.dependencies.langchain_imports")
    _, _LANGCHAIN_INTEGRATION_AVAILABLE = _lazy_import("langchain")

    # haystack, _ = _lazy_import("ronin.dependencies.haystack_imports")
    haystack, _HAYSTACK_INTEGRATION_AVAILABLE = _lazy_import("haystack")

__all__ = [
    # Lazy-loaded modules
    "polars",
    "langchain",
    "haystack",
    # Flags
    "_PANDAS_INTEGRATION_AVAILABLE",
    "_LANGCHAIN_INTEGRATION_AVAILABLE",
    "_HAYSTACK_INTEGRATION_AVAILABLE",
]
