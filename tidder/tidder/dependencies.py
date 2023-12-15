"""Manage optional dependencies for tidder.

We do that by tapping into the `polars` library's lazy import mechanism.

Note from Polars' mechanism
---------------------------
We do NOT register this module with `sys.modules` so as not to cause
confusion in the global environment. This way we have a valid proxy
module for our own use, but it lives *exclusively* within ~polars~ tidder.
"""

from typing import TYPE_CHECKING

from polars.dependencies import _lazy_import

_WEBVTT_AVAILABLE = True

# Handle static type checking
# https://docs.python.org/3/library/typing.html#typing.TYPE_CHECKING
if TYPE_CHECKING:
    import webvtt
else:
    webvtt, _WEBVTT_AVAILABLE = _lazy_import("webvtt")

__all__ = [
    # Lazy-loaded modules
    "webvtt",
    # Flags
    "_WEBVTT_AVAILABLE",
]
