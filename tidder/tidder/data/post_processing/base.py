from abc import ABC, abstractmethod

import attrs
from tidder.generic import InstanceableGeneric, T


@attrs.define
class PostProcessor(InstanceableGeneric[T], ABC):
    """Define base class for post processors."""

    @property
    @abstractmethod
    def requires(self) -> list[str]:
        raise NotImplementedError

    @property
    @abstractmethod
    def sets(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def apply(self, document: T) -> T:
        raise NotImplementedError
