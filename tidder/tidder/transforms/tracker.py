from typing import Any, Callable

import attrs
from sklearn.base import BaseEstimator, TransformerMixin


@attrs.define
class Tracker(BaseEstimator, TransformerMixin):
    """Track information in the middle of a pipeline."""

    info_extractor: Callable
    info: Any = attrs.field(default=None, init=False)

    def transform(self, X):
        self.info = self.info_extractor(X)
        return X

    def fit(self, X, y=None, **fit_params):
        return self
