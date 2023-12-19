from abc import ABCMeta
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin


class ColumnTransformer(TransformerMixin):
    """
    Similar to `sklearn.compose.ColumnTransformer`,
    but allows inputs with different columns set.
    This is done by ignoring certain columns, rather than dropping them.

    Be aware that this means there can be unexpected errors with the
    transformer, especially if it expects a certain set of columns
    (i.e.: StandardScaler)
    """

    def __init__(self, transformer: TransformerMixin, ignore_columns: List[str]):
        self._transformer = transformer
        self._ignore_columns = ignore_columns

    def _get_filtered_columns(self, columns: pd.Index):
        return columns.difference(self._ignore_columns)

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        filtered_columns = self._get_filtered_columns(X.columns)
        self._transformer.fit(X[filtered_columns])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Copy input to not overwrite it
        X = X.copy()

        filtered_columns = self._get_filtered_columns(X.columns)
        X[filtered_columns] = self._transformer.transform(X[filtered_columns])

        return X

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Copy input to not overwrite it
        X = X.copy()

        filtered_columns = self._get_filtered_columns(X.columns)
        X[filtered_columns] = self._transformer.inverse_transform(X[filtered_columns])

        return X


class Transformer(TransformerMixin, metaclass=ABCMeta):
    def fit_transform(
        self,
        X: Iterable,
        y: Optional[Iterable] = None,
        fit_params: Dict[str, Any] = dict(),
        transform_params: Dict[str, Any] = dict(),
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        return self.fit(X, y, **fit_params).transform(X, **transform_params)

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None):
        raise NotImplementedError

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
