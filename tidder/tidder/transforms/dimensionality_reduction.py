from sklearn.base import BaseEstimator, TransformerMixin


class DummyDimensionalityReduction(BaseEstimator, TransformerMixin):
    def fit(self, X):
        return self

    def transform(self, X):
        return X
