from abc import ABC, abstractmethod
from typing import Any, Self

import attrs
from sentence_transformers import SentenceTransformer
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from torch import Tensor
from umap import UMAP

from tidder.transforms.dimensionality_reduction import DummyDimensionalityReduction

from .gaussian_mixture import AutoGaussianMixture
from .kmeans import AutoKMeans
from .spectral_clustering import AutoSpectralClustering


@attrs.define
class ClusteringBuilder(ABC):
    r"""Builder of clustering components.

    Centralizes the creation of clustering components:
    - Text embedding;
    - Dimensionality reduction;
    - Clustering.

    Attributes
    ----------
    random_state : int, default to 42
        Random state to be used in experiments.
    plot : bool, default to False
        Whether to plot the results of experiments.
    embedding_model : `SentenceTransformer` or None
        Embedding model to be used in experiments.
    clustering_model : `ClusterMixin` or None
        Clustering model to be used in experiments.
    dimensionality_reduction_model : `TransformerMixin` or None
        Dimensionality reduction model to be used in experiments.
    """

    random_state: int = attrs.field(default=42, kw_only=True)
    plot: bool = attrs.field(default=False, kw_only=True)
    fit: bool = attrs.field(default=True, kw_only=True)

    embedding_model: SentenceTransformer | None = attrs.field(default=None, init=False)
    clustering_model: ClusterMixin | None = attrs.field(default=None, init=False)
    dimensionality_reduction_model: ClusterMixin | None = attrs.field(
        default=None, init=False
    )

    @abstractmethod
    def build(self, *args, **kwargs) -> Any:
        """Build experiment pipeline."""
        raise NotImplementedError

    def produce_sentence_transformer(self, model_name: str) -> Self:
        """Use SentenceTransformer for embedding."""
        self.embedding_model = SentenceTransformer(model_name)
        return self

    def produce_pca(self, n_components: int) -> Self:
        """Use PCA for dimensionality reduction.

        Parameters
        ----------
        n_components: int
            Number of desired components (n_features)
        """
        self.dimensionality_reduction_model = PCA(n_components=n_components)
        return self

    def produce_umap_dimensionality_reduction(self, **kwargs) -> Self:
        """Use UMAP for dimensionality reduction."""
        kwargs["random_state"] = self.random_state
        self.dimensionality_reduction_model = UMAP(**kwargs)
        return self

    def produce_identity_dimensionality_reduction(self, **kwargs) -> Self:
        """Do not use dimensionality reduction."""
        self.dimensionality_reduction_model = DummyDimensionalityReduction()
        return self

    def produce_gaussian_mixture(
        self, embedded_data: list[Tensor] = None, **kwargs
    ) -> Self:
        """Use GaussianMixture for clustering."""
        kwargs["random_state"] = self.random_state
        kwargs["plot"] = self.plot
        self.clustering_model = AutoGaussianMixture(**kwargs)

        if self.fit and embedded_data is not None:
            self.clustering_model.fit(embedded_data)

        return self

    def produce_kmeans(self, embedded_data: list[Tensor] = None, **kwargs) -> Self:
        """Use KMeans for clustering."""
        kwargs["random_state"] = self.random_state
        kwargs["plot"] = self.plot
        self.clustering_model = AutoKMeans(**kwargs)

        if self.fit and embedded_data is not None:
            self.clustering_model.fit(embedded_data)

        return self

    def produce_spectral_clustering(
        self, embedded_data: list[Tensor] = None, **kwargs
    ) -> Self:
        """Use SpectralClustering for clustering."""
        kwargs["random_state"] = self.random_state
        kwargs["plot"] = self.plot
        self.clustering_model = AutoSpectralClustering(**kwargs)

        if self.fit and embedded_data is not None:
            self.clustering_model.fit(embedded_data)

        return self

    def produce_dbscan(self, embedded_data: list[Tensor] = None) -> Self:
        """Use DBSCAN for clustering."""
        self.clustering_model = DBSCAN()
        if self.fit and embedded_data is not None:
            self.clustering_model.fit(embedded_data)
        return self
