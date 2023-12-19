import attrs
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from tidder.dependencies import pyplot as plt

from .gaussian_mixture import AutoGaussianMixture
from .kmeans import AutoKMeans
from .spectral_clustering import AutoSpectralClustering


@attrs.define
class ClusteringExperimenter:
    r"""Builder runner of clustering experiments.

    Attributes
    ----------
    random_state : int, default to 42
        Random state to be used in experiments.
    experiment_pipeline : `Pipeline` or None
        Experiment pipeline.
    raw_data : list of str
        Raw data to be used in experiments.
    embed_model : `SentenceTransformer` or None
        Embedding model to be used in experiments.
    clustering_model : `ClusterMixin` or None
        Clustering model to be used in experiments.
    dimensionality_reduction : `TransformerMixin` or None
        Dimensionality reduction model to be used in experiments.
    """

    random_state: int = attrs.field(default=42, kw_only=True)
    plot: int = attrs.field(default=True, kw_only=True)

    experiment_pipeline: Pipeline = attrs.field(default=None, init=False)

    raw_data: list[str] = attrs.field(default=None, init=False)
    embed_model: SentenceTransformer = attrs.field(default=None, init=False)
    clustering_model: ClusterMixin = attrs.field(default=None, init=False)
    dimensionality_reduction: ClusterMixin = attrs.field(default=None, init=False)

    def build(self, return_self: bool = True) -> "ClusteringExperimenter" | Pipeline:
        """Build experiment pipeline."""
        if self.embed_model is None:
            raise ValueError("Embedding model must be provided.")
        if self.clustering_model is None:
            raise ValueError("Clustering model must be provided.")

        steps = []
        steps.append(("embed_model", self.embed_model))
        if self.dimensionality_reduction is not None:
            steps.append(("dimensionality_reduction", self.dimensionality_reduction))
        steps.append(("clustering_model", self.clustering_model))

        self.experiment_pipeline = Pipeline(steps)

        if return_self:
            return self
        else:
            return self.experiment_pipeline

    def experiment(self) -> tuple[Pipeline, np.ndarray]:
        """Run experiment."""
        if self.raw_data is None:
            raise ValueError("Raw data must be provided.")
        if self.experiment_pipeline is None:
            raise ValueError("Experiment pipeline must be built first.")

        labels = self.experiment_pipeline.fit_predict(self.raw_data)

        # if self.plot:
        #     self._plot_clustering(
        #         self.experiment_pipeline["embed_model"].embeddings, labels
        #     )

        return self.experiment_pipeline, labels

    def produce_sentence_transformer(self, model_name: str) -> "ClusteringExperimenter":
        """Use SentenceTransformer for embedding."""
        self.embedding_model = SentenceTransformer(model_name)
        return self

    def produce_pca(self, n_components: int) -> "ClusteringExperimenter":
        """Use PCA for dimensionality reduction.

        Parameters
        ----------
        n_components: int
            Number of desired components (n_features)
        """
        self.dimensionality_reduction = PCA(n_components=n_components)
        return self

    def produce_gaussian_mixture(self, k_range: range) -> "ClusteringExperimenter":
        """Use GaussianMixture for clustering."""
        self.clustering_model = AutoGaussianMixture(k_range=k_range, plot=self.plot)
        return self

    def produce_kmeans(self, k_range: range) -> "ClusteringExperimenter":
        """Use KMeans for clustering."""
        self.clustering_model = AutoKMeans(k_range=k_range, plot=self.plot)
        return self

    def produce_spectral_clustering(self) -> "ClusteringExperimenter":
        """Use SpectralClustering for clustering."""
        self.clustering_model = AutoSpectralClustering(plot=self.plot)
        return self

    def produce_dbscan(self) -> "ClusteringExperimenter":
        """Use DBSCAN for clustering."""
        self.clustering_model = DBSCAN()
        return self

    # def _plot_clustering(self, embedded_data, labels: np.ndarray):
    #     """Plot clusters and embeddings in 2D space for interpreting results."""
    #     # Project embeddings
    #     plottable_embedded_data = embedded_data
    #     if embedded_data.shape[1] > 2:
    #         plottable_embedded_data = self._apply_pca(embedded_data, n_components=2)

    #     # Getting unique labels
    #     u_labels = np.unique(labels)

    #     # Plotting the results:
    #     for i in u_labels:
    #         plt.scatter(
    #             plottable_embedded_data[labels == i, 0],
    #             plottable_embedded_data[labels == i, 1],
    #             label=i,
    #         )
    #     plt.legend()
    #     plt.show()
