from functools import partial
from typing import Self

import attrs
import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from tidder.dependencies import pyplot as plt
from tidder.transforms.tracker import Tracker

from .builder import ClusteringBuilder


@attrs.define(kw_only=True)
class ClusteringExperimenter(ClusteringBuilder):
    r"""Builder runner of clustering experiments.

    Attributes
    ----------
    plot : bool, default to True
        Whether to plot the clustering results.
    experiment_pipeline : `Pipeline` or None
        Experiment pipeline.

    Attributes (inherited from `ClusteringBuilder`)
    ----------
    random_state : int, default to 42
        Random state to be used in experiments.
    embedding_model : `SentenceTransformer` or None
        Embedding model to be used in experiments.
    clustering_model : `ClusterMixin` or None
        Clustering model to be used in experiments.
    dimensionality_reduction_model : `TransformerMixin` or None
        Dimensionality reduction model to be used in experiments.
    """

    plot: int = attrs.field(default=True, kw_only=True)
    experiment_pipeline: Pipeline = attrs.field(default=None, init=False)

    _embeddings_tracker_step_name: str = attrs.field(
        default="_embeddings_tracker", init=False, repr=False
    )
    _embeddings_tracker: Tracker = attrs.field(
        factory=partial(Tracker, info_extractor=lambda x: x), init=False, repr=False
    )

    def build(self, return_self: bool = True) -> Self | Pipeline:
        """Build experiment pipeline."""
        if self.embedding_model is None:
            raise ValueError("Embedding model must be provided.")
        if self.clustering_model is None:
            raise ValueError("Clustering model must be provided.")

        steps = []
        steps.append(("embedding", self.embedding_model))
        if self.dimensionality_reduction_model is not None:
            steps.append(
                ("dimensionality_reduction", self.dimensionality_reduction_model)
            )
        steps.append((self._embeddings_tracker_step_name, self._embeddings_tracker))
        steps.append(("clustering", self.clustering_model))

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

        if self.plot:
            self._plot_clustering(
                self.experiment_pipeline.named_steps[
                    self._embeddings_tracker_step_name
                ].info,
                labels,
            )

        return self.experiment_pipeline, labels

    def _plot_clustering(self, embedded_data, labels: np.ndarray):
        """Plot clusters and embeddings in 2D space for interpreting results."""
        # Project embeddings
        plottable_embedded_data = embedded_data
        if embedded_data.shape[1] > 2:
            pca = PCA(n_components=2)
            plottable_embedded_data = pca.fit_transform(embedded_data)

        # Getting unique labels
        u_labels = np.unique(labels)

        # Plotting the results:
        for i in u_labels:
            plt.scatter(
                plottable_embedded_data[labels == i, 0],
                plottable_embedded_data[labels == i, 1],
                label=i,
            )
        plt.legend()
        plt.show()
