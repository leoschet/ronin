import time
from typing import Iterable

import attrs
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

from tidder.dependencies import pyplot as plt


@attrs.define
class AutoKMeans(KMeans):
    r"""Auto KMeans for processing text inputs.

    Attributes
    ----------
    k_range : iterable of int
        Options for the number of clusters.

    Parameters
    ----------
    n_clusters : int
        Number of clusters, automatically derived during initialization.
    random_state : int, default 42
        Random state.
    """

    k_range: Iterable[int] = attrs.field(default=range(2, 15), repr=False, kw_only=True)
    random_state: int = attrs.field(default=42, kw_only=True)
    plot: bool = attrs.field(default=False, kw_only=True)

    def __attrs_post_init__(self):
        # Placeholder value for n_clusters, will be set in `fit`
        super().__init__(n_clusters=-1, random_state=self.random_state, n_init="auto")

    def fit(self, X, y=None, sample_weight=None):
        """Fit the AutoKMeans model to the data.

        Parameters (copied from sklearn)
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.
        y : Ignored
            Not used, present here for API consistency by convention.
        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight. `sample_weight` is not used during
            initialization if `init` is a callable or a user provided array.
        """
        k = self._select_k(X)
        self.n_clusters = k
        return super().fit(X, None, sample_weight)

    def _select_k(self, embedded_data):
        all_kmeans = {}
        for k in self.k_range:
            try:
                kmeans = KMeans(
                    n_clusters=k, random_state=self.random_state, n_init="auto"
                )
                start_time = time.time()
                kmeans.fit(embedded_data)
                fit_time = time.time() - start_time

                all_kmeans[k] = {
                    "model": kmeans,
                    "inertia": kmeans.inertia_,
                    "fit_time": fit_time,
                }
            except Exception as e:
                logger.error(f"Could not fit KMeans with k={k}: {e}")

        inertia_points = np.array(
            [(k, info["inertia"]) for k, info in all_kmeans.items()]
        )
        logger.trace(f"{inertia_points.tolist()=}")
        start = inertia_points[0]
        end = inertia_points[-1]

        # https://www.youtube.com/watch?v=tYUtWYGUqgw
        all_kmeans_score = np.cross(
            end - start, inertia_points - start
        ) / np.linalg.norm(end - start)
        best_kmeans_index = np.argmin(all_kmeans_score)
        best_k = self.k_range[best_kmeans_index]

        logger.trace(f"{all_kmeans_score=}")

        if self.plot:
            self._plot_elbow_analysis(all_kmeans, best_k)

        return best_k

    def _plot_elbow_analysis(self, kmeans_info, best_k):
        k_range, inertias, fit_times = zip(
            *[(k, info["inertia"], info["fit_time"]) for k, info in kmeans_info.items()]
        )

        fig, inertia_ax = plt.subplots()
        fit_time_ax = inertia_ax.twinx()

        fit_time_ax.plot(k_range, fit_times, "o--", c="#88b984")
        inertia_ax.plot(k_range, inertias, "o-", c="C0")

        fit_time_ax.set_ylabel("fit time (seconds)", color="#88b984")
        fit_time_ax.tick_params(axis="y", colors="#88b984")
        fit_time_ax.grid(False)

        inertia_ax.set_xlabel("k")
        inertia_ax.set_ylabel("inertia score", color="C0")
        inertia_ax.tick_params(axis="y", colors="C0")
        inertia_ax.grid(True)

        plt.title("Elbow analysis for KMeans clustering")
        plt.axvline(
            x=best_k,
            c="black",
            linestyle="dashed",
            label=f"elbow at k = {best_k}, score = {round(kmeans_info[best_k]['inertia'], 3)}",
        )
        plt.legend(loc="upper right", frameon=True, framealpha=1)
        plt.show()


@attrs.define
class AutoKMeansText(AutoKMeans):
    r"""Auto KMeans for processing text inputs.

    Attributes
    ----------
    raw_data : iterable of str
        Raw text data.
    embed_model : SentenceTransformer, default `all-MiniLM-L6-v2`
        Model to use for embedding the raw data.
    """

    raw_data: Iterable[str] = attrs.field(repr=False)
    embed_model: SentenceTransformer = attrs.field()

    def __init__(
        self,
        raw_data: Iterable[str],
        *,
        embed_model: SentenceTransformer = None,
        k_range: Iterable[int] = range(2, 15),
        random_state: int = 42,
    ):
        if embed_model is None:
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.raw_data = raw_data
        self.embed_model = embed_model
        embedded_data = self._embed_data(self.raw_data)
        self.__attrs_init__(
            embedded_data,
            raw_data,
            embed_model,
            k_range=k_range,
            random_state=random_state,
        )

    def predict(self, text_data, sample_weight=None):
        embedded_text = self._embed_data(text_data)
        return self.kmeans.predict(embedded_text, sample_weight)

    def _embed_data(self, text_data):
        return self.embed_model.encode(
            text_data,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
