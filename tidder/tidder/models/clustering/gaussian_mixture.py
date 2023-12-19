import time
from typing import Iterable

import attrs
import numpy as np
from loguru import logger
from sklearn.mixture import GaussianMixture

from tidder.dependencies import pyplot as plt


@attrs.define
class AutoGaussianMixture(GaussianMixture):
    r"""Auto GaussianMixture for processing text inputs.

    Attributes
    ----------
    k_range : iterable of int
        Options for the number of clusters.

    Parameters
    ----------
    n_components : int
        Number of clusters, automatically derived during initialization.
    random_state : int, default 42
        Random state.
    """

    k_range: Iterable[int] = attrs.field(default=range(2, 15), repr=False, kw_only=True)
    random_state: int = attrs.field(default=42, kw_only=True)
    plot: bool = attrs.field(default=False, kw_only=True)

    labels_: np.ndarray = attrs.field(repr=False, init=False)
    fitted_: bool = attrs.field(repr=False, init=False)

    def __attrs_post_init__(self):
        # Placeholder value for n_clusters, will be set in `fit`
        super().__init__(n_components=-1, random_state=self.random_state)

    def fit_predict(self, X, y=None):
        """Fit and predict labels for given data.

        Override method to set labels_ parameter.
        Under the hood, GaussianMixture.fit calls GaussianMixture.fit_predict.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : Ignored
            Not used, present here for API consistency by convention.
        """
        k = self._select_k(X)
        self.n_components = k
        self.labels_ = super().fit_predict(X, None)
        self.fitted_ = True
        return self.labels_

    def predict(self, X):
        r"""Override predict function to skip `check_is_fitted`"""
        X = self._validate_data(X, reset=False)
        return self._estimate_weighted_log_prob(X).argmax(axis=1)

    def _select_k(self, embedded_data):
        all_gmm = {}
        for k in self.k_range:
            try:
                gmm = GaussianMixture(n_components=k, random_state=self.random_state)
                start_time = time.time()
                gmm.fit(embedded_data)
                fit_time = time.time() - start_time

                all_gmm[k] = {
                    "model": gmm,
                    # AIC vs BIC: https://stats.stackexchange.com/a/767/207255
                    # > "AIC tries to select the model that most adequately describes an
                    #   unknown, high dimensional reality. [...] BIC tries to find the
                    #   TRUE model among the set of candidates."
                    "aic": gmm.aic(embedded_data),
                    "fit_time": fit_time,
                }
            except Exception as e:
                logger.error(f"Could not fit GaussianMixture with k={k}: {e}")

        aic_points = np.array([(k, info["aic"]) for k, info in all_gmm.items()])
        logger.trace(f"{aic_points.tolist()=}")
        start = aic_points[0]
        end = aic_points[-1]

        # https://www.youtube.com/watch?v=tYUtWYGUqgw
        all_gmm_score = np.cross(end - start, aic_points - start) / np.linalg.norm(
            end - start
        )
        best_gmm_index = np.argmin(all_gmm_score)
        best_k = self.k_range[best_gmm_index]

        logger.trace(f"{all_gmm_score=}")

        if self.plot:
            self._plot_elbow_analysis(all_gmm, best_k)

        return best_k

    def _plot_elbow_analysis(self, gmm_info, best_k):
        k_range, aics, fit_times = zip(
            *[(k, info["aic"], info["fit_time"]) for k, info in gmm_info.items()]
        )

        fig, aic_ax = plt.subplots()
        fit_time_ax = aic_ax.twinx()

        fit_time_ax.plot(k_range, fit_times, "o--", c="#88b984")
        aic_ax.plot(k_range, aics, "o-", c="C0")

        fit_time_ax.set_ylabel("fit time (seconds)", color="#88b984")
        fit_time_ax.tick_params(axis="y", colors="#88b984")
        fit_time_ax.grid(False)

        aic_ax.set_xlabel("k")
        aic_ax.set_ylabel("aic score", color="C0")
        aic_ax.tick_params(axis="y", colors="C0")
        aic_ax.grid(True)

        plt.title("Elbow analysis for Gaussian Mixture clustering")
        plt.axvline(
            x=best_k,
            c="black",
            linestyle="dashed",
            label=f"elbow at k = {best_k}, score = {round(gmm_info[best_k]['aic'], 3)}",
        )
        plt.legend(loc="upper right", frameon=True, framealpha=1)
        plt.show()
