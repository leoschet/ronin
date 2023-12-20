import attrs
import numpy as np
import torch
from loguru import logger
from numpy.typing import ArrayLike
from scipy.sparse import csgraph
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import SpectralClustering

from tidder.dependencies import pyplot as plt


@attrs.define
class AutoSpectralClustering(SpectralClustering):
    r"""Auto SpectralClustering for processing text inputs.

    Attributes
    ----------
    embedded_data : Tensor
        Embeddings of raw text data.
        If `affinity="precomputed"`, or `affinity="precomputed_nearest_neighbors"`,
        `embedded_data` will be interpreted as a the affinity_matrix.
    internal_affinity : optional str
        Attribute used internally to speed up k selection by using a pre computed
        affinity matrix.
    internal_affinity_matrix : optional array-like
        Attribute used internally to speed up k selection by using a pre computed
        affinity matrix.

    Parameters
    ----------
    n_clusters : int
        Number of clusters, automatically derived during initialization.
    affinity : int, default "rbf"
        Copied from https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html
        How to construct the affinity matrix.
            - ‘nearest_neighbors’: construct the affinity matrix by computing
                a graph of nearest neighbors.
            - ‘rbf’: construct the affinity matrix using a radial basis function (RBF) kernel.
            - ‘precomputed’: interpret X as a precomputed affinity matrix, where
                larger values indicate greater similarity between instances.
            - ‘precomputed_nearest_neighbors’: interpret X as a sparse graph of
                precomputed distances, and construct a binary affinity matrix from
                the n_neighbors nearest neighbors of each instance.
            one of the kernels supported by pairwise_kernels.
        Only kernels that produce similarity scores (non-negative values that
        increase with similarity) should be used. This property is not checked
        by the clustering algorithm.
    random_state : int, default 42
        Random state.
    """

    affinity: str = attrs.field(default="rbf", kw_only=True)
    random_state: int = attrs.field(default=42, kw_only=True)
    plot: bool = attrs.field(default=False, kw_only=True)

    internal_affinity: str = attrs.field(
        default=None, init=False, repr=False, kw_only=True
    )
    internal_affinity_matrix: ArrayLike = attrs.field(
        default=None, init=False, repr=False, kw_only=True
    )

    @classmethod
    def build_affinity_matrix(cls, embedded_data: torch.Tensor, k: int = 7):
        """Build affinity matrix.

        Inpired by
        https://github.com/ciortanmadalina/high_noise_clustering/blob/master/spectral_clustering.ipynb

        This approach calculates the scale parameter for each point individually based on
        kth nearest neighbour.

        References:
        - https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
        - https://towardsdatascience.com/spectral-graph-clustering-and-optimal-number-of-clusters-estimation-32704189afbe

        Credit to ciortanmadalina@github
        """
        # Calculate euclidian distance matrix
        dists = squareform(pdist(embedded_data.numpy()))

        # For each row, sort the distances ascendingly and take the index of the
        # k-th position (nearest neighbour)
        knn_distances = np.sort(dists, axis=0)[k]
        knn_distances = knn_distances[np.newaxis].T

        # Calculate sigma_i * sigma_j
        local_scale = knn_distances.dot(knn_distances.T)

        affinity_matrix = dists * dists
        affinity_matrix = -affinity_matrix / local_scale

        # Divide square distance matrix by local scale
        affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0

        # Apply exponential
        affinity_matrix = np.exp(affinity_matrix)
        np.fill_diagonal(affinity_matrix, 0)

        return affinity_matrix

    def __attrs_post_init__(self):
        # Placeholder value for n_clusters, will be set in `fit`
        super().__init__(
            n_clusters=-1,
            affinity=self.internal_affinity,
            random_state=self.random_state,
        )

    def fit(self, X=None, y=None):
        """Fit the AutoSpectralClustering model to the data.

        Parameters
        ----------
        X : Ignored
            Not used, present here for API consistency by convention.
        y : Ignored
            Not used, present here for API consistency by convention.
        """
        self._set_internal_affinity()
        k = self._select_k()
        self.n_clusters = k
        return super().fit(self.embedded_data, None)

    def _set_internal_affinity(self):
        if (
            self.affinity == "precomputed"
            or self.affinity == "precomputed_nearest_neighbors"
        ):
            self.internal_affinity_matrix = self.embedded_data

        if self.internal_affinity_matrix is None:
            model = SpectralClustering(
                # Set a dummy value to the number of clusters.
                affinity=self.affinity,
                random_state=self.random_state,
            )

            model.fit(self.embedded_data)

            self.internal_affinity_matrix = model.affinity_matrix_

        if self.affinity == "nearest_neighbors":
            self.internal_affinity = "precomputed_nearest_neighbors"
        else:
            self.internal_affinity = "precomputed"

    def _select_k(self):
        """Select optimal number of clusters using eigen decomposition.

        Inspired by
        https://github.com/ciortanmadalina/high_noise_clustering/blob/master/spectral_clustering.ipynb
        """
        laplacian_affinity_matrix = csgraph.laplacian(
            self.internal_affinity_matrix, normed=True
        )
        n_components = self.internal_affinity_matrix.shape[0]  # noqa: F841

        # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh), that is, largest eigenvalues in
        # the euclidean norm of complex numbers.
        # eigenvalues, eigenvectors = eigsh(
        #     L, k=n_components, which="LM", sigma=1.0, maxiter=5000
        # )
        eigenvalues, eigenvectors = np.linalg.eig(laplacian_affinity_matrix)

        # Remove first eigenvalue, as it represents only one cluster
        eigenvalues = eigenvalues[1:]

        if self.plot:
            self._plot_eigen_values(eigenvalues)

        # Identify the optimal number of clusters as the index corresponding
        # to the larger gap between eigen values
        # Get top 5 results
        index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:5]

        # TODO: Review this
        # Add 1 to get the position instead of index
        # Add 1 to compensate removing the first eigenvalue
        n_clusters = index_largest_gap + 2

        logger.trace(f"{n_clusters=}")
        logger.trace(f"{np.diff(eigenvalues)=}")
        return n_clusters[0]

    def _plot_eigen_values(self, eigenvalues):
        plt.title("Largest eigen values of input matrix")
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
        plt.show()

        # plt.title("Elbow analysis for Gaussian Mixture clustering")
        # plt.axvline(
        #     x=best_k,
        #     c="black",
        #     linestyle="dashed",
        #     label=f"elbow at k = {best_k}, score = {round(model_info[best_k]['aic'], 3)}",
        # )
        # plt.legend(loc="upper right", frameon=True, framealpha=1)
        # plt.show()
