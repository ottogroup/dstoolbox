"""Additional clustering algorithms and estimators."""

import numpy as np
from scipy.cluster.hierarchy import fcluster
from scipy.cluster.hierarchy import linkage
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.utils import check_array


def hierarchical_clustering(
        X,
        max_dist=0.5,
        method='single',
        metric='euclidean',
        criterion='inconsistent',
):
    """Performs hierarchical/agglomerative clustering on the input
    array.

    Parameters
    ----------
    max_dist : float (default=0.5)
      Maximum allowed distance for two clusters to be merged.

    method : str (default='single')
      How distances are applied, see scipy.cluster.hierarchy.linkage
      documentation.

    metric : str (default='euclidean')
      Which distance metric to use. See
      scipy.cluster.hierarchy.linkage documentation.

    criterion : str (default='inconsistent')
      Criterion to use in forming flat clusters. See
      scipy.cluster.hierarchy.fcluster documentation.

    """
    # pylint: disable=len-as-condition
    if len(X) < 1:
        return np.array([])
    if len(X) == 1:
        return np.array([0])

    labels = fcluster(
        linkage(
            X,
            method=method,
            metric=metric,
        ),
        t=max_dist,
        criterion=criterion,
    )
    return labels


class HierarchicalClustering(BaseEstimator, ClusterMixin):
    """Use hierarchical clustering with cut-off value.

    Similar to sklearn.cluster.hierarchical.linkage_tree but does not
    require to indicate the number of clusters beforehand. Instead,
    the number of clusters is determined dynamically by use of the
    cut-off value `max_dist`. Therefore, the number of different
    clusters will depend on the input data, similarly to DBSCAN.

    Note: `HierarchicalClustering` does not support sparse
    matrices. If you want to use sparse matrices, pre-compute the
    pair-wise distance matrix (e.g. with
    `scipy.spatial.distance.pdist`), transform it using
    `scipy.spatial.distance.squareform`, and pass the result to this
    estimator with parameter `metric` set to None.

    Parameters
    ----------
    max_dist : float (default=0.5)
      Maximum allowed distance for two clusters to be merged.

    method : str (default='single')
      How distances are applied, see scipy.cluster.hierarchy.linkage
      documentation.

    metric : str (default='euclidean')
      Which distance metric to use. See
      scipy.cluster.hierarchy.linkage documentation.

    criterion : str (default='inconsistent')
      Criterion to use in forming flat clusters. See
      scipy.cluster.hierarchy.fcluster documentation.

    Attributes
    ----------
    labels_ : array [n_samples]
      cluster labels for each point

    """
    def __init__(
            self,
            max_dist=0.5,
            method='single',
            metric='euclidean',
            criterion='inconsistent',
    ):
        self.max_dist = max_dist
        self.method = method
        self.metric = metric
        self.criterion = criterion

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def fit(self, X, y=None, **fit_params):
        """Fit the hierarchical clustering on the data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          The samples a.k.a. observations.

        Returns
        -------
        self : instance of HierarchicalClustering

        """
        X = check_array(X, ensure_min_samples=2, estimator=self)
        self.labels_ = hierarchical_clustering(
            X,
            max_dist=self.max_dist,
            method=self.method,
            metric=self.metric,
            criterion=self.criterion,
        )
        return self
