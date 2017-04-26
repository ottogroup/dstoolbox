"""Collection of utility functions and classes.

This should never import from any other dstoolbox module.

"""

import numpy as np
from sklearn.base import BaseEstimator


def normalize_matrix(arr):
    """Normalize a matrix along its rows.

    The L2 norm is normalized to 1 for each row.

    Parameters
    ----------
    arr : np.ndarray 2D
      Matrix to be normalized.

    Returns
    -------
    normalized : np.ndarray 2D
      Normalized matrix with the same shape as input matrix.

    """
    normalized = arr / np.sqrt(np.sum(arr ** 2, axis=1, keepdims=True))
    return normalized


def cosine_similarity(arr0, arr1=None):
    """Compute the pairwise cosine similarities between 2 arrays.

    In contrast to `scipy.spacial.distance.cosine`, this works for more
    than 1 sample (and is also faster for many samples); furthermore,
    this calculates cosine similarity, not cosine distance.

    Parameters
    ----------
    arr0 : numpy.ndarray 2d
      First array.

    arr1 : numpy.ndarray 2d or None (default=None)
      Second array; if not given, calculate similarity to first array.

    Returns
    -------
    similarity : numpy.ndarray
      The rowwise cosine similarities.

    """
    if arr1 is not None:
        if arr0.shape[1] != arr1.shape[1]:
            raise ValueError("Incompatible shapes: {} vs {}.".format(
                arr0.shape[1], arr1.shape[1]))

    norm0 = normalize_matrix(arr0)
    if arr1 is None:
        norm1 = norm0
    else:
        norm1 = normalize_matrix(arr1)
    similarity = np.dot(norm0, norm1.T)
    return similarity


def fast_argsort(vec, n_neighbors):
    """Perform a fast argsort on `vec` that only looks at `n_neighbors
    + 1` smallest values.

    This implementation only sorts the smallest values, which makes it
    faster than performing a complete argsort and only then selecting
    the smallest values.

    Parameters
    ----------
    vec : np.ndarray 1D
      Vector to be sorted, from smallest to largest.

    n_neighbors : int
      Number of elements to sort. Only these elements will be sorted
      and returned.

    Returns
    -------
    args : np.ndarray 1D
      Indices of the `n_neighbors` smallest values in `vec`.

    """
    # no need for partition
    if n_neighbors >= len(vec):
        return np.argsort(vec)

    partition = np.argpartition(vec, n_neighbors)[:n_neighbors]
    args = partition[np.argsort(vec[partition])]
    return args


def _get_last_node(step, prefix=''):
    """Find the last node(s) of an estimator."""
    def get_name(*args):
        new_name = '__'.join(args).strip('_')
        return new_name

    name, est = step
    name = get_name(prefix, name)
    found = set()

    steps = getattr(est, 'steps', None)
    transformer_list = getattr(est, 'transformer_list', None)

    if steps:
        found.update(_get_last_node(steps[-1], prefix=name))
    elif transformer_list:
        for step_ in transformer_list:
            found.update(_get_last_node(step_, prefix=name))
    else:
        found.add((name, est))

    return found


def get_nodes_edges(name, model):
    """From an (sklearn) Pipeline, create nodes and edges that show
    which estimators are connected and how.

    This is useful if you want to create and analyze or visualize a
    graph form of the Pipeline.

    Parameters
    ----------
    ----------
    name : string
      Name of the model

    model : sklearn.pipeline.Pipeline
      The (sklearn) Pipeline or FeatureUnion.

    Returns
    -------
    nodes : dict
      A dictionary mapping the name of the nodes to the
      estimators. Names correspond to the names returned by a
      `.get_params()` call.

    edges : list of tuples
      Each tuple in the list consists of the name of the outgoing node
      and the name of the incoming node.

    """
    steps = getattr(model, 'steps', None)
    transformer_list = getattr(model, 'transformer_list', None)
    if not (steps or transformer_list):
        raise TypeError("Need a (sklearn) Pipeline or FeatureUnion as input.")

    nodes = {k: v for k, v in model.get_params().items()
             if isinstance(v, BaseEstimator)}
    nodes[name] = model

    def get_name(*args):
        if args[0] == name:
            args = args[1:]
        new_name = '__'.join(args)
        return new_name

    edges = []
    for k, v in nodes.items():
        transformer_list = getattr(v, 'transformer_list', None)
        steps = getattr(v, 'steps', None)

        if transformer_list:
            for step in transformer_list:
                edges.append((k, get_name(k, step[0])))

        if steps:
            edges.append((k, get_name(k, steps[0][0])))
            for (k0, v0), (k1, _) in zip(steps, steps[1:]):
                transformer_list_ = getattr(v0, 'transformer_list', None)
                steps_ = getattr(v0, 'steps', None)
                if transformer_list_ or steps_:
                    for step in _get_last_node((k0, v0)):
                        edges.append((get_name(k, step[0]), get_name(k, k1)))
                else:
                    edges.append((get_name(k, k0), get_name(k, k1)))

    return nodes, edges
