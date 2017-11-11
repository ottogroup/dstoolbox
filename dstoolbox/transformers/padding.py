"""Collection of transformers that perform padding in some form."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class Padder2d(BaseEstimator, TransformerMixin):
    """Takes a list of list of scalars (e.g. indices) and homogenizes
    it (i.e. makes a 2d array of shape num_samples x seq length) by
    truncating or padding the list.

    This is useful e.g. when you have indices representing words from
    sentences with different lengths.

    Parameters
    ----------
    max_len : int
      Maximum sequence length.

    pad_value : numerical (default=0)
      Value to pad with.

    dtype : dtype (default=np.float32)
      Data type of output.

    """
    def __init__(self, max_len, pad_value=0, dtype=np.float32):
        self.max_len = max_len
        self.pad_value = pad_value
        self.dtype = dtype

    # pylint: disable=unused-argument
    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, X):
        """Transform incoming data to a homogeneous 2d array.

        Parameters
        ----------
        X : list/array of list/array
          Heterogeneous data of len n.

        Returns
        -------
        Xt : np.ndarray
            Homogenous array of shape (n, max_len).

        """
        shape = len(X), self.max_len
        Xt = self.pad_value * np.ones(shape, dtype=self.dtype)

        for i, arr in enumerate(X):
            m = min(self.max_len, len(arr))
            if not m:
                continue
            arr = np.array(arr[:m])
            Xt[i, :m] = arr
        return Xt


class Padder3d(BaseEstimator, TransformerMixin):
    """Takes a list of list of vectors (e.g. word2vec) and homogenizes
    it (i.e. makes a 3d array of shape num_samples x seq length x
    vector size) by truncating or padding the list.

    This is useful e.g. when you have indices representing words with
    different lengths and sentences with different amount of words.

    Parameters
    ----------
    max_size : tuple (int, int)
      Maximum size in the two trailing dimensions.

    emb_size : int (default=300)
      Size of the embeddings.

    pad_value : numerical (default=0)
      Value to pad with.

    dtype : dtype (default=np.float32)
      Data type of output.

    """

    def __init__(self, max_size, pad_value=0, dtype=np.float32):
        self.max_size = max_size
        self.pad_value = pad_value
        self.dtype = dtype

    # pylint: disable=unused-argument
    def fit(self, X=None, y=None, **fit_params):
        return self

    def transform(self, X):
        """Transform incoming data to a homogeneous 3d array.

        Parameters
        ----------
        X : list/array of list/array of list/array
          Heterogeneous data of len n.

        Returns
        -------
        Xt : np.ndarray
            Homogenous array of shape (n, max_size[0], max_size[1])

        """
        n = len(X)
        Xt = self.pad_value * np.ones((n,) + self.max_size, dtype=self.dtype)

        for i, arr in enumerate(X):
            m = min(self.max_size[0], len(arr))
            if not m:
                continue
            arr = np.array(arr[:m])
            for j, vec in enumerate(arr):
                n = min(self.max_size[1], len(vec))
                Xt[i, j, :n] = vec[:n]
        return Xt
