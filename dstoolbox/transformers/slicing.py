"""Collection of transformers designed to index, slice, etc."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):
    """Return only the columns specified by `key`.

    Parameters
    ----------
    key : (list of) int, str, slice
      Specifies the columns to return.

    force_2d : bool (default=False)
      If true, forces output to be 2d; if output is already 2d,
      nothing changes, if it has more dimensions, a `ValueError` is
      raised.

    """
    def __init__(self, key, force_2d=False):
        self.key = key
        self.force_2d = force_2d

    # pylint: disable=attribute-defined-outside-init
    def _check_key(self):

        keys_as_list = self.key
        if not isinstance(keys_as_list, list):
            keys_as_list = [keys_as_list]

        self.only_ints_ = all([isinstance(s, int) for s in keys_as_list])
        self.only_strings_ = all([isinstance(s, str) for s in keys_as_list])
        self.is_slice_ = isinstance(self.key, slice)

    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        """Just checks if key is valid.

        Parameters
        ----------
        X : array-like or sparse matrix
          Data whose columns will be sliced.

        """
        self._check_key()

        if not (self.only_ints_ or self.only_strings_ or self.is_slice_):
            raise ValueError(
                'List must contain only strings, only integers or slices.'
            )

        return self

    def transform(self, X):
        """Select columns indicated in ``key`` from ``X``.

        Parameters
        ----------
        X : array-like, scipy.sparse matrix or pandas dataframe
          Data whose columns will be sliced.

        Returns
        -------
        X : array-like, scipy.sparse matrix or pandas dataframe
          Data with columns selected according to ``key``.

        """
        if isinstance(X, pd.DataFrame):
            Xt = X[self.key]
            if self.force_2d:
                Xt = Xt.values
        elif self.only_strings_ and not isinstance(X, pd.DataFrame):
            raise ValueError(
                'List of strings as keys works only with pd.DataFrame.'
            )
        else:
            Xt = X[:, self.key]

        ndim = Xt.ndim
        if self.force_2d and (ndim == 1):
            Xt = np.expand_dims(Xt, axis=1)
        if self.force_2d and (ndim > 2):
            raise ValueError("ItemSelector cannot force 2d on {}d data."
                             "".format(ndim))
        return Xt
