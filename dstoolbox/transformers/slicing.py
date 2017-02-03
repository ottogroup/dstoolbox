"""Collection of transformers designed to index, slice, etc."""

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ItemSelector(BaseEstimator, TransformerMixin):
    """Return only the column specified by ``key``.

    Parameters
    ----------
    key : int, slice
      Specifies the columns to return.

    Returns
    -------
    X
    """
    def __init__(self, key):
        self.key = key

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
            return X.ix[:, self.key]
        elif self.only_strings_ and not isinstance(X, pd.DataFrame):
            raise ValueError(
                'List of strings as key works only with pd.DataFrame.'
            )
        else:
            return X[:, self.key]
