"""Collection of transformers designed to index, slice, etc."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


def _is_callable_with_str(key):
    valid_key = True
    try:
        bool(key('test'))
    except TypeError:
        valid_key = False
    return valid_key


class ItemSelector(BaseEstimator, TransformerMixin):
    """Return only the columns specified by `key`.

    Parameters
    ----------
    key : (list of) int or str, slice, callable
      Specifies the columns to return. If callable, it must take a
      string as single input; it will be applied to all columns of the
      incoming DataFrame and only those will be returned that match
      positively.

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
        """Check if key satisfies one of the requirements, set
        attributes to indicate which one.

        """
        key = self.key

        self.is_callable_ = _is_callable_with_str(key)
        self.is_slice_ = isinstance(self.key, slice)

        if not isinstance(key, (list, tuple)):
            key = [key]

        self.only_ints_ = all([isinstance(k, int) for k in key])
        self.only_strings_ = all([isinstance(k, str) for k in key])

    # pylint: disable=unused-argument
    def fit(self, X, y=None):
        """Just checks if key is valid.

        Parameters
        ----------
        X : array-like or sparse matrix
          Data whose columns will be sliced.

        """
        self._check_key()

        if not (
                self.is_callable_ or
                self.is_slice_ or
                self.only_ints_ or
                self.only_strings_
        ):
            raise ValueError(
                "Key must be a callable that takes a string, a (list "
                "of) strings or ints, or a slice."
            )

        return self

    def transform(self, X):
        """Select columns indicated in ``key`` from ``X``.

        Parameters
        ----------
        X : array-like, scipy.sparse matrix or pandas DataFrame
          whose columns will be sliced.

        Returns
        -------
        X : array-like, scipy.sparse matrix or pandas DataFrame
          with columns selected according to ``key``.

        """
        key = self.key
        if isinstance(X, dict):
            Xt = X[key]
        elif isinstance(X, pd.DataFrame):
            if self.is_callable_:
                key = [col for col in X.columns if key(col)]
            Xt = X[key]
            if self.force_2d:
                Xt = Xt.values
        elif self.only_strings_ or self.is_callable_:
            raise ValueError(
                "List of strings as keys works only with pd.DataFrame or dict."
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
