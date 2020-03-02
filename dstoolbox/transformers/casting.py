"""Collection of transformers designed for casting data."""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class ToDataFrame(BaseEstimator, TransformerMixin):
    """This transformer casts incoming data to a pandas DataFrame.

    This transformer works with numpy ndarrays, pandas Series or
    DataFrames, dicts, and lists. It will convert the incoming data to
    a pandas DataFrame and take care of naming the columns.

    It is possible to pass explicit column names, but see the caveats
    below.

    Often, this transformer won't do more than a simple
    `FunctionTransformer(pd.DataFrame)`. There are some edge cases,
    however, where that will not work, in which case you should use
    `ToDataFrame`.

    Parameters
    ----------
    columns : None, str, or list/tuple of str (default=None)
        If not None, column names are taken from the `columns`
        argument. This does not work with dictionaries or DataFrames
        because those have fixed column names already. When incoming
        data is a list or pandas Series, only one column name can be
        passed.

    """
    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None, **fit_params):  # pylint: disable=unused-argument
        return self

    def _check_columns(self, X, columns):
        """Perform checks on whether incoming data and passed columns
        are compatible.

        """
        if isinstance(X, dict):
            raise ValueError(
                "ToDataFrame with explicit column names cannot "
                "transform a dictionary because the dictionary's "
                "keys already determine the column names.")

        if isinstance(X, pd.DataFrame):
            raise ValueError(
                "ToDataFrame with explicit column names cannot "
                "transform a DataFrame because the DataFrame's "
                "columns already determine the column names.")

        if isinstance(X, list) and len(columns) > 1:
            raise ValueError(
                "ToDataFrame with more than one column name cannot "
                "transform a list.")

        if isinstance(X, pd.Series) and len(columns) > 1:
            raise ValueError(
                "ToDataFrame with more than one column name cannot "
                "transform a Series object.")

        if not isinstance(X, np.ndarray):
            return

        if (X.ndim > 1) and (X.shape[1] != len(columns)):
            raise ValueError(
                "ToDataFrame was given data with {} columns but "
                "was initialized with {} column names.".format(
                    X.shape[1], len(columns)))

    def _series_to_df(self, X, column):
        """Transforms a pandas Series to a DataFrame."""
        Xt = pd.DataFrame(X)
        # pylint: disable=unsubscriptable-object
        return Xt.rename(columns={Xt.columns[0]: column})

    def _dict_to_df(self, X):
        """Transforms a dict to a DataFrame."""
        shapes = []
        for val in X.values():
            try:
                shapes.append(val.shape)
            except AttributeError:
                pass
        if any(len(shape) > 2 for shape in shapes):
            raise ValueError("dict values must be 1d arrays.")

        if any(shape for shape in shapes if len(shape) == 2 and shape[1] != 1):
            raise ValueError("dict values must be 1d arrays.")

        df = pd.DataFrame({k: np.squeeze(v) for k, v in X.items()})
        return df[sorted(df)]

    def transform(self, X):
        """Transform incoming data to a pandas DataFrame.

        Parameters
        ----------
        X : numpy.ndarray, pandas.DataFrame, pandas.Series, dict, list
            Data to be cast.

        Returns
        -------
        df : pandas.DataFrame
            Data cast to a DataFrame.

        """
        columns = self.columns
        if (columns is not None) and (not isinstance(columns, (list, tuple))):
            columns = [columns]

        if columns is not None:
            self._check_columns(X, columns)

        if isinstance(X, pd.Series) and (columns is not None):
            return self._series_to_df(X, columns[0])

        if isinstance(X, dict):
            return self._dict_to_df(X)

        df = pd.DataFrame(X, columns=columns)
        return df
