"""Collection of transformers designed for preprocesing."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from joblib import Parallel
from joblib import delayed


def _identity(X):
    """The identity function.
    """
    return X


def _partition(X, n_parts=1):
    """The partition function for the parallel function transformer
    """
    try:
        n = X.shape[0]
    except AttributeError:
        n = len(X)

    length = np.ceil(n / n_parts).astype(int)

    for i in range(n_parts):
        yield X[i * length: (i + 1) * length]


class XLabelEncoder(BaseEstimator, TransformerMixin):
    """LabelEncoder that may be used for feature data.

    LabelEncoder-variant that, instead of raising an error when
    encountering a new label, returns a default encoded value for that
    label. This implementation allows to keep on going even if an
    unseen label appears.

    Attributes
    ----------
    classes_dict_ : dict
        Dictionary containing the class mappings. The class name of
        unknown labels is '<UNKNOWN>'.

    """

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """Fit label encoder

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Target values.

        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returned instance of self.

        """
        # pylint: disable=attribute-defined-outside-init
        self.classes_dict_ = {k: v + 1 for v, k in enumerate(np.unique(X))}
        self.classes_dict_['<UNKNOWN>'] = 0
        return self

    def transform(self, X):
        """Transform labels to normalized encodings.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        Xt : array-like of shape (n_samples,)
            Target values.

        """
        check_is_fitted(self, 'classes_dict_')
        Xt = np.array([self.classes_dict_.get(x, 0) for x in X.flatten()])
        return Xt.reshape(-1, 1)

    # pylint: disable=unused-argument
    def fit_transform(self, X, y=None, **fit_params):
        """Fit labels and transform to normalized encodings.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Target values.

        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        Xt : array-like of shape (n_samples,)
            Target values.

        """
        return self.fit(X, y, **fit_params).transform(X)


class ParallelFunctionTransformer(BaseEstimator, TransformerMixin):
    """Constructs a transformer from an arbitrary callable with option
    to parallelize the transformation step.

    A FunctionTransformer forwards its X (and optionally y) arguments to a
    user-defined function or function object and returns the result of this
    function. This is useful for stateless transformations such as taking the
    log of frequencies, doing custom scaling, etc.

    A FunctionTransformer will not do any checks on its function's output.

    Note: If a lambda is used as the function, then the resulting
    transformer will not be pickleable.

    Parameters
    ----------
    func : callable, optional default=None
        The callable to use for the transformation. This will be passed
        the same arguments as transform, with args and kwargs forwarded.
        If func is None, then func will be the identity function.

    validate : bool, optional default=True
        Indicate that the input X array should be checked before calling
        func. If validate is false, there will be no input validation.
        If it is true, then X will be converted to a 2-dimensional NumPy
        array or sparse matrix. If this conversion is not possible or X
        contains NaN or infinity, an exception is raised.

    accept_sparse : boolean, optional
        Indicate that func accepts a sparse matrix as input. If validate is
        False, this has no effect. Otherwise, if accept_sparse is false,
        sparse matrix inputs will cause an exception to be raised.

    n_jobs : int (default=1)
        Number of jobs to run in parallel. Data will be partitioned
        into `n_jobs` equally sized chunks.

    Note
    ----
    The parameter `pass_y` is not supported.

    """
    def __init__(
            self,
            func=None,
            validate=True,
            accept_sparse=False,
            n_jobs=1,
    ):
        self.func = func
        self.validate = validate
        self.accept_sparse = accept_sparse
        self.n_jobs = n_jobs

    def fit(self, X, y=None):  # pylint: disable=unused-argument
        """Fit parallel function transformer

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input values.

        Returns
        -------
        self : object
            Returned instance of self.

        """
        if self.validate:
            check_array(X, self.accept_sparse)
        return self

    def transform(self, X, y=None):  # pylint: disable=unused-argument
        """Transform values with the given function.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
                Input values.

        Returns
        -------
        Xt : array-like of shape (n_samples,)
                Transformed values.

        """
        if self.validate:
            X = check_array(X, self.accept_sparse)
        func = self.func if self.func is not None else _identity

        Xt = np.vstack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(func)(x) for x in _partition(X, self.n_jobs))
        )
        return Xt
