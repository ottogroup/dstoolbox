"""Extend sklearn's Pipeline and FeatureUnion."""

import itertools
from functools import wraps
import time
import types
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.pipeline import _transform_one
from sklearn.pipeline import _fit_transform_one
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Parallel
from sklearn.pipeline import Pipeline
from sklearn.pipeline import delayed
from sklearn.utils import tosequence
from sklearn.utils.metaestimators import if_delegate_has_method


class PipelineY(Pipeline):
    """Extension of sklearn Pipeline with tranformer for y values.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    memory : Instance of sklearn.external.joblib.Memory or string, optional \
            (default=None)
        Used to cache the fitted transformers of the pipeline. By
        default, no caching is performed. If a string is given, it is
        the path to the caching directory. Enabling caching triggers a
        clone of the transformers before fitting. Therefore, the
        transformer instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    y_transformer : transformer object
        Transformer object that transforms the y values (e.g.,
        discretiziation). May optionally support inverse_transform
        method.

    predict_use_inverse : bool (default=False)
        Determine if ``predict`` should use the inverse transform of
        y_transformer on the output.

    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    """
    def __init__(
            self,
            steps,
            y_transformer,
            predict_use_inverse=False,
            **kwargs
    ):
        warnings.warn(DeprecationWarning(
            "PipelineY is deprecated and will be removed in a future release. "
            "Please use sklearn.compose.TransformedTargetRegressor instead."
        ))
        self.y_transformer = y_transformer
        self.predict_use_inverse = predict_use_inverse
        super().__init__(steps=steps, **kwargs)

        if not hasattr(y_transformer, "transform"):
            raise TypeError("y_transform should have a transform method.")

    def y_transform(self, y):
        """Calls transform method on transformer object.

        Parameters
        ----------
        y : iterable
            Targets.

        Returns
        -------
        yt : iterable
            Transformed targets.
        """
        return self.y_transformer.transform(y)

    def y_inverse_transform(self, yt):
        """If available, transformed target values are transformed back to the
        original representation.

        Parameters
        ----------
        yt : iterable
            Transformed targets.

        Returns
        -------
        y : iterable
            Original targets.
        """
        return self.y_transformer.inverse_transform(yt)

    def fit(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform the
        data, then fit the transformed data using the final estimator. Target
        values are tranformed before being passed to original fit method.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.
        """
        self.y_transformer.fit(y)
        yt = self.y_transform(y)
        return super().fit(X, yt, **fit_params)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all the transforms one after the other and transform
        the data, then use fit_transform on transformed data using the
        final estimator. Target values are tranformed before being
        passed to original fit method.

        Parameters
        ----------
        X : iterable
            Training data. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable
            Training targets. Must fulfill label requirements for all steps of
            the pipeline.

        """
        self.fit(X, y, **fit_params)
        return self.transform(X)

    # pylint: disable=arguments-differ
    @if_delegate_has_method(delegate='_final_estimator')
    def predict(self, X, inverse=None):
        """Applies transforms to the data, and the predict method of the
        final estimator. Valid only if the final estimator implements
        predict.

        Parameters
        ----------
        X : iterable
            Data to predict on. Must fulfill input requirements of
            first step of the pipeline.

        inverse : bool, default: None
            Whether to apply inverse_transform on predicted values.
            If not provided, I will use ``predict_use_inverse`` to
            determine whether the inverse transform should be applied.

        """
        if inverse is None:
            inverse = self.predict_use_inverse
        y_pred = super().predict(X)
        if inverse:
            y_pred = self.y_inverse_transform(y_pred)
        return y_pred

    @if_delegate_has_method(delegate='_final_estimator')
    def score(self, X, y=None):
        """Applies transforms to the data, and the score method of the
        final estimator. Valid only if the final estimator implements
        score. Target values are tranformed before being
        passed to original score method.

        Parameters
        ----------
        X : iterable
            Data to score. Must fulfill input requirements of first step of the
            pipeline.

        y : iterable
            Targets used for scoring. Must fulfill label requirements
            for all steps of the pipeline.

        """
        yt = self.y_transform(y)
        return super().score(X, yt)

    def get_params(self, deep=True):
        # BBB This is not required for scikit-learn 0.17
        out = super().get_params(deep)
        out['steps'] = self.steps
        out['y_transformer'] = self.y_transformer
        return out


class SliceMixin:
    """Allows more comfortable access to steps of Pipeline or
    FeatureUnion.

    Create a new class that subclasses Pipeline or FeatureUnion and
    this. That class allows to:
        1) access by name (e.g. pipeline['clf'])
        2) access by index (e.g. pipeline[-1])
        3) access by slice (e.g. pipeline[:3])

    Example
    -------
    >>>  class SlicePipeline(SliceMixin, Pipeline):
    >>>      pass

    """
    def __getitem__(self, idx):
        container = (getattr(self, 'steps', False) or
                     getattr(self, 'transformer_list', False))

        if not container:
            raise AttributeError("SliceMixin requires a 'steps' or a "
                                 "'transformer_list' attribute.")

        if isinstance(idx, str):
            return dict(container)[idx]
        if isinstance(idx, slice):
            return container[idx]
        return container[idx][1]


class DictFeatureUnion(FeatureUnion):
    """This is like sklearn's FeatureUnion class, but intead of
    stacking the final features, merge them to a dictionary.

    The dictionaries keys correspond to the transformer step names, the
    values to the result of the transformation. Name collisions are not
    resolved, the user has to take care not to duplicate names.

    DictFeatureUnions can be nested.

    Parameters
    ----------
    transformer_list : list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.

    n_jobs : int, optional
        Number of jobs to run in parallel (default 1).

    transformer_weights : dict, optional
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.

    """

    def _update_transformed_dict(self, Xs):
        Xt = {}
        for (name, _), xs in zip(self.transformer_list, Xs):
            if isinstance(xs, dict):
                Xt.update(xs)
            else:
                Xt[name] = xs
        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers using X, transform the data and
        merge results into a dictionary.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.

        y : iterable, default=None
            Training targets.

        **fit_params : dict, optional
            Parameters to pass to the fit method.

        Returns
        -------
        Xt : dict
            Dictionary with the step names as keys and transformed
            data as values.

        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight,
                                        **fit_params)
            for _, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return {}

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        Xt = self._update_transformed_dict(Xs)
        return Xt

    def transform(self, X):
        """Transform X separately by each transformer, merge results
        into a dictionary.

        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        Xt : dict
            Dictionary with the step names as keys and transformed
            data as values.

        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())

        if not Xs:
            # All transformers are None
            return {}

        Xt = self._update_transformed_dict(Xs)
        return Xt


class DataFrameFeatureUnion(FeatureUnion):
    """Extends FeatureUnion to output Pandas Dataframe.

    Modified FeatureUnion that outputs a pandas dataframe if all
    transformers output a dataframe.

    Parameters
    ----------
    transformer_list: list of (string, transformer) tuples
        List of transformer objects to be applied to the data. The first
        half of each tuple is the name of the transformer.

    n_jobs: int, optional
        Number of jobs to run in parallel (default 1).

    transformer_weights: dict, optional
        Multiplicative weights for features per transformer.
        Keys are transformer names, values the weights.

    ignore_index: boolean, optional
        Strips all indexs from all dataframes before concatenation.

    copy: boolean, optional
        Set copy-Parameter of pandas concat-Function.

    keep_original: bool (default=False)
        If True, instead of only returning the transformed data,
        concatenate them to the original data and return the
        result.

    """

    def __init__(
            self,
            transformer_list,
            n_jobs=1,
            transformer_weights=None,
            verbose=False,
            ignore_index=True,
            copy=True,
            keep_original=False,
    ):
        self.ignore_index = ignore_index
        self.copy = copy
        self.keep_original = keep_original

        super(DataFrameFeatureUnion, self).__init__(
            transformer_list=transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers using X, transform the data and
        concatenate results.

        Parameters
        ----------
        X : array-like, sparse matrix or dataframe,
            shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        X_t : array-like, sparse matrix or dataframe,
            shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.

        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight,
                                        **fit_params)
            for _, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)
        if self.keep_original:
            Xs = list(itertools.chain([X], Xs))
        self._update_transformer_list(transformers)

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        elif all(isinstance(f, (pd.DataFrame, pd.Series)) for f in Xs):
            if self.ignore_index:
                Xs = [f.reset_index(drop=True) for f in Xs]
            Xs = pd.concat(Xs, axis=1, copy=self.copy)
        else:
            Xs = np.hstack(Xs)
        return Xs

    def transform(self, X):
        """Transform X separately by each transformer, concatenate
        results.

        Parameters
        ----------
        X : array-like, sparse matrix or dataframe,
            shape (n_samples, n_features)
            Input data to be transformed.

        Returns
        -------
        X_t : array-like, sparse matrix or dataframe,
            shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.

        """
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for _, trans, weight in self._iter())

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        if self.keep_original:
            Xs = list(itertools.chain([X], Xs))

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()
        elif all(isinstance(f, (pd.DataFrame, pd.Series)) for f in Xs):
            if self.ignore_index:
                Xs = [f.reset_index(drop=True) for f in Xs]
            Xs = pd.concat(Xs, axis=1, copy=self.copy)
        else:
            Xs = np.hstack(Xs)

        return Xs


def timing_decorator(
        est,
        name,
        method_name,
        sink=print,
):
    """Decorator that wraps the indicated method of the estimator into
    a wrapper that measures time.

    By default, the outputs are just printed to the console. They take a
    form that allows the user to parse each line as a dict or json.

    est : sklearn.BaseEstimator
      An sklearn estimator that is part of the profiled pipeline
      steps.

    name : str
      Name to be displayed; by default, the name given in the `steps`
      parameter of the pipeline.

    method_name : str
      Method to be profiled; either one of 'fit', 'transform',
      'fit_transform', 'predict', 'predict_proba'.

    sink : callable (default=print)
      A callable that the profiling message is sent to; e.g. the print
      function or a logger.

    """
    func = getattr(est, method_name)

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Measure time of method call and send message to sink."""
        tic = time.time()
        result = func(*args[1:], **kwargs)
        toc = time.time()

        s_name = '"name": {:<30}'.format('"' + name[:28] + '"')
        s_method = '"method": {:<20}'.format('"' + method_name[:18] + '"')
        s_dura = '"duration": {:>12.3f}'.format(toc - tic)
        s_shape_tmpl = '"shape": {:<}'
        try:
            shape = result.shape
            shape_x = '"' + 'x'.join(map(str, shape)) + '"'
            s_shape = s_shape_tmpl.format(shape_x)
        except AttributeError:
            s_shape = s_shape_tmpl.format('"-"')

        out = '{}, {}, {}, {}'.format(
            s_name,
            s_method,
            s_dura,
            s_shape,
        )

        sink("{" + out + "}")
        return result
    # pylint: disable=protected-access
    wrapper._has_timing = True
    return wrapper


def _add_timed_sequence(steps, sink):
    """For each step in steps, decorate its relevant methods."""
    seq = tosequence(steps)
    method_names = ('fit', 'transform', 'fit_transform', 'predict',
                    'predict_proba')
    for name, step in seq:
        for method_name in method_names:
            old_func = getattr(step, method_name, None)
            # pylint: disable=protected-access
            if not old_func or hasattr(old_func, '_has_timing'):
                continue

            new_func = timing_decorator(step, name, method_name, sink)
            setattr(
                step,
                new_func.__name__,
                types.MethodType(new_func, step),
            )
    return seq


def _shed_timed_sequence(steps):
    """For each step in steps, remove the decorator."""
    method_names = ('fit', 'transform', 'fit_transform', 'predict',
                    'predict_proba')
    for _, step in steps:
        for method_name in method_names:
            if not hasattr(step, method_name):
                continue

            decorated = getattr(step, method_name)
            closure = decorated.__closure__
            if closure:
                meth = closure[0].cell_contents
                setattr(step, meth.__name__, meth)


class TimedPipeline(Pipeline):
    """Timed pipeline of transforms with a final estimator.

    Note: In contrast to sklearn.pipeline.Pipeline, this additionally
    prints information about how long each fit, transformation, and
    prediction step took. Although sklearn's Pipeline has a verbose
    argument since 0.21 which also prints how long transformation
    steps took, the functionality is not exactly the
    same. E.g. TimedPipeline also prints results from prediction and
    allows to pass in a custom sink for the logs.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.

    memory : Instance of sklearn.external.joblib.Memory or string, optional \
            (default=None)
        Used to cache the fitted transformers of the pipeline. By
        default, no caching is performed. If a string is given, it is
        the path to the caching directory. Enabling caching triggers a
        clone of the transformers before fitting. Therefore, the
        transformer instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.

    sink : callable (default=print)
        The target where the string messages are sent to. Is print by
        default but could, for example, be switched to a logger.

    verbose : boolean, optional(default=False)
        If True, the time elapsed while fitting each transformer will
        be printed as it is completed. Note: This is sklearn
        functionality, not dstoolbox.

    Attributes
    ----------
    named_steps : dict
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.

    """

    def __init__(self, steps, memory=None, verbose=False, sink=print):
        # pylint: disable=super-init-not-called
        self.steps = _add_timed_sequence(steps, sink)
        self.sink = sink
        self.memory = memory
        self.verbose = verbose

        self._validate_steps()

    def __setstate__(self, state):
        state['steps'] = _add_timed_sequence(state['steps'], state['sink'])
        self.__dict__.update(state)

    def shed_timing(self):
        """Call this if you want to get rid of timing messages."""
        _shed_timed_sequence(self.steps)

    def add_timing(self):
        """Call this if you want to re-apply timing messages (after
        having called `shed_timing`).

        """
        self.steps = _add_timed_sequence(self.steps, self.sink)
