"""Tests for transformers.preprocessing.py."""

import numpy as np
import pytest


class TestXLabelEncoder:
    """
    Test XLabelEncoder
    """
    @pytest.fixture
    def encoder(self):
        from dstoolbox.transformers.preprocessing import XLabelEncoder
        return XLabelEncoder()

    @pytest.fixture
    def data(self):
        X = np.array(['c', '++a', 'b', 'c'])
        return X

    @pytest.fixture
    def data_2d(self):
        X = np.array(['c', '++a', 'b', 'c']).reshape(-1, 1)
        return X

    def test_xlabelencoder_fit(self, encoder, data):
        encoder.fit(data)
        expected_dict = {
            '<UNKNOWN>': 0,
            '++a': 1,
            'b': 2,
            'c': 3}
        assert encoder.classes_dict_ == expected_dict

    def test_xlabelencoder_transform(self, encoder, data):
        encoder.fit(data)
        Xt = encoder.transform(data)
        expected = np.array([3, 1, 2, 3]).reshape(-1, 1)
        assert (Xt == expected).all()

    def test_xlabelencoder_transform_unknown(self, encoder, data):
        encoder.fit(data)
        Xt = encoder.transform(np.array(['++a', 'a', 'b', '++']))
        expected = np.array([1, 0, 2, 0]).reshape(-1, 1)
        assert (Xt == expected).all()

    def test_xlabelencoder_fit_transform(self, encoder, data):
        Xt = encoder.fit_transform(data)
        expected = np.array([3, 1, 2, 3]).reshape(-1, 1)
        assert (Xt == expected).all()

    def test_xlabelencoder_with_2d_data(self, encoder, data_2d):
        Xt = encoder.fit_transform(data_2d)
        expected = np.array([3, 1, 2, 3]).reshape(-1, 1)
        assert (Xt == expected).all()


def mean_func(X):
    # function that has side-effects with regards to the partitioning
    # of the data
    return np.ones_like(X) * np.mean(X)


def plus_one_func(X):
    # function without side-effects with regards to the partitioning
    # of the data
    return X + 1


class TestParallelFunctionTransformer:
    @pytest.fixture
    def parallel_function_transformer_cls(self):
        from dstoolbox.transformers.preprocessing import (
            ParallelFunctionTransformer)
        return ParallelFunctionTransformer

    @pytest.fixture
    def function_transformer_cls(self):
        from sklearn.preprocessing import FunctionTransformer
        return FunctionTransformer

    @pytest.fixture
    def data(self):
        return np.arange(20).reshape(-1, 1) + 1

    @pytest.fixture
    def parallel_function_transformer(
            self, parallel_function_transformer_cls, data):
        return parallel_function_transformer_cls(func=mean_func).fit(data)

    @pytest.fixture
    def function_transformer(
            self, function_transformer_cls, data):
        return function_transformer_cls(func=mean_func, validate=True).fit(data)

    @pytest.mark.parametrize('n_jobs, expected', [
        (1, 10.5 * np.ones((20, 1)).reshape(-1, 1)),
        (2, np.vstack([
            5.5 * np.ones((10, 1)),
            15.5 * np.ones((10, 1)),
        ])),
        (3, np.vstack([
            4 * np.ones((7, 1)),
            11 * np.ones((7, 1)),
            17.5 * np.ones((6, 1)),
        ])),
        (4, np.vstack([
            3 * np.ones((5, 1)),
            8 * np.ones((5, 1)),
            13 * np.ones((5, 1)),
            18 * np.ones((5, 1)),
        ])),
        (5, np.vstack([
            2.5 * np.ones((4, 1)),
            6.5 * np.ones((4, 1)),
            10.5 * np.ones((4, 1)),
            14.5 * np.ones((4, 1)),
            18.5 * np.ones((4, 1)),
        ])),
    ])
    def test_parallel_function_transformer_n_jobs(
            self, parallel_function_transformer, data, n_jobs, expected):
        parallel_function_transformer.set_params(n_jobs=n_jobs)

        Xt = parallel_function_transformer.transform(data)
        assert (Xt == expected).all()

    @pytest.mark.parametrize('rand_data', [
        (np.random.rand(2, 3)),
        (np.random.rand(10, 1)),
        (np.random.rand(1, 6)),
        (np.random.rand(7, 1)),
    ])
    def test_parallel_function_transformer_same_as_sklearn_wo_side_effect(
            self,
            parallel_function_transformer,
            function_transformer,
            rand_data,
    ):
        parallel_function_transformer.set_params(n_jobs=2, func=plus_one_func)
        function_transformer.set_params(func=plus_one_func)

        Xt_parallel = parallel_function_transformer.transform(rand_data)
        Xt_linear = function_transformer.transform(rand_data)

        assert (Xt_parallel == Xt_linear).all()
