"""Tests for transformers.slicing.py."""

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import pytest


class TestItemSelector:
    @pytest.fixture
    def item_selector_cls(self):
        from dstoolbox.transformers import ItemSelector
        return ItemSelector

    @pytest.fixture
    def X(self):
        X = np.array([
            ['tmp', 'a', 'tmp', 'd', 'tmp'],
            ['tmp', 'b', 'tmp', 'e', 'tmp'],
            ['tmp', 'c', 'tmp', 'f', 'tmp'],
        ])
        return X

    @pytest.fixture
    def df(self):
        df = pd.DataFrame(data={
            'names': ['Alice', 'Bob', 'Charles', 'Dora', 'Eve'],
            'surnames': ['Carroll', 'Meister', 'Darwin', 'Explorer', 'Wally'],
            'age': [14., 30., 55., 7., 25.],
        })
        return df

    def test_transform(self, item_selector_cls, X):
        item_selector = item_selector_cls(key=1)
        expected = np.array(['a', 'b', 'c'])

        result = item_selector.fit_transform(X)
        assert (result == expected).all()

    def test_transform_list(self, item_selector_cls, X):
        item_selector = item_selector_cls(key=[1, 3])
        expected = np.array([['a', 'd'],
                             ['b', 'e'],
                             ['c', 'f']])

        result = item_selector.fit_transform(X)
        assert (result == expected).all()

    def test_transform_slice(self, item_selector_cls, X):
        item_selector = item_selector_cls(key=slice(1, 4, 2))
        expected = np.array([['a', 'd'],
                             ['b', 'e'],
                             ['c', 'f']])

        result = item_selector.fit_transform(X)
        assert (result == expected).all()

    def test_transform_sparse(self, item_selector_cls):
        from scipy import sparse
        X = np.zeros((3, 5))
        X[1, 1] = 9
        Xs = sparse.csr_matrix(X)

        item_selector = item_selector_cls(key=[1, -1])
        expected = np.array([[0, 0],
                             [9, 0],
                             [0, 0]])

        result = item_selector.fit_transform(Xs).toarray()
        assert (result == expected).all()

    def test_fit(self, item_selector_cls):
        transformer = item_selector_cls(key=1)
        X = 'dummy'
        y = 'dummy2'
        result = transformer.fit(X, y)
        assert result is transformer

    def test_transform_df_strings(self, item_selector_cls, df):
        item_selector = item_selector_cls(key=['names', 'age'])
        expected = pd.DataFrame(data={
            'age': [14., 30., 55., 7., 25.],
            'names': ['Alice', 'Bob', 'Charles', 'Dora', 'Eve'],
        })

        result = item_selector.fit_transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_transform_df_callable(self, item_selector_cls, df):
        item_selector = item_selector_cls(key=lambda x: x.endswith('names'))
        expected = pd.DataFrame(data={
            'names': ['Alice', 'Bob', 'Charles', 'Dora', 'Eve'],
            'surnames': ['Carroll', 'Meister', 'Darwin', 'Explorer', 'Wally'],
        })

        result = item_selector.fit_transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_transform_df_regex(self, item_selector_cls, df):
        import re
        pattern = re.compile(r'n*a')
        item_selector = item_selector_cls(key=pattern.match)
        expected = pd.DataFrame(data={
            'names': ['Alice', 'Bob', 'Charles', 'Dora', 'Eve'],
            'age': [14., 30., 55., 7., 25.],
        })

        result = item_selector.fit_transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_df_multiple_types_in_keys(self, item_selector_cls, df):
        item_selector = item_selector_cls(key=['names', 0])

        with pytest.raises(ValueError):
            item_selector.fit_transform(df)

    def test_array_wrong_type_in_key(self, item_selector_cls, X):
        item_selector = item_selector_cls(key=['1'])

        with pytest.raises(ValueError):
            item_selector.fit_transform(X)

    def test_force_2d_array_1d(self, item_selector_cls, X):
        item_selector = item_selector_cls(key=1, force_2d=True)
        result = item_selector.fit_transform(X)
        expected = np.array(['a', 'b', 'c']).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_force_2d_array_2d(self, item_selector_cls, X):
        item_selector = item_selector_cls(key=[1, 3], force_2d=True)
        result = item_selector.fit_transform(X)
        expected = np.array([['a', 'b', 'c'],
                             ['d', 'e', 'f']]).T
        assert np.array_equal(result, expected)

    def test_force_2d_array_3d(self, item_selector_cls):
        item_selector = item_selector_cls(key=[1, 3], force_2d=True)
        X = np.zeros((5, 5, 5))

        with pytest.raises(ValueError) as exc:
            item_selector.fit_transform(X)

        assert str(exc.value) == "ItemSelector cannot force 2d on 3d data."

    def test_force_2d_series(self, item_selector_cls, df):
        item_selector = item_selector_cls('names', force_2d=True)
        result = item_selector.fit_transform(df)
        expected = np.array(
            ['Alice', 'Bob', 'Charles', 'Dora', 'Eve']).reshape(-1, 1)
        assert np.array_equal(result, expected)

    def test_force_2d_df_1d(self, item_selector_cls, df):
        item_selector = item_selector_cls(['names'], force_2d=True)
        result = item_selector.fit_transform(df)
        expected = np.array(
            ['Alice', 'Bob', 'Charles', 'Dora', 'Eve']).reshape(-1, 1)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, expected)

    def test_force_2d_df_2d(self, item_selector_cls, df):
        item_selector = item_selector_cls(['names', 'age'], force_2d=True)
        result = item_selector.fit_transform(df)
        expected = np.array([
            ['Alice', 'Bob', 'Charles', 'Dora', 'Eve'],
            [14., 30., 55., 7., 25.],
        ], dtype=object).T
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, expected)

    def test_dict_input(self, item_selector_cls):
        item_selector = item_selector_cls('a')
        X = {'a': np.arange(10), 'b': np.arange(10, 20)}
        expected = np.arange(10)

        result = item_selector.fit_transform(X)
        assert (result == expected).all()
        assert result.shape == expected.shape

    def test_dict_input_2d(self, item_selector_cls):
        item_selector = item_selector_cls('a', force_2d=True)
        X = {'a': np.arange(10), 'b': np.arange(10, 20)}
        expected = np.expand_dims(np.arange(10), 1)

        result = item_selector.fit_transform(X)
        assert (result == expected).all()
        assert result.shape == expected.shape
