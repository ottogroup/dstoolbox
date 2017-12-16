"""Tests for transformers.casting.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from dstoolbox.pipeline import DictFeatureUnion
from dstoolbox.transformers import ItemSelector


class TestToDataFrame:
    @pytest.fixture
    def todf_cls(self):
        from dstoolbox.transformers import ToDataFrame
        return ToDataFrame

    @pytest.fixture
    def df(self):
        return pd.DataFrame({'a': np.arange(10), 'b': np.arange(10, 20)})

    def test_with_list(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector('a')),
            ('values', FunctionTransformer(
                lambda x: x.values.tolist(), validate=False)),
            ('to_df', todf_cls()),
        ])

        result = pipe.fit_transform(df)
        expected = df.rename(columns={'a': 0})[[0]]
        assert result.equals(expected)

    def test_with_1d_array(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector('a')),
            ('values', FunctionTransformer(
                lambda x: x.values, validate=False)),
            ('to_df', todf_cls()),
        ])

        result = pipe.fit_transform(df)
        expected = df.rename(columns={'a': 0})[[0]]
        assert result.equals(expected)

    def test_with_2d_array(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector(['a', 'b'])),
            ('values', FunctionTransformer(
                lambda x: x.values, validate=False)),
            ('to_df', todf_cls()),
        ])

        result = pipe.fit_transform(df)
        expected = df.rename(columns={'a': 0, 'b': 1})
        assert result.equals(expected)

    def test_with_series(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector('a')),
            ('to_df', todf_cls()),
        ])

        result = pipe.fit_transform(df)
        expected = df[['a']]
        assert result.equals(expected)

    def test_with_df(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector(['a', 'b'])),
            ('to_df', todf_cls()),
        ])

        result = pipe.fit_transform(df)
        assert result.equals(df)

    def test_with_dict_1_value(self, todf_cls, df):
        pipe = Pipeline([
            ('union', DictFeatureUnion([
                ('a', ItemSelector('a')),
            ])),
            ('to_df', todf_cls()),
        ])

        result = pipe.fit_transform(df)
        expected = df[['a']]
        assert result.equals(expected)

    def test_with_dict_2_values(self, todf_cls, df):
        pipe = Pipeline([
            ('union', DictFeatureUnion([
                ('b', ItemSelector('b')),
                ('a', ItemSelector('a')),
            ])),
            ('to_df', todf_cls()),
        ])

        result = pipe.fit_transform(df)
        assert result.equals(df)

    def test_with_dict_list_value(self, todf_cls):
        X = {'a': [0, 1, 2, 3, 4]}
        result = todf_cls().fit_transform(X)
        expected = pd.DataFrame(data=np.arange(5), columns=['a'])

        assert result.equals(expected)

    def test_with_dict_nx1_array_value(self, todf_cls):
        X = {'a': np.arange(5).reshape(-1, 1)}
        result = todf_cls().fit_transform(X)
        expected = pd.DataFrame(data=np.arange(5), columns=['a'])

        assert result.equals(expected)

    def test_with_dict_nxm_array_value(self, todf_cls):
        X = {'a': np.arange(15).reshape(5, 3),
             'b': np.arange(5)}
        with pytest.raises(ValueError) as exc:
            todf_cls().fit_transform(X)

        expected = "dict values must be 1d arrays."
        assert str(exc.value) == expected

    def test_with_dict_3d_array_value(self, todf_cls):
        X = {'a': np.arange(15).reshape(5, 3, 1)}
        with pytest.raises(ValueError) as exc:
            todf_cls().fit_transform(X)

        expected = "dict values must be 1d arrays."
        assert str(exc.value) == expected

    def test_with_dict_columns_sorted(self, todf_cls):
        import string
        az = string.ascii_lowercase
        df = pd.DataFrame({c: np.arange(3) for c in az})

        pipe = Pipeline([
            ('union', DictFeatureUnion([
                (c, ItemSelector(c)) for c in az[::-1]
            ])),
            ('to_df', todf_cls()),
        ])

        result = pipe.fit_transform(df)
        cols = result.columns.tolist()
        assert cols == sorted(cols)

    def test_with_column_name_str(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector('a')),
            ('values', FunctionTransformer(
                lambda x: x.values, validate=False)),
            ('to_df', todf_cls(columns='abc')),
        ])

        result = pipe.fit_transform(df)
        expected = df.rename(columns={'a': 'abc'})[['abc']]
        assert result.equals(expected)

    def test_with_list_column_name(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector('a')),
            ('values', FunctionTransformer(
                lambda x: x.values.tolist(), validate=False)),
            ('to_df', todf_cls(columns=['c'])),
        ])

        result = pipe.fit_transform(df)
        expected = df.rename(columns={'a': 'c'})[['c']]
        assert result.equals(expected)

    def test_with_array_column_names(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector(['a', 'b'])),
            ('values', FunctionTransformer(
                lambda x: x.values, validate=False)),
            ('to_df', todf_cls(columns=['c', 'd'])),
        ])

        result = pipe.fit_transform(df)
        expected = df.rename(columns={'a': 'c', 'b': 'd'})
        assert result.equals(expected)

    def test_with_series_column_name(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector('a')),
            ('to_df', todf_cls(columns=['c'])),
        ])

        result = pipe.fit_transform(df)
        expected = df.rename(columns={'a': 'c'})[['c']]
        assert result.equals(expected)

    def test_with_dict_column_name_raises(self, todf_cls, df):
        pipe = Pipeline([
            ('union', DictFeatureUnion([
                ('b', ItemSelector('b')),
                ('a', ItemSelector('a')),
            ])),
            ('to_df', todf_cls(columns=['c', 'd'])),
        ])

        with pytest.raises(ValueError) as exc:
            pipe.fit_transform(df)

        expected = ("ToDataFrame with explicit column names cannot "
                    "transform a dictionary because the dictionary's "
                    "keys already determine the column names.")
        assert str(exc.value) == expected

    def test_with_df_column_name_raises(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector(['a', 'b'])),
            ('to_df', todf_cls(columns=['c', 'd'])),
        ])

        with pytest.raises(ValueError) as exc:
            pipe.fit_transform(df)

        expected = ("ToDataFrame with explicit column names cannot "
                    "transform a DataFrame because the DataFrame's "
                    "columns already determine the column names.")
        assert str(exc.value) == expected

    def test_with_list_wrong_number_of_column_names_raises(self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector('a')),
            ('values', FunctionTransformer(
                lambda x: x.values.tolist(), validate=False)),
            ('to_df', todf_cls(columns=['c', 'd'])),
        ])

        with pytest.raises(ValueError) as exc:
            pipe.fit_transform(df)

        expected = ("ToDataFrame with more than one column name cannot "
                    "transform a list.")
        assert str(exc.value) == expected

    def test_with_series_wrong_number_of_column_names_raises(
            self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector('a')),
            ('to_df', todf_cls(columns=['c', 'd'])),
        ])

        with pytest.raises(ValueError) as exc:
            pipe.fit_transform(df)

        expected = ("ToDataFrame with more than one column name cannot "
                    "transform a Series object.")
        assert str(exc.value) == expected

    def test_with_array_wrong_number_of_column_names_raises(
            self, todf_cls, df):
        pipe = Pipeline([
            ('select', ItemSelector(['a', 'b'])),
            ('values', FunctionTransformer(
                lambda x: x.values, validate=False)),
            ('to_df', todf_cls(columns=['c', 'd', 'e'])),
        ])

        with pytest.raises(ValueError) as exc:
            pipe.fit_transform(df)

        expected = ("ToDataFrame was given data with 2 columns but "
                    "was initialized with 3 column names.")
        assert str(exc.value) == expected
