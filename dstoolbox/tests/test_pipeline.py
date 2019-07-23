"""Tests for pipeline.py."""

from functools import partial
import json
import pickle
import time
from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import pytest
from sklearn.datasets import make_classification
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer


class TestPipelineY:
    @pytest.fixture
    def pipeliney_cls(self):
        from dstoolbox.pipeline import PipelineY
        return PipelineY

    @pytest.fixture
    def X(self):
        X = np.array(['Alice', 'Bob', 'Charles', 'Dora', 'Eve'])
        return X

    @pytest.fixture
    def y(self):
        y = np.array(['F', 'M', 'M', 'F', 'F'])
        return y

    @pytest.fixture(params=[{'memory': False}, {'memory': True}])
    def pipeline(self, pipeliney_cls, X, y, request, tmpdir):
        """Pipeline, once with and once without memory."""
        if request.param['memory']:
            memory = str(tmpdir.mkdir('dstoolbox').join('memory'))
        else:
            memory = None
        pipeline = pipeliney_cls(
            steps=[('count', CountVectorizer(analyzer='char')),
                   ('clf', BernoulliNB())],
            memory=memory,
            y_transformer=LabelEncoder(),
        )
        return pipeline.fit(X, y)

    @pytest.fixture
    def pipeline_regr(self, pipeliney_cls):
        pipeline = pipeliney_cls(
            steps=[('count', CountVectorizer(analyzer='char')),
                   ('clf', LinearRegression())],
            y_transformer=FunctionTransformer(
                lambda x: 0.5 * x, validate=False),
        )
        return pipeline

    @pytest.fixture
    def pipeline_inverse(self, pipeliney_cls, X, y):
        pipeline = pipeliney_cls(
            steps=[('count', CountVectorizer(analyzer='char')),
                   ('clf', BernoulliNB())],
            y_transformer=LabelEncoder(),
            predict_use_inverse=True,
        )
        return pipeline.fit(X, y)

    def test_init(self, pipeliney_cls):
        with pytest.raises(TypeError):
            pipeliney_cls(
                steps=[('count', CountVectorizer(analyzer='char')),
                       ('clf', BernoulliNB())],
                y_transformer='tmp')

    def test_fit(self, pipeliney_cls, X, y):
        steps = [('dummy', Mock())]
        yt = 'tmp'

        y_transformer = Mock(transform=Mock(return_value=yt))
        pipeliney = pipeliney_cls(steps=steps, y_transformer=y_transformer)
        pipeliney.fit(X, y)

        y_expected = y_transformer.fit.call_args_list[0][0]
        assert (y_expected == y).all()
        y_expected = y_transformer.transform.call_args_list[0][0]
        assert (y_expected == y).all()

        X_expected, y_expected = steps[0][1].fit.call_args_list[0][0]
        assert (X_expected == X).all()
        assert y_expected == yt

    def test_fit_transform(self, pipeliney_cls, X, y):
        steps = [('dummy', Mock())]
        yt = 'tmp'

        y_transformer = Mock(transform=Mock(return_value=yt))
        pipeliney = pipeliney_cls(steps=steps, y_transformer=y_transformer)
        pipeliney.fit_transform(X, y)

        y_expected = y_transformer.fit.call_args_list[0][0]
        assert (y_expected == y).all()
        y_expected = y_transformer.transform.call_args_list[0][0]
        assert (y_expected == y).all()

        X_expected, y_expected = steps[0][1].fit.call_args_list[0][0]
        assert (X_expected == X).all()
        assert y_expected == yt

        X_expected = steps[0][1].transform.call_args_list[0][0]
        assert (X_expected == X).all()

    def test_y_transform(self, pipeline, y):
        assert (pipeline.y_transform(y) == [0, 1, 1, 0, 0]).all()

    def test_y_transform_inverse(self, pipeline, y):
        assert (pipeline.y_inverse_transform(pipeline.y_transform(y)) ==
                ['F', 'M', 'M', 'F', 'F']).all()

    def test_predict(self, pipeline, X):
        assert (pipeline.predict(X) == [0, 1, 1, 0, 0]).all()

    def test_predict_inverse(self, pipeline, X):
        assert (pipeline.predict(X, inverse=True) ==
                ['F', 'M', 'M', 'F', 'F']).all()

    def test_predict_use_inverse(self, pipeline_inverse, X):
        assert (pipeline_inverse.predict(X) ==
                ['F', 'M', 'M', 'F', 'F']).all()

    def test_score(self, pipeline, X, y):
        result = pipeline.score(X, y)
        assert 0.0 <= result <= 1.0

    def test_set_params(self, pipeline_regr):
        assert pipeline_regr.get_params()['y_transformer'].validate is False
        pipeline_regr.set_params(**{'y_transformer__validate': True})
        assert pipeline_regr.y_transformer.validate is True

    def test_grid_search_functional(self, pipeline_regr, X):
        y = np.array([10., 0., 0., 10., 10.])
        gs_params = {
            'count__max_features': [3, 4, 5],
            'clf__fit_intercept': [True, False],
            'y_transformer__func': [lambda x: 0.5 * x, lambda x: 2 * x],
        }
        gs = GridSearchCV(pipeline_regr, gs_params, cv=2)
        gs.fit(X, y)

    def test_is_deprecated(self, pipeliney_cls, recwarn):
        pipeliney_cls([
            ('count', CountVectorizer(analyzer='char')),
            ('clf', BernoulliNB()),
        ], y_transformer=LabelEncoder())
        warn = recwarn.pop(DeprecationWarning).message.args[0]
        msg = (
            "PipelineY is deprecated and will be removed in a future release. "
            "Please use sklearn.compose.TransformedTargetRegressor instead."
        )
        assert warn == msg


class TestSliceMixin:
    @pytest.fixture
    def slice_mixin_cls(self):
        from dstoolbox.pipeline import SliceMixin
        return SliceMixin

    @pytest.fixture
    def slice_pipeline_cls(self, slice_mixin_cls):
        class SlicePipeline(slice_mixin_cls, Pipeline):
            pass

        return SlicePipeline

    @pytest.fixture
    def X(self):
        X = np.array(['Alice', 'Bob', 'Charles', 'Dora', 'Eve'])
        return X

    @pytest.fixture
    def y(self):
        y = np.array([0, 1, 2, 3, 4])
        return y

    @pytest.fixture
    def pipeline(self, slice_pipeline_cls, X, y):
        pipeline = slice_pipeline_cls([
            ('count', CountVectorizer(analyzer='char')),
            ('tfidf', TfidfTransformer()),
            ('clf', LinearRegression()),
        ])
        return pipeline.fit(X, y)

    def test_slice_mixin_pipeline_select_item_by_name(self, pipeline):
        assert pipeline['count'] is pipeline.steps[0][1]
        assert pipeline['tfidf'] is pipeline.steps[1][1]
        assert pipeline['clf'] is pipeline.steps[2][1]

    def test_slice_mixin_pipeline_select_item_by_index(self, pipeline):
        assert pipeline[0] is pipeline.steps[0][1]
        assert pipeline[1] is pipeline.steps[1][1]
        assert pipeline[2] is pipeline.steps[2][1]

    def test_slice_mixin_pipeline_select_item_by_negative_index(
            self, pipeline):
        assert pipeline[-3] is pipeline.steps[0][1]
        assert pipeline[-2] is pipeline.steps[1][1]
        assert pipeline[-1] is pipeline.steps[2][1]

    def test_slice_mixin_pipeline_select_item_by_slice(self, pipeline):
        assert pipeline[:] == pipeline.steps
        assert pipeline[:1] == pipeline.steps[:1]
        assert pipeline[:2] == pipeline.steps[:2]
        assert pipeline[-2:] == pipeline.steps[-2:]

    def test_slice_mixin_pipeline_select_slice_copy_is_shallow(self, pipeline):
        assert pipeline[:][0][1] is pipeline.steps[0][1]
        assert pipeline[:][1][1] is pipeline.steps[1][1]
        assert pipeline[:][2][1] is pipeline.steps[2][1]

    def test_slice_mixin_pipeline_select_slice_for_new_pipeline(
            self, pipeline, X):
        # This pipeline doesn't allow to call transform because the
        # last step does not support it, but with slice we can create
        # a new pipeline that does
        with pytest.raises(AttributeError):
            pipeline.transform(X)

        new_pipeline = Pipeline(pipeline[:-1])
        new_pipeline.transform(X)

    @pytest.fixture
    def slice_feature_union_cls(self, slice_mixin_cls):
        class SliceFeatureUnion(FeatureUnion, slice_mixin_cls):
            pass

        return SliceFeatureUnion

    @pytest.fixture
    def feature_union(self, slice_feature_union_cls, X):
        feature_union = slice_feature_union_cls([
            ('count0', CountVectorizer()),
            ('count1', CountVectorizer(analyzer='char')),
        ])
        return feature_union.fit(X)

    def test_slice_mixin_feature_union_select_item_by_name(
            self, feature_union):
        assert feature_union['count0'] is feature_union.transformer_list[0][1]
        assert feature_union['count1'] is feature_union.transformer_list[1][1]

    def test_slice_mixin_feature_union_select_item_by_index(
            self, feature_union):
        assert feature_union[0] is feature_union.transformer_list[0][1]
        assert feature_union[1] is feature_union.transformer_list[1][1]

    def test_slice_mixin_feature_union_select_item_by_negative_index(
            self, feature_union):
        assert feature_union[-2] is feature_union.transformer_list[0][1]
        assert feature_union[-1] is feature_union.transformer_list[1][1]

    def test_slice_mixin_feature_union_select_item_by_slice(
            self, feature_union):
        assert feature_union[:] == feature_union.transformer_list
        assert feature_union[:2] == feature_union.transformer_list[:2]
        assert feature_union[:-1] == feature_union.transformer_list[:-1]
        assert feature_union[-1:] == feature_union.transformer_list[-1:]

    def test_slice_mixin_feature_union_select_slice_copy_is_shallow(
            self, feature_union):
        assert feature_union[:][0][1] is feature_union.transformer_list[0][1]
        assert feature_union[:][1][1] is feature_union.transformer_list[1][1]

    def test_slice_mixin_raises_if_not_pipeline_or_feature_union(
            self, slice_mixin_cls):
        class MyLR(LinearRegression, slice_mixin_cls):
            pass

        clf = MyLR()
        with pytest.raises(AttributeError):
            # pylint: disable=pointless-statement
            clf[0]


class TestDictFeatureUnion:
    @pytest.fixture
    def transformer_list(self):
        return [
            ('scaler', StandardScaler()),
            ('polynomialfeatures', PolynomialFeatures()),
        ]

    @pytest.fixture
    def mock_data(self):
        X = np.array([
            [0, 10],
            [2, 20],
        ]).astype(float)
        return X

    @pytest.fixture
    def dict_feature_union_cls(self):
        from dstoolbox.pipeline import DictFeatureUnion

        return DictFeatureUnion

    # pylint: disable=missing-docstring
    @pytest.fixture(params=[
        {'transformer_weights': None},
        {'transformer_weights': {'scaler': 1, 'polynomialfeatures': 1.5}},
    ])
    def dict_feature_union(
            self,
            dict_feature_union_cls,
            transformer_list,
            request,
    ):
        transformer_weights = request.param
        union = dict_feature_union_cls(
            transformer_list,
            transformer_weights=transformer_weights,
        )
        return union

    @pytest.fixture
    def mock_transformed(self, transformer_list, mock_data):
        Xt = {name: transformer.fit_transform(mock_data)
              for name, transformer in transformer_list}
        return Xt

    def test_dict_feature_union_transform(
            self, dict_feature_union, mock_data, mock_transformed):
        dict_feature_union.fit(mock_data)
        Xt = dict_feature_union.transform(mock_data)

        assert Xt.keys() == mock_transformed.keys()
        for k in Xt:
            assert np.allclose(Xt[k], mock_transformed[k])

    def test_dict_feature_union_fit_transform(
            self, dict_feature_union, mock_data, mock_transformed):
        Xt = dict_feature_union.fit_transform(mock_data)

        assert Xt.keys() == mock_transformed.keys()
        for k in Xt:
            assert np.allclose(Xt[k], mock_transformed[k])

    def test_nested_dict_feature_union(
            self,
            dict_feature_union_cls,
            transformer_list,
            mock_data,
            mock_transformed,
        ):
        union = dict_feature_union_cls([
            ('nested', dict_feature_union_cls(transformer_list)),
            ('another_scaler', StandardScaler()),
        ])
        Xt = union.fit_transform(mock_data)

        expected_keys = ['another_scaler', 'polynomialfeatures', 'scaler']
        assert sorted(Xt.keys()) == expected_keys

        for k in mock_transformed:
            assert np.allclose(Xt[k], mock_transformed[k])

        expected_scaled = StandardScaler().fit_transform(mock_data)
        assert np.allclose(Xt['another_scaler'], expected_scaled)


class TestDataFrameFeatureUnion:
    @pytest.fixture
    def item_selector_cls(self):
        from dstoolbox.transformers.slicing import ItemSelector
        return ItemSelector

    @pytest.fixture
    def df_feature_union_cls(self):
        from dstoolbox.pipeline import DataFrameFeatureUnion
        return DataFrameFeatureUnion

    # pylint: disable=missing-docstring
    @pytest.fixture
    def df(self):
        df = pd.DataFrame(
            data={
                'surnames': ['Carroll', 'Meister', 'Darwin',
                             'Explorer', 'Wally'],
                'age': [14., 30., 55., 7., 25.]
            },
            index=['Alice', 'Bob', 'Charles', 'Dora', 'Eve'],
        )
        return df

    @pytest.fixture
    def expected(self):
        expected = pd.DataFrame(data={
            'surnames': ['Carroll', 'Meister', 'Darwin', 'Explorer', 'Wally'],
            'age': [14., 30., 55., 7., 25.]})
        return expected

    @pytest.fixture
    def X(self):
        X = np.array([
            ['tmp', 'a', 'tmp', 'd', 'tmp'],
            ['tmp', 'b', 'tmp', 'e', 'tmp'],
            ['tmp', 'c', 'tmp', 'f', 'tmp'],
        ])
        return X

    def test_two_dataframes(
            self, item_selector_cls, df_feature_union_cls, df, expected):
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-df-1', item_selector_cls(['surnames'])),
                ('select-df-2', item_selector_cls(['age'])),
            ], ignore_index=True, copy=False)

        result = feat_union.fit_transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_two_dataframes_with_transformer_weights(
            self, item_selector_cls, df_feature_union_cls, df, expected):
        transformer_weights = {'select-df-1': 1, 'select-df-2': 2}
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-df-1', item_selector_cls(['surnames'])),
                ('select-df-2', item_selector_cls(['age'])),
            ],
            transformer_weights=transformer_weights,
            ignore_index=True,
            copy=False,
        )
        expected['age'] = 2 * expected['age']

        result = feat_union.fit_transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_two_series(
            self, item_selector_cls, df_feature_union_cls, df, expected):
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-df-1', item_selector_cls('surnames')),
                ('select-df-2', item_selector_cls('age')),
            ], ignore_index=True, copy=False)

        result = feat_union.fit_transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_dataframe_and_series(
            self, item_selector_cls, df_feature_union_cls, df, expected):
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-df-1', item_selector_cls(['surnames'])),
                ('select-df-2', item_selector_cls('age')),
            ], ignore_index=True, copy=False)

        result = feat_union.fit_transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_dataframe_and_series_fit_and_transform(
            self, item_selector_cls, df_feature_union_cls, df, expected):
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-df-1', item_selector_cls(['surnames'])),
                ('select-df-2', item_selector_cls('age')),
            ], ignore_index=True, copy=False)

        result = feat_union.fit(df).transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_two_dataframes_keep_index(
            self,
            item_selector_cls,
            df_feature_union_cls,
            df):
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-df-1', item_selector_cls(['surnames'])),
                ('select-df-2', item_selector_cls(['age'])),
            ], ignore_index=False, copy=False)

        expected = pd.DataFrame(
            data={
                'surnames': ['Carroll', 'Meister', 'Darwin', 'Explorer',
                             'Wally'],
                'age': [14., 30., 55., 7., 25.]},
            index=['Alice', 'Bob', 'Charles', 'Dora', 'Eve'],
        )

        result = feat_union.fit_transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_df_and_array(self, item_selector_cls, df_feature_union_cls, df):
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-df', item_selector_cls(['age'])),
                ('select-array', Pipeline([
                    ('select', item_selector_cls(['age'])),
                    ('to-array', FunctionTransformer(
                        lambda x: x, validate=True)),
                ])),
            ],
            ignore_index=True,
            copy=False)

        expected = np.array([
            [14., 30., 55., 7., 25.],
            [14., 30., 55., 7., 25.]
        ]).T

        result = feat_union.fit_transform(df)
        assert (result == expected).all()

    def test_arrays_only(self, item_selector_cls, df_feature_union_cls, X):
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-col-1', item_selector_cls(key=[1])),
                ('select-col-3', item_selector_cls(key=[3])),
            ],
            ignore_index=True,
            copy=False)

        expected = np.array([['a', 'd'],
                             ['b', 'e'],
                             ['c', 'f']])

        result = feat_union.fit_transform(X)
        assert (result == expected).all()

    def test_arrays_only_2(self, item_selector_cls, df_feature_union_cls, X):
        df_feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-col-1', item_selector_cls(key=[1])),
                ('select-col-3', item_selector_cls(key=[3])),
            ],
            ignore_index=True,
            copy=False)

        feat_union = FeatureUnion(
            transformer_list=[
                ('select-col-1', item_selector_cls(key=[1])),
                ('select-col-3', item_selector_cls(key=[3])),
            ])

        df_feat_result = df_feat_union.fit_transform(X)
        feat_result = feat_union.fit_transform(X)

        assert (df_feat_result == feat_result).all()

    def test_two_dataframes_fit(
            self, item_selector_cls, df_feature_union_cls, df, expected):
        feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-df-1', item_selector_cls(['surnames'])),
                ('select-df-2', item_selector_cls(['age'])),
            ], ignore_index=True, copy=False)

        feat_union.fit(df)
        result = feat_union.transform(df)
        assert_frame_equal(result.sort_index(axis=1),
                           expected.sort_index(axis=1))

    def test_sparse_data_fit(
            self, item_selector_cls, df_feature_union_cls, X):
        df_feat_union = df_feature_union_cls(
            transformer_list=[
                ('select-array01', Pipeline([
                    ('select', item_selector_cls([0, 1])),
                ])),
                ('select-array02', Pipeline([
                    ('select', item_selector_cls([0, 2])),
                ])),
            ],
            ignore_index=True,
            copy=False)

        feat_union = FeatureUnion(
            transformer_list=[
                ('select-array01', Pipeline([
                    ('select', item_selector_cls([0, 1])),
                ])),
                ('select-array02', Pipeline([
                    ('select', item_selector_cls([0, 2])),
                ])),
            ])

        X_sparse = CountVectorizer(analyzer='char').fit_transform(X[:, 1])

        df_feat_union.fit(X_sparse)
        feat_union.fit(X_sparse)

        df_feat_result = df_feat_union.transform(X_sparse)
        feat_result = feat_union.transform(X_sparse)

        assert (df_feat_result != feat_result).nnz == 0

    def test_keep_original_df(
            self, df_feature_union_cls, item_selector_cls, df, expected):
        df_feat_union = df_feature_union_cls([
            ('double_age', Pipeline([
                ('select_age', item_selector_cls('age', force_2d=True)),
                ('double', FunctionTransformer(
                    lambda x: 2 * x, validate=False)),
                ('to_df', FunctionTransformer(
                    partial(pd.DataFrame, columns=['double_age']))),
            ])),
        ], keep_original=True)
        expected['double_age'] = 2 * expected['age']

        result = df_feat_union.fit_transform(df)
        assert result.equals(expected)

    def test_keep_original_ndarray(
            self, df_feature_union_cls, item_selector_cls, df, expected):
        df_feat_union = df_feature_union_cls([
            ('double_age', Pipeline([
                ('select_age', item_selector_cls('age', force_2d=True)),
                ('double', FunctionTransformer(
                    lambda x: 2 * x, validate=False)),
            ])),
        ], keep_original=True)
        expected['double_age'] = 2 * expected['age']

        result = df_feat_union.fit_transform(df)
        assert (result == expected.values).all()

    def test_keep_original_df_transform(
            self, df_feature_union_cls, item_selector_cls, df, expected):
        df_feat_union = df_feature_union_cls([
            ('double_age', Pipeline([
                ('select_age', item_selector_cls('age', force_2d=True)),
                ('double', FunctionTransformer(
                    lambda x: 2 * x, validate=False)),
                ('to_df', FunctionTransformer(
                    partial(pd.DataFrame, columns=['double_age']))),
            ])),
        ], keep_original=True)
        expected['double_age'] = 2 * expected['age']

        result = df_feat_union.fit(df).transform(df)
        assert result.equals(expected)


def _slow23(X):
    time.sleep(0.023 - 5e-4)
    return X


def _slow55(X):
    time.sleep(0.055 - 5e-4)
    return X


class TestTimedPipeline:
    def split_line(self, line):
        line = line.strip('{}')
        parts = line.split(',')
        return [part.strip() for part in parts]

    # pylint: disable=missing-docstring
    def assert_lines_correct_form(self, lines):
        for line in lines:
            assert line.startswith('{')
            assert line.endswith('}')

            parts = self.split_line(line)
            assert parts[0].startswith('"name":')
            assert parts[1].startswith('"method":')
            assert parts[2].startswith('"duration":')
            assert parts[3].startswith('"shape":')

            json.loads(line)  # is parseable as json

    def assert_lines_same_output(self, line0, line1):
        # problem: timing is a little random, so exact equality cannot be
        # guaranteed.
        assert len(line0.strip('{} ')) == len(line1.strip('{} '))
        dct0 = json.loads(line0)
        dct1 = json.loads(line1)
        for key in ('name', 'method', 'shape'):
            assert dct0[key] == dct1[key]
        assert np.isclose(dct0['duration'], dct1['duration'], atol=0.02)

    @pytest.fixture
    def timed_pipeline_cls(self):
        from dstoolbox.pipeline import TimedPipeline
        return TimedPipeline

    @pytest.fixture
    def steps(self):
        """Pipeline steps with 2 transformers and 1 classifier."""
        clf = LogisticRegression(solver='liblinear')
        # add a mock transform method so that we can call
        # fit_transform on pipeline
        clf.transform = clf.predict
        steps = [
            ('sleep_0023', FunctionTransformer(_slow23, validate=True)),
            ('sleep_0055', FunctionTransformer(_slow55, validate=True)),
            ('clf', clf),
        ]
        return steps

    @pytest.fixture(params=[{'memory': False}, {'memory': False}])
    def timed_pipeline(self, timed_pipeline_cls, steps, request, tmpdir):
        if request.param['memory']:
            memory = str(tmpdir.mkdir('dstoolbox').join('memory'))
        else:
            memory = None
        sink = Mock()
        timed_pipeline = timed_pipeline_cls(steps, sink=sink, memory=memory)
        return timed_pipeline

    @pytest.fixture
    def data(self):
        return make_classification()

    @pytest.fixture
    def expected(self):
        return [(
            '{"name": "sleep_0023"                  , "method": "transform"   '
            '      , "duration":        0.023, "shape": "100x20"}'
        ), (
            '{"name": "sleep_0055"                  , "method": "transform"   '
            '      , "duration":        0.055, "shape": "100x20"}'
        )] * 2

    def test_pipeline_is_functional(self, timed_pipeline, data):
        X, y = data
        # does not raise
        timed_pipeline.fit(X, y)
        timed_pipeline.predict(X)

    def test_sink_called_correctly_fit(self, timed_pipeline, data, expected):
        sink = timed_pipeline.sink
        X, y = data
        timed_pipeline.fit(X, y)

        lines = [c[0][0] for c in sink.call_args_list]
        assert len(lines) == 3 + 3 + 1
        self.assert_lines_correct_form(lines)
        self.assert_lines_same_output(lines[1], expected[0])
        self.assert_lines_same_output(lines[4], expected[1])

    def test_sink_called_correctly_predict(
            self, timed_pipeline, data, expected):
        sink = timed_pipeline.sink
        X, y = data
        timed_pipeline.fit(X, y).predict(X)

        lines = [c[0][0] for c in sink.call_args_list]
        assert len(lines) == 3 + 3 + 1 + 1 + 1 + 1
        self.assert_lines_correct_form(lines)
        self.assert_lines_same_output(lines[1], expected[0])
        self.assert_lines_same_output(lines[4], expected[1])
        self.assert_lines_same_output(lines[7], expected[2])
        self.assert_lines_same_output(lines[8], expected[3])

    def test_sink_called_correctly_predict_proba(
            self, timed_pipeline, data, expected):
        sink = timed_pipeline.sink
        X, y = data
        timed_pipeline.fit(X, y).predict_proba(X)

        lines = [c[0][0] for c in sink.call_args_list]
        assert len(lines) == 3 + 3 + 1 + 1 + 1 + 1
        self.assert_lines_correct_form(lines)
        self.assert_lines_same_output(lines[1], expected[0])
        self.assert_lines_same_output(lines[4], expected[1])
        self.assert_lines_same_output(lines[7], expected[2])
        self.assert_lines_same_output(lines[8], expected[3])

    def test_sink_called_correctly_other_shape(self, timed_pipeline, data):
        sink = timed_pipeline.sink
        X, y = data
        timed_pipeline.fit(X[:50, :5], y[:50]).predict_proba(X[:75, :5])

        lines = [c[0][0] for c in sink.call_args_list]
        assert len(lines) == 3 + 3 + 1 + 1 + 1 + 1
        self.assert_lines_correct_form(lines)
        self.assert_lines_same_output(lines[1], (
            '{"name": "sleep_0023"                  , "method": "transform"   '
            '      , "duration":        0.023, "shape": "50x5"}'))
        self.assert_lines_same_output(lines[4], (
            '{"name": "sleep_0055"                  , "method": "transform"   '
            '      , "duration":        0.055, "shape": "50x5"}'))
        self.assert_lines_same_output(lines[7], (
            '{"name": "sleep_0023"                  , "method": "transform"   '
            '      , "duration":        0.023, "shape": "75x5"}'))
        self.assert_lines_same_output(lines[8], (
            '{"name": "sleep_0055"                  , "method": "transform"   '
            '      , "duration":        0.055, "shape": "75x5"}'))

    def test_very_long_name(self, timed_pipeline_cls, steps, data, expected):
        steps[0] = (
            'a name that is much longer than the line', steps[0][1])
        timed_pipeline = timed_pipeline_cls(steps, sink=Mock())
        sink = timed_pipeline.sink
        X, y = data
        timed_pipeline.fit(X, y)

        lines = [c[0][0] for c in sink.call_args_list]
        assert len(lines) == 3 + 3 + 1
        self.assert_lines_correct_form(lines)
        self.assert_lines_same_output(lines[1], (
            '{"name": "a name that is much longer t", "method": "transform"'
            '         , "duration":        0.023, "shape": "100x20"}'))
        self.assert_lines_same_output(lines[4], expected[1])

    def test_pipeline_is_pickleable(
            self, timed_pipeline_cls, data, tmpdir, capsys):
        # Can't pickle Mocks or slow_func, thus test is more
        # convoluted.
        X, y = data
        timed_pipeline = timed_pipeline_cls([
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(solver='liblinear')),
        ], sink=print)

        timed_pipeline.fit(X, y)
        y_before = timed_pipeline.predict_proba(X)

        p = tmpdir.join('timed_pipeline.pkl')
        with p.open('wb') as f:
            pickle.dump(timed_pipeline, f)
        with p.open('rb') as f:
            loaded_pipeline = pickle.load(f)

        y_after = loaded_pipeline.predict_proba(X)
        assert np.allclose(y_before, y_after)

        stdout = capsys.readouterr()[0].strip()
        lines = stdout.split('\n')
        assert len(lines) == 4 + 2 + 2  # from fit + 2 x predict_proba
        self.assert_lines_correct_form(lines)

    def test_shed_timing(self, timed_pipeline, data):
        sink = timed_pipeline.sink
        X, y = data

        timed_pipeline.shed_timing()
        timed_pipeline.fit(X, y).predict(X)
        assert sink.call_count == 0

    def test_add_timing(self, timed_pipeline, data, expected):
        sink = timed_pipeline.sink
        X, y = data

        timed_pipeline.shed_timing()
        timed_pipeline.add_timing()
        timed_pipeline.fit(X, y).predict(X)

        lines = [c[0][0] for c in sink.call_args_list]
        assert len(lines) == 3 + 3 + 1 + 1 + 1 + 1
        self.assert_lines_correct_form(lines)
        self.assert_lines_same_output(lines[1], expected[0])
        self.assert_lines_same_output(lines[4], expected[1])
        self.assert_lines_same_output(lines[7], expected[2])
        self.assert_lines_same_output(lines[8], expected[3])

    def test_excess_add_timing(self, timed_pipeline, data, expected):
        sink = timed_pipeline.sink
        X, y = data

        timed_pipeline.add_timing()
        timed_pipeline.fit(X, y).predict(X)

        lines = [c[0][0] for c in sink.call_args_list]
        assert len(lines) == 3 + 3 + 1 + 1 + 1 + 1
        self.assert_lines_correct_form(lines)
        self.assert_lines_same_output(lines[1], expected[0])
        self.assert_lines_same_output(lines[4], expected[1])
        self.assert_lines_same_output(lines[7], expected[2])
        self.assert_lines_same_output(lines[8], expected[3])

    def test_excess_shed_timing(self, timed_pipeline, data):
        sink = timed_pipeline.sink
        X, y = data

        timed_pipeline.shed_timing()
        timed_pipeline.shed_timing()
        timed_pipeline.fit(X, y).predict(X)
        assert sink.call_count == 0

    def test_with_pipeline(self, timed_pipeline_cls, data, steps):
        # Currently, sklearn's Pipeline.transform is a property!
        X, y = data
        timed_pipeline = timed_pipeline_cls([
            ('step0', Pipeline(steps))
        ])
        timed_pipeline.fit(X, y)
        timed_pipeline.transform(X)
        timed_pipeline.shed_timing()
        timed_pipeline.transform(X)
