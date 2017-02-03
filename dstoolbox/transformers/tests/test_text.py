"""Tests for transformers.text.py."""

import types

import numpy as np
import pytest


class TestW2VTransformer:
    @pytest.fixture
    def w2v_transformer_cls(self):
        from dstoolbox.transformers import W2VTransformer
        return W2VTransformer

    @pytest.fixture
    def mock_load_w2v_format(self):
        word2idx = {"herren": 0, "damen": 1, "nike": 2}
        syn0 = np.arange(15).reshape(3, 5).astype(float)
        return word2idx, syn0

    @pytest.fixture
    def mock_w2v_transformer_cls(
            self, w2v_transformer_cls, mock_load_w2v_format):
        # pylint: disable=unused-argument
        def mock_fit(self, X=None, y=None):
            self.word2idx_, self.syn0_ = mock_load_w2v_format
            return self

        w2v_transformer_cls.fit = types.MethodType(
            mock_fit, w2v_transformer_cls('a/path'))
        return w2v_transformer_cls

    @pytest.fixture
    def transformer(self, mock_w2v_transformer_cls):
        return mock_w2v_transformer_cls('a_path').fit()

    def test_get_vector_from_words_known_words(self, transformer):
        words = ["herren", "damen"]
        # pylint: disable=protected-access
        result = transformer._get_vector_from_words(words)
        expected = np.array([[2.5, 3.5, 4.5, 5.5, 6.5]])
        assert np.array_equal(result, expected)

    def test_get_vector_from_words_min_aggregation(self, transformer):
        words = ["herren", "damen"]
        transformer.aggr_func = np.min
        # pylint: disable=protected-access
        result = transformer._get_vector_from_words(words)
        expected = np.array([[0., 1., 2., 3., 4.]])
        assert np.array_equal(result, expected)

    def test_get_vector_from_words_unknown_word(self, transformer):
        words = ["unbekannt"]
        # pylint: disable=protected-access
        result = transformer._get_vector_from_words(words)
        expected = np.array([[0., 0., 0., 0., 0.]])
        assert np.array_equal(result, expected)

    def test_get_vector_from_words_mixed(self, transformer):
        words = ["unbekannt", "damen"]
        # pylint: disable=protected-access
        result = transformer._get_vector_from_words(words)
        expected = np.array([[5., 6., 7., 8., 9.]])
        assert np.array_equal(result, expected)

    def test_transform(self, transformer):
        X = np.array(["herren damen", "unbekannt", "damen unbekannt", "nike"])
        result = transformer.transform(X)
        expected = np.array([
            [2.5, 3.5, 4.5, 5.5, 6.5],
            [0., 0., 0., 0., 0.],
            [5., 6., 7., 8., 9.],
            [10., 11., 12., 13., 14.],
        ])
        assert np.array_equal(result, expected)

    def test_fit_load_w2v_data(self, w2v_transformer_cls):
        transformer = w2v_transformer_cls.fit()
        assert transformer.word2idx_ == {"herren": 0, "damen": 1, "nike": 2}
        assert (transformer.syn0_ ==
                np.arange(15).reshape(3, 5).astype(float)).all()
