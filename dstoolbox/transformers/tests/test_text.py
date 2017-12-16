"""Tests for transformers.text.py."""

import numpy as np
import pytest


class TestTextFeaturizer:
    @pytest.fixture
    def featurizer_cls(self):
        from dstoolbox.transformers import TextFeaturizer
        return TextFeaturizer

    @pytest.fixture
    def featurizer(self, featurizer_cls):
        return featurizer_cls()

    @pytest.fixture
    def text(self):
        return [
            'aa bb Bbb',
            'Bbb bb',
            'cc bb',
            '',
        ]

    @pytest.fixture
    def expected(self):
        return np.asarray([
            [0, 1, 2],
            [2, 1],
            [3, 1],
            [],
        ])

    @pytest.fixture
    def vocab(self):
        return {'aa', 'bb', 'bbb', 'cc'}

    def assert_arrs_equal(self, t0, t1):
        for t in (t0, t1):
            assert isinstance(t, np.ndarray)
        assert len(t0) == len(t1)
        for row0, row1 in zip(t0, t1):
            assert isinstance(row0, list)
            assert isinstance(row1, list)
            assert row0 == row1

    def test_fit_vocab(self, featurizer, text, vocab):
        featurizer.fit(text)
        assert set(featurizer.vocabulary_) == vocab

    def test_fit_transform_vocab(self, featurizer, text, vocab):
        featurizer.fit_transform(text)
        assert set(featurizer.vocabulary_) == vocab

    def test_fit_and_transform_correct_output(self, featurizer, text, expected):
        result = featurizer.fit(text).transform(text)
        self.assert_arrs_equal(result, expected)

    def test_fit_transform_correct_output(self, featurizer, text, expected):
        result = featurizer.fit_transform(text)
        self.assert_arrs_equal(result, expected)

    def test_max_features(self, featurizer_cls, text):
        result = featurizer_cls(max_features=2).fit_transform(text)
        expected = np.asarray([
            [0, 1],
            [1, 0],
            [0],
            [],
        ])
        self.assert_arrs_equal(result, expected)

    def test_lower_case_false(self, featurizer_cls, text):
        featurizer = featurizer_cls(lowercase=False)
        featurizer.fit(text)
        assert set(featurizer.vocabulary_) == {'aa', 'bb', 'Bbb', 'cc'}

    def test_bigrams(self, featurizer_cls, text):
        featurizer = featurizer_cls(ngram_range=(2, 2))
        featurizer.fit(text)
        expected = {'aa bb', 'bb bbb', 'bbb bb', 'cc bb'}
        assert set(featurizer.vocabulary_) == expected

    def test_charachter_unigram(self, featurizer_cls, text):
        featurizer = featurizer_cls(analyzer='char')
        result = featurizer.fit(text).transform(text)
        a = featurizer.vocabulary_['a']
        b = featurizer.vocabulary_['b']
        c = featurizer.vocabulary_['c']
        s = featurizer.vocabulary_[' ']
        expected = np.asarray([
            [a, a, s, b, b, s, b, b, b],
            [b, b, b, s, b, b],
            [c, c, s, b, b],
            [],
        ])
        self.assert_arrs_equal(result, expected)

    def test_unknown_token_fit_and_transform(self, featurizer_cls, text):
        featurizer = featurizer_cls(min_df=2, unknown_token='<UNK>').fit(text)
        expected_vocab = {'bb': 0, 'bbb': 1, '<UNK>': 2}
        assert featurizer.vocabulary_ == expected_vocab

        result = featurizer.transform(text)
        expected = np.asarray([
            [2, 0, 1],
            [1, 0],
            [2, 0],
            [],
        ])
        self.assert_arrs_equal(result, expected)

    def test_unknown_token_fit_transform(self, featurizer_cls, text):
        featurizer = featurizer_cls(min_df=2, unknown_token='<UNK>')
        result = featurizer.fit_transform(text)

        expected_vocab = {'bb': 0, 'bbb': 1, '<UNK>': 2}
        assert featurizer.vocabulary_ == expected_vocab

        result = featurizer.transform(text)
        expected = np.asarray([
            [2, 0, 1],
            [1, 0],
            [2, 0],
            [],
        ])
        self.assert_arrs_equal(result, expected)

    def test_unknown_token_wrong_type(self, featurizer_cls):
        with pytest.raises(TypeError) as exc:
            featurizer_cls(unknown_token=3)
        expected = ("unknown_token must be None or a str, "
                    "not of type <class 'int'>.")
        assert exc.value.args[0] == expected

    def test_binary_true_raises(self, featurizer_cls):
        with pytest.raises(ValueError) as exc:
            featurizer_cls(binary=True)
        expected = "binary=True does not work with TextFeaturizer."
        assert exc.value.args[0] == expected

    @pytest.fixture
    def padder(self):
        from dstoolbox.transformers import Padder2d
        return Padder2d(pad_value=55, max_len=2, dtype=int)

    @pytest.fixture
    def pipe(self, featurizer, padder):
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ('featurizer', featurizer),
            ('padder', padder),
        ])

    def test_text_featurizer_works_with_padder2d(self, pipe, text):
        result = pipe.fit_transform(text)
        expected = np.asarray([
            [0, 1],
            [2, 1],
            [3, 1],
            [55, 55],
        ])
        assert np.allclose(result, expected)
