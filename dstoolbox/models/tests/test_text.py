"""Tests for models.text.py."""

import types
from unittest.mock import patch

import numpy as np
import pytest


class TestW2VClassifier:
    @pytest.fixture
    def mock_load_w2v_format(self):
        word2idx = {'first': 0, 'second': 1, 'third': 2, 'fourth': 3}
        syn0 = np.array([
            [0.1, 0.1, 0.1],
            [10.0, 10.0, 10.1],
            [-0.1, -0.1, -0.1],
            [-0.1, -0.1, 0.1],
        ])
        return word2idx, syn0

    @pytest.fixture
    def mock_w2v_classifier_cls(self, mock_load_w2v_format):
        """Return a W2VClassifier with a mocked fit method."""
        from dstoolbox.models import W2VClassifier
        from dstoolbox.utils import normalize_matrix

        # pylint: disable=unused-argument
        def mock_fit(self, X=None, y=None):
            word2idx, syn0 = mock_load_w2v_format
            self.word2idx_ = word2idx
            self.classes_ = np.array(sorted(word2idx, key=word2idx.get))
            self.syn0_ = normalize_matrix(syn0)
            return self

        W2VClassifier.fit = types.MethodType(mock_fit, W2VClassifier('a/path'))
        return W2VClassifier

    @pytest.fixture
    def clf(self, mock_w2v_classifier_cls):
        return mock_w2v_classifier_cls('a/path').fit()

    @staticmethod
    def assert_most_similar_are_equal(expected, results, atol=1e-5):
        for (r_word, r_simil), (e_word, e_simil) in zip(results, expected):
            assert r_word == e_word
            assert np.isclose(r_simil, e_simil, atol=atol)

    def test_word2vec_classifier_mocked_fit(self, mock_load_w2v_format):
        with patch('dstoolbox.models.text.load_w2v_format') as load:
            word2idx, syn0 = mock_load_w2v_format
            load.return_value = mock_load_w2v_format

            from dstoolbox.models import W2VClassifier
            from dstoolbox.utils import normalize_matrix

            clf = W2VClassifier('a/path')
            clf.fit()

            assert (clf.syn0_ == normalize_matrix(syn0)).all()
            assert clf.word2idx_ == word2idx
            assert clf.classes_.tolist() == ['first', 'second',
                                             'third', 'fourth']

    def test_word2vec_classifier_fit(self, mock_w2v_classifier_cls):
        clf = mock_w2v_classifier_cls.fit()
        assert isinstance(clf.word2idx_, dict)
        assert isinstance(clf.syn0_, np.ndarray)
        assert (clf.classes_ ==
                np.array(['first', 'second', 'third', 'fourth'])).all()

    @pytest.mark.parametrize('word, expected', [
        ('first', np.array([[0.1, 0.1, 0.1]])),
        ('second', np.array([[10.0, 10.0, 10.1]])),
    ])
    def test_word2vec_classifier_get_vector_from_word(
            self, clf, word, expected):
        from dstoolbox.models.text import normalize_matrix

        # pylint: disable=protected-access
        result = clf._get_vector_from_word(word)
        expected = normalize_matrix(expected).tolist()[0]
        assert result.tolist() == expected

    @pytest.mark.parametrize('vector', [
        np.array([1, 2, 3]),
        np.array([[1, 2, 3]])
    ])
    def test_word2vec_classifier_update_vocabulary_vector_1d(
            self, clf, vector):
        # pylint: disable=protected-access
        clf._update_vocabulary('fifth', vector)

        assert clf.word2idx_['fifth'] == 4
        assert clf.classes_[-1] == 'fifth'
        assert (clf.syn0_[-1] == vector).all()

    @pytest.mark.parametrize('w1, w2', [
        ('first', 'second'),
        ('second', 'third'),
        ('first', 'fourth'),
    ])
    def test_word2vec_classifier_add_word_vectors(self, clf, w1, w2):
        # pylint: disable=protected-access
        from dstoolbox.utils import normalize_matrix
        result = clf._add_word_vectors([w1, w2])
        v1 = clf._get_vector_from_word(w1)
        v2 = clf._get_vector_from_word(w2)
        expected = normalize_matrix((v1 + v2).reshape(1, -1))
        assert (result == expected).all()

    def test_word2vec_classifier_get_vector_unknown_word_raises(self, clf):
        with pytest.raises(KeyError):
            # pylint: disable=protected-access
            clf._get_vector_from_word('unknown_word')

    def test_word2vec_classifier_kneighbors_no_distances(self, clf):
        neighbors = clf.kneighbors(['first'], return_distance=False)
        assert neighbors.tolist()[0] == [1, 3, 2]

    @pytest.mark.parametrize('word, expected', [
        ('first', [1, 3, 2]),
        ('third', [3, 1, 0]),
    ])
    def test_word2vec_classifier_kneighbors_with_distances_top3(
            self, clf, word, expected):
        neighbors, distances = clf.kneighbors([word])
        assert neighbors.tolist()[0] == expected
        assert 0 <= distances[0][0] <= distances[0][1] <= distances[0][2]

    @pytest.mark.parametrize('word, expected', [
        ('first', [1, 3]),
        ('third', [3, 1]),
    ])
    def test_word2vec_classifier_kneighbors_with_distances_top2(
            self, clf, word, expected):
        neighbors, distances = clf.kneighbors([word], 2)
        assert neighbors.tolist()[0] == expected
        assert 0 <= distances[0][0] <= distances[0][1]

    @pytest.mark.parametrize('word, expected', [
        ('first', [1]),
        ('third', [3]),
    ])
    def test_word2vec_classifier_kneighbors_with_distances_top1(
            self, clf, word, expected):
        neighbors, distances = clf.kneighbors([word], 1)
        assert neighbors.tolist()[0] == expected
        assert distances[0][0] >= 0

    def test_word2vec_classifier_most_similar_multiple_positives(
            self, clf):
        results = clf.most_similar(positive=['first', 'second'])
        expected = [('first', 1.0), ('second', 1.0),
                    ('fourth', 0.33444), ('third', 0.0)]
        self.assert_most_similar_are_equal(expected, results)

    def test_word2vec_classifier_most_similar_negative_keyword_used(
            self, clf):
        with pytest.raises(NotImplementedError):
            clf.most_similar(positive=['first'], negative=['second'])

    def test_word2vec_classifier_most_similar_positive_not_exactly_one_word(
            self, clf):
        with pytest.raises(ValueError):
            clf.most_similar(positive=[])

    @pytest.mark.parametrize('word, expected', [
        ('first', [('second', 1.0),
                   ('fourth', 1.0 / 3.0),
                   ('third', 0.0)]),
        ('third', [('fourth', 2.0 / 3.0),
                   ('second', 0.0),
                   ('first', 0.0)]),
    ])
    def test_word2vec_classifier_most_similar_top3(self, word, expected, clf):
        results = clf.most_similar(word)
        self.assert_most_similar_are_equal(expected, results)

    @pytest.mark.parametrize('word, expected', [
        ('first', [('second', 1.0),
                   ('fourth', 1.0 / 3.0)]),
        ('third', [('fourth', 2.0 / 3.0),
                   ('second', 0.0)]),
    ])
    def test_word2vec_classifier_most_similar_top2(self, word, expected, clf):
        results = clf.most_similar(word)
        self.assert_most_similar_are_equal(expected, results)

    @pytest.mark.parametrize('word, expected', [
        ('first', 'second'),
        ('third', 'fourth'),
    ])
    def test_word2vec_classifier_predict_one_sample(self, word, expected, clf):
        y_pred = clf.predict([word])
        result = clf.classes_[y_pred].tolist()[0]

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (1,)
        assert result == expected

    def test_word2vec_classifier_predict_two_samples(self, clf):
        y_pred = clf.predict(['first', 'third'])
        result = clf.classes_[y_pred].tolist()
        expected = ['second', 'fourth']

        assert isinstance(y_pred, np.ndarray)
        assert y_pred.shape == (2,)
        assert result == expected

    def test_word2vec_classifier_predict_proba_raises(self, clf):
        with pytest.raises(NotImplementedError):
            clf.predict_proba(['first'])
