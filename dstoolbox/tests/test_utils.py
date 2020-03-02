"""Tests for utils.py."""

from operator import itemgetter
import time

import numpy as np
import pytest
from scipy.spatial.distance import cosine
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


np.random.seed(17411)


class TestNormalizeMatrix:
    @pytest.fixture
    def normalize_matrix(self):
        from dstoolbox.utils import normalize_matrix
        return normalize_matrix

    @pytest.mark.parametrize('arr', [
        np.eye(5),
        np.ones((100, 100)),
        np.random.rand(70, 3),
        np.random.randn(10, 30),
    ])
    def test_normalize_matrix_diagonal_is_one(self, arr, normalize_matrix):
        normalized = normalize_matrix(arr)
        dot = np.dot(normalized, normalized.T)
        assert np.allclose(np.diag(dot), 1.0)


class TestCosineSimilarity:
    @pytest.fixture
    def cosine_similarity(self):
        from dstoolbox.utils import cosine_similarity
        return cosine_similarity

    def test_cosine_similarity_incompatible_shapes(self, cosine_similarity):
        arr0 = np.zeros((100, 45))
        arr1 = np.zeros((200, 46))
        with pytest.raises(ValueError) as exc:
            cosine_similarity(arr0, arr1)

        assert str(exc.value) == "Incompatible shapes: 45 vs 46."

    def test_cosine_similarity_specific_values(self, cosine_similarity):
        arr0 = np.array([[0, 0.5, -1., 3, 0]])
        arr1 = 2 * arr0
        arr2 = -1 * arr0

        assert np.isclose(cosine_similarity(arr0, arr1), 1.)
        assert np.isclose(cosine_similarity(arr0, arr2), -1.)

    def test_cosine_similarity_specific_values_2d(self, cosine_similarity):
        arr0 = np.array([
            [0, 0.5, -1., 3, 0],
            [0, 0.5, -1., 3, 0],
        ])
        arr1 = np.array([[-1], [1]]) * arr0

        assert np.allclose(cosine_similarity(arr0, arr1), [-1., 1])

    def test_cosine_similarity_different_num_samples(self, cosine_similarity):
        arr0 = np.random.random((50, 10))
        arr1 = np.random.random((40, 10))
        result = cosine_similarity(arr0, arr1)

        assert result.shape == (50, 40)

    def test_cosine_similarity_arity_1(self, cosine_similarity):
        arr = np.random.random((50, 10))
        result_arity_1 = cosine_similarity(arr)
        result_arity_2 = cosine_similarity(arr, arr)

        assert np.allclose(result_arity_1, result_arity_2)

    @pytest.mark.parametrize('vec', np.random.random((100, 40)))
    def test_cosine_similarity_comparison_scipy(self, cosine_similarity, vec):
        arr0 = vec[:20].reshape(1, -1)
        arr1 = vec[20:].reshape(1, -1)
        result = cosine_similarity(arr0, arr1)
        expected = 1 - cosine(arr0, arr1)
        assert np.isclose(result, expected)


class TestFastArgsort:
    @pytest.fixture
    def fast_argsort(self):
        from dstoolbox.utils import fast_argsort
        return fast_argsort

    def timeit(self, func, *args):
        tic = time.time()
        func(*args)
        toc = time.time()
        return toc - tic

    @pytest.mark.parametrize('vec', [
        np.arange(10)[::-1],
        np.random.rand(20),
        np.random.randn(30),
        np.arange(100)[::-1],
        np.random.rand(200),
        np.random.randn(300),
    ])
    def test_compare_fast_argsort_with_argsort_all(self, vec, fast_argsort):
        slow = np.argsort(vec)
        fast = fast_argsort(vec, len(vec))
        assert (slow == fast).all()

    @pytest.mark.parametrize('vec, n', [
        (np.arange(10)[::-1], 5),
        (np.random.rand(20), 10),
        (np.random.randn(30), 15),
        (np.arange(100)[::-1], 15),
        (np.random.rand(200), 10),
        (np.random.randn(300), 5),
    ])
    def test_compare_fast_argsort_with_argsort_only_n_smallest(
            self, vec, n, fast_argsort):
        slow = np.argsort(vec)[:n]
        fast = fast_argsort(vec, n)
        assert (slow == fast).all()

    def test_fast_argsort_is_faster_than_argsort(self, fast_argsort):
        num_tests = 20
        min_success = 18
        successes = []

        for _ in range(num_tests):
            vec = np.random.rand(100000)
            time_slow = self.timeit(np.argsort, vec)
            time_fast = self.timeit(fast_argsort, vec, 10)

            # fast_argsort is at least 5 times faster than np.argsort
            successes.append(5 * time_fast < time_slow)

        assert sum(successes) >= min_success


class TestGetNodesEdges:
    @staticmethod
    def assert_nodes_equal(nodes0, nodes1):
        assert len(nodes0) == len(nodes1)

        first = itemgetter(0)
        nodes0 = sorted(list(nodes0.items()), key=first)
        nodes1 = sorted(list(nodes1.items()), key=first)
        for (k0, v0), (k1, v1) in zip(nodes0, nodes1):
            assert k0 == k1
            assert v0 == v1

    @pytest.fixture
    def get_nodes_edges(self):
        from dstoolbox.utils import get_nodes_edges
        return get_nodes_edges

    def test_not_pipeline_or_feature_union(self, get_nodes_edges):
        with pytest.raises(TypeError) as exc:
            get_nodes_edges('a name', FunctionTransformer(validate=True))

        assert str(exc.value) == (
            "Need a (sklearn) Pipeline or FeatureUnion as input.")

    def test_case_empty_pipeline(self, get_nodes_edges):
        pipe = Pipeline([('bar', None)])
        nodes, edges = get_nodes_edges('empty pipe', pipe)

        expected_nodes = {'empty pipe': pipe}
        self.assert_nodes_equal(nodes, expected_nodes)
        assert edges == [('empty pipe', 'bar')]

    def test_case_empty_feature_union(self, get_nodes_edges):
        union = FeatureUnion([('bar', 'drop')])
        nodes, edges = get_nodes_edges('empty union', union)

        expected_nodes = {'empty union': union}
        self.assert_nodes_equal(nodes, expected_nodes)
        assert edges == [('empty union', 'bar')]

    def test_case_simple_pipeline(self, get_nodes_edges):
        pipe = Pipeline([
            ('bar', FunctionTransformer(validate=True)),
            ('baz', FunctionTransformer(validate=True)),
        ])
        nodes, edges = get_nodes_edges('my_pipe', pipe)

        expected_nodes = {
            'my_pipe': pipe,
            'bar': pipe.steps[0][1],
            'baz': pipe.steps[1][1],
        }
        expected_edges = sorted([('my_pipe', 'bar'), ('bar', 'baz')])

        self.assert_nodes_equal(nodes, expected_nodes)
        assert sorted(edges) == expected_edges

    def test_case_simple_feature_union(self, get_nodes_edges):
        union = FeatureUnion([
            ('bar', FunctionTransformer(validate=True)),
            ('baz', FunctionTransformer(validate=True)),
        ])
        nodes, edges = get_nodes_edges('my_union', union)

        expected_nodes = {
            'my_union': union,
            'bar': union.transformer_list[0][1],
            'baz': union.transformer_list[1][1],
        }
        expected_edges = sorted([('my_union', 'bar'), ('my_union', 'baz')])

        self.assert_nodes_equal(nodes, expected_nodes)
        assert sorted(edges) == expected_edges

    def test_case_many_nested_pipelines(self, get_nodes_edges):
        pipe = Pipeline([
            ('bar', Pipeline([
                ('baz', Pipeline([
                    ('spam', Pipeline([
                        ('baz', FunctionTransformer(validate=True)),
                    ])),
                ])),
            ])),
        ])
        nodes, edges = get_nodes_edges('my_pipe', pipe)

        expected_nodes = {
            'my_pipe': pipe,
            'bar': pipe.steps[0][1],
            'bar__baz': pipe.steps[0][1].steps[0][1],
            'bar__baz__spam': pipe.steps[0][1].steps[0][1].steps[0][1],
            'bar__baz__spam__baz': (
                pipe.steps[0][1].steps[0][1].steps[0][1].steps[0][1]),
        }
        expected_edges = sorted([
            ('my_pipe', 'bar'),
            ('bar', 'bar__baz'),
            ('bar__baz', 'bar__baz__spam'),
            ('bar__baz__spam', 'bar__baz__spam__baz')
        ])

        self.assert_nodes_equal(nodes, expected_nodes)
        assert sorted(edges) == expected_edges

    def test_case_nested_pipeline_feature_union(self, get_nodes_edges):
        pipe = Pipeline([
            ('bar', FunctionTransformer(validate=True)),
            ('baz', FeatureUnion([
                ('bar', FunctionTransformer(validate=True)),
                ('baz', FunctionTransformer(validate=True)),
            ])),
            ('spam', FunctionTransformer(validate=True)),
        ])
        nodes, edges = get_nodes_edges('my_pipe', pipe)

        expected_nodes = {
            'my_pipe': pipe,
            'bar': pipe.steps[0][1],
            'baz': pipe.steps[1][1],
            'baz__bar': pipe.steps[1][1].transformer_list[0][1],
            'baz__baz': pipe.steps[1][1].transformer_list[1][1],
            'spam': pipe.steps[2][1],
        }
        expected_edges = sorted([
            ('my_pipe', 'bar'),
            ('bar', 'baz'),
            ('baz', 'baz__bar'),
            ('baz', 'baz__baz'),
            ('baz__bar', 'spam'),
            ('baz__baz', 'spam'),
        ])

        self.assert_nodes_equal(nodes, expected_nodes)
        assert sorted(edges) == expected_edges

    def test_case_multiple_nested_pipelines_and_feature_unions(
            self, get_nodes_edges):
        pipe = Pipeline([
            ('1', FunctionTransformer(validate=True)),
            ('2', FunctionTransformer(validate=True)),
            ('3', FeatureUnion([
                ('1', FunctionTransformer(validate=True)),
                ('2', Pipeline([
                    ('30', FunctionTransformer(validate=True)),
                    ('20', FeatureUnion([
                        ('p', FeatureUnion([
                            ('p0', FunctionTransformer(validate=True)),
                            ('p1', FunctionTransformer(validate=True)),
                        ])),
                        ('q', FeatureUnion([
                            ('q0', FunctionTransformer(validate=True)),
                            ('q1', FunctionTransformer(validate=True)),
                        ])),
                    ])),
                    ('10', FunctionTransformer(validate=True)),
                ])),
                ('3', FeatureUnion([
                    ('f0', FunctionTransformer(validate=True)),
                    ('f1', FunctionTransformer(validate=True)),
                ])),
            ])),
            ('4', FunctionTransformer(validate=True)),
            ('5', FeatureUnion([
                ('100', FunctionTransformer(validate=True)),
                ('200', FunctionTransformer(validate=True)),
                ('300', FunctionTransformer(validate=True)),
            ])),
            ('06', FunctionTransformer(validate=True)),
        ])
        nodes, edges = get_nodes_edges('my_pipe', pipe)

        expected_nodes = {
            'my_pipe': pipe,
            '1': pipe.steps[0][1],
            '2': pipe.steps[1][1],
            '3': pipe.steps[2][1],
            '3__1': pipe.steps[2][1].transformer_list[0][1],
            '3__2': pipe.steps[2][1].transformer_list[1][1],
            '3__2__30': pipe.steps[2][1].transformer_list[1][1].steps[0][1],
            '3__2__20': pipe.steps[2][1].transformer_list[1][1].steps[1][1],

            '3__2__20__p': (pipe.steps[2][1].transformer_list[1][1].steps[1][1]
                            .transformer_list[0][1]),
            '3__2__20__p__p0': (pipe.steps[2][1].transformer_list[1][1]
                                .steps[1][1].transformer_list[0][1]
                                .transformer_list[0][1]),
            '3__2__20__p__p1': (pipe.steps[2][1].transformer_list[1][1]
                                .steps[1][1].transformer_list[0][1]
                                .transformer_list[1][1]),
            '3__2__20__q': (pipe.steps[2][1].transformer_list[1][1].steps[1][1]
                            .transformer_list[1][1]),
            '3__2__20__q__q0': (pipe.steps[2][1].transformer_list[1][1]
                                .steps[1][1].transformer_list[1][1]
                                .transformer_list[0][1]),
            '3__2__20__q__q1': (pipe.steps[2][1].transformer_list[1][1]
                                .steps[1][1].transformer_list[1][1]
                                .transformer_list[1][1]),
            '3__2__10': pipe.steps[2][1].transformer_list[1][1].steps[2][1],
            '3__3': pipe.steps[2][1].transformer_list[2][1],
            '3__3__f0': (pipe.steps[2][1].transformer_list[2][1]
                         .transformer_list[0][1]),
            '3__3__f1': (pipe.steps[2][1].transformer_list[2][1]
                         .transformer_list[1][1]),
            '4': pipe.steps[3][1],
            '5': pipe.steps[4][1],
            '5__100': pipe.steps[4][1].transformer_list[0][1],
            '5__200': pipe.steps[4][1].transformer_list[1][1],
            '5__300': pipe.steps[4][1].transformer_list[2][1],
            '06': pipe.steps[5][1],
        }

        expected_edges = sorted([
            ('my_pipe', '1'),
            ('1', '2'),
            ('2', '3'),
            ('3', '3__1'),
            ('3', '3__2'),
            ('3__2', '3__2__30'),
            ('3__2__30', '3__2__20'),
            ('3__2__20', '3__2__20__p'),
            ('3__2__20__p', '3__2__20__p__p0'),
            ('3__2__20__p', '3__2__20__p__p1'),
            ('3__2__20', '3__2__20__q'),
            ('3__2__20__q', '3__2__20__q__q0'),
            ('3__2__20__q', '3__2__20__q__q1'),
            ('3__2__20__p__p0', '3__2__10'),
            ('3__2__20__p__p1', '3__2__10'),
            ('3__2__20__q__q0', '3__2__10'),
            ('3__2__20__q__q1', '3__2__10'),
            ('3', '3__3'),
            ('3__3', '3__3__f0'),
            ('3__3', '3__3__f1'),
            ('3__1', '4'),
            ('3__2__10', '4'),
            ('3__3__f0', '4'),
            ('3__3__f1', '4'),
            ('4', '5'),
            ('5', '5__100'),
            ('5', '5__200'),
            ('5', '5__300'),
            ('5__100', '06'),
            ('5__200', '06'),
            ('5__300', '06'),
        ])

        self.assert_nodes_equal(nodes, expected_nodes)
        assert sorted(edges) == expected_edges
