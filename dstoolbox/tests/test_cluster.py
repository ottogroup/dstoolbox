"""Tests for cluster.py"""

from functools import partial
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.metrics import adjusted_mutual_info_score


class TestHierarchicalClustering:
    @pytest.fixture
    def clustering(self):
        from dstoolbox.cluster import hierarchical_clustering
        return partial(
            hierarchical_clustering,
            criterion='distance',
            method='complete',
            metric='cosine',
        )

    @pytest.fixture
    def data(self):
        return np.array([
            [0, 0, 1.0],
            [0, 0, 0.9],
            [0, -1.0, 0],
            [0, -0.9, 0],
            [0, -0.5, 0.6],
            [1.0, 0, 0],
        ])

    @pytest.mark.parametrize('max_dist, expected', [
        (-0.1, [0, 1, 2, 3, 4, 5]),
        (0.2, [0, 0, 1, 1, 2, 3]),
        (0.7, [0, 0, 1, 1, 0, 2]),
        (1.1, [0, 0, 0, 0, 0, 0]),
    ])
    def test_functional(self, clustering, data, max_dist, expected):
        y_pred = clustering(data, max_dist=max_dist)
        assert adjusted_mutual_info_score(y_pred, expected) == 1

    def test_array_empty(self, clustering):
        result = clustering(np.zeros((0, 10)))
        assert (result == np.array([])).all()

    def test_only_1_sample(self, clustering):
        result = clustering(np.zeros((1, 10)))
        assert (result == np.array([0])).all()

    @pytest.yield_fixture
    def patched_clustering_cls_and_mocks(self):
        with patch('dstoolbox.cluster.linkage') as lk:
            with patch('dstoolbox.cluster.fcluster') as fc:
                from dstoolbox.cluster import HierarchicalClustering
                lk.return_value = 123
                fc.side_effect = ['1st_result', '2nd_result']
                yield HierarchicalClustering, lk, fc

    def test_linkage_tree_call_default(
            self, patched_clustering_cls_and_mocks):
        hc_cls, lk, fc = patched_clustering_cls_and_mocks
        X = np.zeros((2, 2))
        hc_cls().fit(X)

        assert (lk.call_args_list[0][0][0] == X).all()
        assert lk.call_args_list[0][1] == {'method': 'single',
                                           'metric': 'euclidean'}
        assert fc.call_args_list[0][0][0] == 123
        assert fc.call_args_list[0][1] == {
            't': 0.5, 'criterion': 'inconsistent'}

    def test_linkage_tree_call_non_default(
            self, patched_clustering_cls_and_mocks):
        hc_cls, lk, fc = patched_clustering_cls_and_mocks
        X = np.zeros((2, 2))
        hc_cls(
            max_dist=0.111,
            criterion='crit',
            method='meth',
            metric='metr',
        ).fit(X)

        assert (lk.call_args_list[0][0][0] == X).all()
        assert lk.call_args_list[0][1] == {'method': 'meth',
                                           'metric': 'metr'}
        assert fc.call_args_list[0][0][0] == 123
        assert fc.call_args_list[0][1] == {'t': 0.111, 'criterion': 'crit'}

    def test_repeated_fit_predict(self, patched_clustering_cls_and_mocks):
        model = patched_clustering_cls_and_mocks[0]()
        X = np.random.random((100, 5))

        result = model.fit_predict(X)
        assert result == '1st_result'
        assert model.labels_ == '1st_result'

        result = model.fit_predict(X)
        assert result == '2nd_result'
        assert model.labels_ == '2nd_result'
