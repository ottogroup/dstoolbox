"""Tests for transformers.padding.py."""

import numpy as np
import pytest


class TestPadder2d:
    @pytest.fixture
    def padder_cls(self):
        from dstoolbox.transformers import Padder2d
        return Padder2d

    @pytest.fixture
    def padder(self, padder_cls):
        return padder_cls(max_len=4, pad_value=55, dtype=np.int64)

    @pytest.fixture
    def data(self):
        return [
            [],
            [0, 1, 2],
            [10, 11, 12, 13, 14, 15],
            [100, 101, 102, 103],
        ]

    @pytest.fixture
    def expected(self):
        return np.asarray([
            [55, 55, 55, 55],
            [0, 1, 2, 55],
            [10, 11, 12, 13],
            [100, 101, 102, 103]
        ])

    def test_fit_and_transform_works(self, padder, data, expected):
        result = padder.fit(data).transform(data)
        assert np.allclose(result, expected)
        assert result.dtype == np.int64
        assert isinstance(result, np.ndarray)

    def test_fit_transform_works(self, padder, data, expected):
        result = padder.fit_transform(data)
        assert np.allclose(result, expected)
        assert result.dtype == np.int64
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize('max_len', [1, 2, 3])
    def test_other_max_len(self, padder_cls, data, expected, max_len):
        padder = padder_cls(max_len=max_len, pad_value=55, dtype=np.int64)
        result = padder.fit_transform(data)
        assert np.allclose(result, expected[:, :max_len])
        assert result.dtype == np.int64

    def test_other_pad_value(self, padder_cls, data, expected):
        padder = padder_cls(max_len=4, pad_value=-999, dtype=np.int64)
        result = padder.fit_transform(data)
        expected[expected == 55] = -999
        assert np.allclose(result, expected)
        assert result.dtype == np.int64

    def test_other_dtype(self, padder_cls, data, expected):
        padder = padder_cls(max_len=4, pad_value=55, dtype=np.float16)
        result = padder.fit_transform(data)
        assert np.allclose(result, expected)
        assert result.dtype == np.float16


class TestPadder3d:
    @pytest.fixture
    def padder_cls(self):
        from dstoolbox.transformers import Padder3d
        return Padder3d

    @pytest.fixture
    def padder(self, padder_cls):
        return padder_cls(max_size=(4, 2), pad_value=55, dtype=np.int64)

    @pytest.fixture
    def data(self):
        return [
            [],
            [[0, 0], [1, 1, 1], [2]],
            [[10], [], [12, 12, 12], [13], [], [15]],
            [[100], [101], [102], [103, 104, 105]],
        ]

    @pytest.fixture
    def expected(self):
        return np.asarray([
            [[55, 55], [55, 55], [55, 55], [55, 55]],
            [[0, 0], [1, 1], [2, 55], [55, 55]],
            [[10, 55], [55, 55], [12, 12], [13, 55]],
            [[100, 55], [101, 55], [102, 55], [103, 104]],
        ])

    def test_fit_and_transform_works(self, padder, data, expected):
        result = padder.fit(data).transform(data)
        assert np.allclose(result, expected)
        assert result.dtype == np.int64
        assert isinstance(result, np.ndarray)

    def test_fit_transform_works(self, padder, data, expected):
        result = padder.fit_transform(data)
        assert np.allclose(result, expected)
        assert result.dtype == np.int64
        assert isinstance(result, np.ndarray)

    @pytest.mark.parametrize('max_size_0', [1, 2, 3])
    def test_max_size_0(self, padder_cls, data, expected, max_size_0):
        padder = padder_cls(
            max_size=(max_size_0, 2), pad_value=55, dtype=np.int64)
        result = padder.fit_transform(data)
        assert np.allclose(result, expected[:, :max_size_0])
        assert result.dtype == np.int64

    def test_max_size_1(self, padder_cls, data, expected):
        padder = padder_cls(
            max_size=(4, 1), pad_value=55, dtype=np.int64)
        result = padder.fit_transform(data)
        assert np.allclose(result, expected[:, :, :1])
        assert result.dtype == np.int64

    def test_other_pad_value(self, padder_cls, data, expected):
        padder = padder_cls(
            max_size=(4, 2), pad_value=-999, dtype=np.int64)
        result = padder.fit_transform(data)
        expected[expected == 55] = -999
        assert np.allclose(result, expected)
        assert result.dtype == np.int64

    def test_other_dtype(self, padder_cls, data, expected):
        padder = padder_cls(
            max_size=(4, 2), pad_value=55, dtype=np.float16)
        result = padder.fit_transform(data)
        assert np.allclose(result, expected)
        assert result.dtype == np.float16
