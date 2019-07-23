"""Tests for load.py."""

from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
import pytest


class TestLoadW2VData:
    @pytest.fixture
    def mock_w2v_file(self):
        w2v = pd.DataFrame({
            0: ['first', 'second', 'third'],
            1: [0.9, -0.2, 0.5],
            2: [0.4, 0.3, 0.2],
            3: [0.3, -0.6, -0.5],
            4: [0.2, -0.3, 0],
        })
        return w2v

    @pytest.fixture
    def mock_w2v_header(self):
        header = pd.DataFrame({0: [3], 1: [4]})
        return header

    @pytest.fixture
    def load_w2v_format(self, mock_w2v_file, mock_w2v_header, monkeypatch):
        """Don't load from real file but instead use mocked
        DataFrame.

        """
        # pylint: disable=unused-argument
        def mock_read_csv(*args, **kwargs):
            # pandas.read_csv is called twice, once for the meta
            # information in the header with `nrows=1` and once for
            # vector information; this is reflected in this mocked
            # function.
            nrows = kwargs.get('nrows')
            if nrows == 1:
                return mock_w2v_header
            return mock_w2v_file

        import dstoolbox.data
        monkeypatch.setattr(
            dstoolbox.data.load.pd, 'read_csv',
            mock_read_csv,
        )
        return dstoolbox.data.load_w2v_format

    def test_load_w2v_format_word2idx(self, load_w2v_format):
        word2idx = load_w2v_format('some/path')[0]
        assert len(word2idx) == 3
        assert word2idx == {'first': 0, 'second': 1, 'third': 2}

    def test_load_w2v_format_syn0(self, load_w2v_format):
        syn0 = load_w2v_format('some/path')[1]
        assert syn0.shape == (3, 4)
        assert syn0[1, 1] == 0.3


class TestLoadW2VVocab:
    @pytest.fixture
    def mock_vocab(self):
        vocab = "damen 123\nherren 64\nadidas 33"
        return vocab

    def test_load_w2v_vocab(self, mock_vocab):
        # a somewhat complicated way to mock open -> iterate -> read
        gen = (line for line in mock_vocab.split('\n'))
        mock = Mock()
        mock.return_value.__enter__ = lambda s: s
        mock.return_value.__exit__ = Mock()
        mock.return_value.__iter__ = lambda __: gen
        with patch.dict('dstoolbox.data.load.__builtins__', {'open': mock}):
            from dstoolbox.data.load import load_w2v_vocab
            vocab = load_w2v_vocab('some/path')
            assert vocab == {'damen': 123, 'herren': 64, 'adidas': 33}
