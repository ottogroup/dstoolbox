"""Collection of transformers dealing with text."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer

from dstoolbox.data import load_w2v_format


class W2VTransformer(BaseEstimator, TransformerMixin):
    """Replace words by their word2vec representation.

    For each word in a line, look up the word2vec representation of
    that word (if found in the vocabulary), then aggregate all
    representations.

    Unknown words will be skipped, if there is no known word in the
    string, a zero vector of correct size will be returned.

    Parameters
    ----------
    path_w2v : str
      Filename of w2v data exported with gensim (first row dimensons
      of data in file, each row space-separated word vectors with word
      as string at the beginning.)

    aggr_func : function (default: numpy.mean)
      Aggregation function for all word w2v representations in every
      string to be transformed. Example: aggr_func: numpy.mean: The
      input String "Hello there" will be transformed to the mean of
      the w2v representations of "Hello" and "there".

    analyze : factory function returning a callable
      Defaults to sklearn CountVectorizer().build_analyzer. This
      should return a function that tokenizes a string. It should
      accept strings or list of strings as input.  If None, use the
      default analyze function provided by
      CountVectorizer().build_analyzer().

    max_words : int (default=100)
      The maximum number of words to consider for each input string
      after tokenization.

    """
    def __init__(
            self,
            path_w2v,
            aggr_func=np.mean,
            analyze=CountVectorizer().build_analyzer,
            max_words=100,
    ):
        self.path_w2v = path_w2v
        self.aggr_func = aggr_func
        self.analyze = analyze
        self.max_words = max_words

    def _get_vector_from_words(self, words):
        """Transforms a list of strings to a numpy.array with the
        corresponding word2vec embedings in one vector using the
        aggregation method defined at initialization.

        Parameters
        ----------
        words: [str]
          A list of words

        Results
        -------
        numpy.array 2d
          A 1 x n array containing the aggregated word2vec embeddings.

        """
        syn0 = self.syn0_
        indices = [self.word2idx_.get(word) for word in words
                   if self.word2idx_.get(word) is not None]
        if indices:
            return self.aggr_func(syn0[indices], axis=0, keepdims=True)
        else:
            return np.zeros_like(syn0[0:1])

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def fit(self, X=None, y=None):
        """Imports the method and data needed to transform data.

        Parameters
        ----------
        X : None
          not used (needed for sklearn)

        y : None
          not used (needed for sklearn)

        Attributes
        ----------
        word2idx_, dict{str: int}
          A dictionary to look up the row number of the
          representation vector for a given word.

        syn0_: numpy.array 2d
          Contains the vector represantation for words in every row. The
          row number for a given word can be found with `word2idx`.

        Results
        -------
        self : W2VTransformer
          The object itself.

        """
        self.word2idx_, self.syn0_ = load_w2v_format(self.path_w2v)
        return self

    def transform(self, X):
        """Transforms input samples to word2vec representation.

        Every row in each sample of X will be tranformed
        separatly. Unknown words will be skipped, if there is no known
        word in the string, a zero vector of correct size will be
        returned.

        Parameters
        ----------
        X : numpy.array 1D
          An array containing the strings in the rows.

        Results
        -------
        np.array 2d
          Numpy array containing the aggregated word2vec
          representations.

        """
        Xt = []
        for x in X:
            words = self.analyze()(x)[:self.max_words]
            vector = self._get_vector_from_words(words)
            Xt.append(vector)
        return np.vstack(Xt)
