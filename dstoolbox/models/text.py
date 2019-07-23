"""Collection of classifiers intended to work with text data."""

import numpy as np
from sklearn.base import BaseEstimator

from dstoolbox.data import load_w2v_format
from dstoolbox.utils import normalize_matrix
from dstoolbox.utils import fast_argsort


class W2VClassifier(BaseEstimator):
    """Word2Vec classifier that requires pre-trained word vectors.

    This classifier implements the `kneighbors` interface from
    scikit-learn so that it can be used similarly to a
    KNeighborsClassifier & Co. It also partly re-implements some of
    the `gensim.models.Word2Vec` method.

    Parameters
    ----------
    path_w2v : str
      Filename of w2v data exported with gensim (first row dimensons
      of data in file, each row space-separated word vectors with word
      as string at the beginning.)

    topn : int (default=10)
      The number of similar words to return by default

    """
    def __init__(self, path_w2v, topn=10):
        self.fname = path_w2v
        self.topn = topn

    # pylint: disable=attribute-defined-outside-init,unused-argument
    def fit(self, X=None, y=None):
        """Load word2vec data.

        Parameters
        ----------
        Requires no parameters.

        Attributes
        ----------
        word2idx_, dict{str: int}
          A dictionary to look up the row number of the
          representation vector for a given word.

        syn0_: numpy.array 2d
          Contains the vector represantation for words in every row. The
          row number for a given word can be found with `word2idx`.

        classes_: numpy.array 2d
          Holds the label for each class.

        """
        word2idx, syn0 = load_w2v_format(self.fname)
        self.word2idx_ = word2idx
        self.classes_ = np.array(sorted(word2idx, key=word2idx.get))
        self.syn0_ = normalize_matrix(syn0)
        return self

    def _get_vector_from_word(self, word):
        # return word2vec embedding for a given word
        return self.syn0_[self.word2idx_[word]]

    def _update_vocabulary(self, word, vector):
        # add a new word and embedding to existing ones
        self.word2idx_[word] = self.syn0_.shape[0]
        self.classes_ = np.array(self.classes_.tolist() + [word])
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)
        self.syn0_ = np.vstack((self.syn0_, vector))

    def _add_word_vectors(self, positive):
        # compute the normalized mean of several word embeddings
        vectors = [self._get_vector_from_word(word) for word in positive]
        vectors = np.mean(vectors, axis=0).reshape(1, -1)
        normalized = normalize_matrix(vectors)
        return normalized

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        """Finds the K-neighbors of a point.

        Returns indices of and distances to the neighbors of each
        point.

        Parameters
        ----------
        X: numpy.array 1D
          Array of strings containing the words whose neighbors should
          be determined.

        n_neighbors: int
          Number of neighbors to get (default is the value passed to
          the constructor).

        return_distance: boolean (default=True)
          If False, distances will not be returned.

        Returns
        -------
        neighbors: numpy.array
          Indices of the nearest points in the population matrix.

        distances: numpy.array (optional)
          Array representing the lengths to points, only present if
          `return_distance=True`.

        """
        n_neighbors = n_neighbors or self.topn
        n_neighbors = min(n_neighbors, self.syn0_.shape[0] - 1)
        neighbors, similarities = [], []

        for x in X:
            xvec = self._get_vector_from_word(x)
            similarity = np.dot(self.syn0_, xvec.T)

            # Throw away the smallest index, since it is the initial
            # word itself.
            # pylint: disable=invalid-unary-operand-type
            neighbor_indices = fast_argsort(-similarity, n_neighbors + 1)[1:]

            neighbors.append(neighbor_indices)
            similarities.append(similarity[neighbor_indices])

        neighbors = np.vstack(neighbors)
        if not return_distance:
            return neighbors

        # normalize distances to [0, 1]
        distances = (np.vstack(similarities) - 1) / -2.
        return neighbors, distances

    # pylint: disable=dangerous-default-value
    def most_similar(self, positive=[], negative=[], topn=10):
        """Find the top-N most similar words.

        (verbatim from gensim)
        This method computes cosine similarity between a simple mean
        of the projection weight vectors of the given words and the
        vectors for each word in the model.  The method corresponds to
        the `word-analogy` and `distance` scripts in the original
        word2vec implementation.

        Parameters
        ----------
        positive: str or list of str (default=[])
          Word(s) whose embedding(s) contribute(s) positively.

        negative: str or list of str (default=[])
          Word(s) whose embedding(s) contribute(s) negatively. It is
          currently NOT IMPLEMENTED.

        topn: int (default=10)
          Number of similar words to return.

        Returns
        -------
        results: list of tuples (str, float)
          The `topn` most similar words in a list with corresponding
          similarity measure.

        """
        if negative:
            raise NotImplementedError(
                "The `negative` parameter is not yet supported.")

        if not positive:
            raise ValueError("No words provided to compute similarity.")

        if isinstance(positive, str):
            positive = [positive]

        if len(positive) > 1:
            joined = ' '.join(positive)
            if joined not in self.word2idx_:
                # update vocabulary to contain composite word
                vector = self._add_word_vectors(positive)
                self._update_vocabulary(joined, vector)
            positive = [joined]

        neighbor_indices, distances = self.kneighbors(positive, topn)
        neighbors = self.classes_[neighbor_indices]
        similarities = 1.0 - distances
        results = list(zip(neighbors[0], similarities[0]))
        return results

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X: numpy.array 1d
          Array containing word for each sample.

        Returns
        -------
        y_pred: numpy.array 1d
          Array containing the class labels for each sample.

        """
        neighbors = self.kneighbors(X, n_neighbors=1, return_distance=False)
        y_pred = neighbors.flatten()
        return y_pred

    def predict_proba(self, X):
        """This method is not implemented."""
        raise NotImplementedError("`predict_proba` does not exist for this "
                                  "classifier.")
