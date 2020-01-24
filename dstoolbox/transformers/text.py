"""Collection of transformers dealing with text."""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


_cv_doc = CountVectorizer.__doc__
_idx = _cv_doc.find('Parameters')
_doc_same_part = _cv_doc[_idx:]
_text_featurizer_doc = """    Convert a collection of text documents to
    an array of list of indices.

    This transformer supports the same parameters as sklearn's
    `CountVectorizer` and `TfidfVectorizer`. That means you can decide
    between word and character n-grams, set the maximum vocabulary
    size etc. The only difference is that instead of creating a sparse
    array of bag-of-word features, it creates an array of lists of
    indices, with each index coding for a word/charcter n-gram.

    Use this in conjunction with `dstoolbox.transformers.Padder2d` to
    create a homogeneous 2-dim array of indices, which you can then
    feed, for instance, to an RNN.

    Note: Don't set the parameter `binary=True`.

    Additional parameter
    --------------------
    unknown_token : str or None (default=None)
      By default, words that are out-of-vocabulary are dropped. To
      avoid this, set this value to a string (e.g. '<UNK>'). Now,
      out-of-vocabulary words are replaced by an additional
      index.
      Note: This extra-token is not considered when determining
      `max_features`, meaning that if you set both parameters, your
      vocabulary size will effectively be `max_features + 1`.

    """ + _doc_same_part

class TextFeaturizer(CountVectorizer):  # pylint: disable=missing-docstring
    __doc__ = _text_featurizer_doc

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            input='content',  # pylint: disable=redefined-builtin
            encoding='utf-8',
            decode_error='strict',
            strip_accents=None,
            lowercase=True,
            preprocessor=None,
            tokenizer=None,
            stop_words=None,
            token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1),
            analyzer='word',
            max_df=1.0,
            min_df=1,
            max_features=None,
            vocabulary=None,
            binary=False,
            dtype=np.int64,
            unknown_token=None,
    ):
        super().__init__(
            input=input,
            encoding=encoding,
            decode_error=decode_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            analyzer=analyzer,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype,
        )
        if self.binary:
            raise ValueError("binary=True does not work with TextFeaturizer.")
        if (unknown_token is not None) and not isinstance(unknown_token, str):
            raise TypeError("unknown_token must be None or a str, not of type "
                            "{}.".format(type(unknown_token)))
        self.unknown_token = unknown_token

    def fit_transform(self, raw_documents, y=None):
        """Learn the vocabulary dictionary and return the transformed text.

        Parameters
        ----------
        raw_documents : iterable
          An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples,]
          An array of lists of ints.

        """
        CountVectorizer.fit_transform(self, raw_documents)
        if self.unknown_token is not None:
            self.vocabulary_[self.unknown_token] = len(self.vocabulary_)
        return self.transform(raw_documents)

    def transform(self, raw_documents):
        """Transform documents to array of list of int.

        Parameters
        ----------
        raw_documents : iterable
          An iterable which yields either str, unicode or file objects.

        Returns
        -------
        X : array, [n_samples,]
          An array of lists of ints.

        """
        # This method is solely overridden for the docstring.
        # pylint: disable=useless-super-delegation
        return super().transform(raw_documents)

    def _count_vocab(self, raw_documents, fixed_vocab):
        if not fixed_vocab:
            return super()._count_vocab(raw_documents, fixed_vocab)

        analyze = self.build_analyzer()
        vocabulary = self.vocabulary_

        X = []
        for doc in raw_documents:
            row = []
            for feature in analyze(doc):
                try:
                    feature_idx = vocabulary[feature]
                    row.append(feature_idx)
                except KeyError:
                    if self.unknown_token is None:
                        # Ignore out-of-vocabulary items for fixed_vocab=True
                        continue
                    row.append(vocabulary[self.unknown_token])
            X.append(row)
        return vocabulary, np.asarray(X)
