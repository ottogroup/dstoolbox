"""This module contains useful functions working with textual data.

These functions are often necessary because there is a lack of
alternatives for untidy and/or German text.

"""

import re


# =================== Regex matching using =========================


# Using constants so that regex pattern can be compiled once instead
# of during each function call.

exceptions = [
    r"\s[1-9]\.\s",  # 1., 2., etc.
    r"[\s\?\!]\)",  # end of sentence within parens
    r"\su\.",  # u.
    r"[0-9]\.[0-9]",  # e.g. 2.0
]


regex_abbrevs = [
    r"Abb\.",
    r"Bsp\.",
    r"bzgl\.",
    r"ca\.",
    r"d\.h\.",
    r"d\.\sh\.",
    r"def\.",
    r"div\.",
    r"dt\.",
    r"eig\.",
    r"evtl\.",
    r"Evtl\.",
    r"evt\.",
    r"Fa\.",
    r"gem\.",
    r"ggf\.",
    r"Gr\.",
    r"gr\.",
    r"I\.O\.",
    r"incl\.",
    r"insg\.",
    r"lt\.",
    r"Lt\.",
    r"kl\.",
    r"m\.E\.",
    r"m\.M\.",
    r"m\.M\.n\.",
    r"max\.",
    r"Min\.",
    r"min\.",
    r"od\.",
    r"o\.ä\.",
    r"o\.Ä\.",
    r"o\.g\.",
    r"o\.k.",
    r"o\.\sk\.",
    r"rel\.",
    r"s\.o\.",
    r"seitl\.",
    r"Seitl\.",
    r"Stk\.",
    r"Stck\.",
    r"Std\.",
    r"std\.",
    r"tel\.",
    r"tlg\.",
    r"u\.a\.",
    r"u\.ä\.",
    r"u\.U\.",
    r"usw\.",
    r"v\.a\.",
    r"vllt\.",
    r"vll\.",
    r"vlt\.",
    r"wg\.",
    r"zw\.",
    r"z\.B\.",
    r"Z\.B\.",
    r"z\.b\.",
    r"zB\.",
    r"z\.T\.",
    r"z\.Zt\.",
    r"Zt\.",
    r"z\.Z\.",
]

for abbrev in regex_abbrevs:
    exceptions.append(abbrev)


# group name, see use of 'EOS' below, not necessarily the same as
# EOS_TAG
matches_ = r"(?P<EOS>"
# punctuations (including multiple) and newline
matches_ += r"[\.\?\!]|\n"
# smileys
matches_ += r"|\:\)|\:\(|\:\-\)|\:\-\("
matches_ += r")"
matches = [matches_]


PATTERN_RE = re.compile(r"|".join(exceptions + matches))

EOS_TAG = '<EOS>'


def sentenize(sentence):
    """Tokenize sentences using regular expression approach.

    Uses pre-defined compiled regex (PATTERN_RE).

    Parameters
    ----------
    sentence : str
      The sentences that should be tokenized into individual
      sentences.

    Returns
    -------
    result : list of str
      A list of individual sentences.

    """
    def replace(match):
        if match.group('EOS'):
            return match.group('EOS').strip() + EOS_TAG
        else:
            return match.group()

    replaced = PATTERN_RE.sub(replace, sentence)
    splits = replaced.split(EOS_TAG)
    result = [split.lstrip(' .!?') for split in splits
              if len(split.lstrip(' .!?')) > 1]
    return result


def normalize_word(word):
    """Normalize a word

    Basically strips it from special tokens and casts it to lower
    case. Does not remove Umlauts.

    Parameters
    ----------
    word : str
      The word that should be normalized.

    Returns
    -------
    result : str
      The normalized word.

    """
    result = word.lower().strip().strip('.,?!-;:()\n"')
    return result
