"""Answer scoring by token F1, independent of any backend or LLM.

A port of LoCoMo's task_eval/evaluation.py (f1_score, f1 and the per-category
dispatch in eval_question_answering) so scores are comparable to published
results:

- Answers are normalized: commas dropped, lowercased, punctuation stripped,
  the words a/an/the/and removed, whitespace collapsed, and every token
  Porter-stemmed.
- Token F1 compares the bag of normalized tokens of prediction and gold.
- Multi-hop (1): both sides are split on commas; each gold part takes its best
  F1 against any predicted part, and the parts' scores are averaged.
- Temporal (2), single-hop (4): plain token F1.
- Open-domain (3): plain token F1 against the gold text before the first ";".
- Adversarial (5): 1 if the prediction abstains, else 0.

The reference stems with NLTK's PorterStemmer; this uses snowballstemmer's
original Porter algorithm, which needs no data download. NLTK's extensions
stem some words differently ("day" -> "day" vs "dai", "hopefully" -> "hope"
vs "hopefulli"), but since both sides are stemmed alike the F1 rarely moves:
scoring each locomo10 question and gold answer as predictions, one of ~6000
scores differed from the reference.
"""

import re
import string
from collections import Counter

import snowballstemmer

MULTI_HOP = 1
TEMPORAL = 2
OPEN_DOMAIN = 3
SINGLE_HOP = 4
ADVERSARIAL = 5

# The reference counts an adversarial answer as correct when it contains
# either phrase (case-insensitive).
ABSTENTION_PHRASES = ("no information available", "not mentioned")

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)
_stemmer = snowballstemmer.stemmer("porter")


def normalize_answer(text: str) -> str:
    """LoCoMo's normalize_answer: the text before stemming."""
    text = text.replace(",", "").lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


def answer_tokens(text: str) -> list[str]:
    """Normalized, stemmed tokens, as token F1 compares them."""
    return _stemmer.stemWords(normalize_answer(text).split())


def token_f1(prediction: str, gold: str) -> float:
    """Token-level F1 between one prediction and one gold answer."""
    predicted = answer_tokens(prediction)
    expected = answer_tokens(gold)
    same = sum((Counter(predicted) & Counter(expected)).values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(expected)
    return 2 * precision * recall / (precision + recall)


def multi_answer_f1(prediction: str, gold: str) -> float:
    """Mean over the gold's comma-separated parts of each part's best F1
    against any comma-separated part of the prediction."""
    predictions = [p.strip() for p in prediction.split(",")]
    golds = [g.strip() for g in gold.split(",")]
    return sum(max(token_f1(p, g) for p in predictions) for g in golds) / len(golds)


def is_abstention(prediction: str) -> bool:
    lowered = prediction.lower()
    return any(phrase in lowered for phrase in ABSTENTION_PHRASES)


def score_answer(prediction: str, gold: str | None, category: int) -> float:
    """Score one prediction the way LoCoMo scores its QA category.

    `gold` is ignored for adversarial questions, which have no gold answer."""
    if category == ADVERSARIAL:
        return 1.0 if is_abstention(prediction) else 0.0
    if gold is None:
        raise ValueError(f"category {category} needs a gold answer")
    if category == MULTI_HOP:
        return multi_answer_f1(prediction, gold)
    if category == OPEN_DOMAIN:
        return token_f1(prediction, gold.split(";")[0].strip())
    if category in (TEMPORAL, SINGLE_HOP):
        return token_f1(prediction, gold)
    raise ValueError(f"unknown LoCoMo category {category}")
