import pytest

from locomo_eval.scoring import (
    ADVERSARIAL,
    MULTI_HOP,
    OPEN_DOMAIN,
    SINGLE_HOP,
    TEMPORAL,
    answer_tokens,
    is_abstention,
    multi_answer_f1,
    normalize_answer,
    score_answer,
    token_f1,
)


def test_normalize_answer():
    assert normalize_answer("The Beach, and a  Lake!") == "beach lake"
    assert normalize_answer("An apple's core") == "apples core"
    # Only whole words are articles.
    assert normalize_answer("Theater and Anna") == "theater anna"
    assert normalize_answer("") == ""


def test_answer_tokens_are_stemmed():
    assert answer_tokens("Running the Studies") == ["run", "studi"]
    assert answer_tokens("painted, traveled") == ["paint", "travel"]


def test_token_f1():
    assert token_f1("beach", "beach") == 1.0
    assert token_f1("the beach", "Beach.") == 1.0
    assert token_f1("mountains", "beach") == 0.0
    assert token_f1("", "beach") == 0.0
    # precision 1/2, recall 1/1
    assert token_f1("sandy beach", "beach") == pytest.approx(2 / 3)
    # Repeated tokens only match as often as they occur in both.
    assert token_f1("beach beach", "beach") == pytest.approx(2 / 3)


def test_multi_answer_f1_splits_on_commas():
    # Each gold part takes its best match; the parts are averaged.
    assert multi_answer_f1("camping, pottery", "pottery, camping") == 1.0
    assert multi_answer_f1("pottery", "pottery, camping") == 0.5
    # A predicted part can serve several gold parts.
    assert multi_answer_f1("pottery and camping", "pottery, camping") == pytest.approx(2 / 3)


def test_is_abstention():
    assert is_abstention("That is NOT MENTIONED in the conversation.")
    assert is_abstention("No information available about that.")
    assert not is_abstention("She went hiking.")
    assert not is_abstention("I don't know")


def test_score_answer_dispatches_by_category():
    assert score_answer("pottery", "pottery, camping", MULTI_HOP) == 0.5
    assert score_answer("pottery", "pottery, camping", SINGLE_HOP) == pytest.approx(2 / 3)
    # Open-domain scores only the gold text before the first ";".
    assert score_answer("Liberal", "Liberal; because of her views", OPEN_DOMAIN) == 1.0
    assert score_answer("not mentioned", None, ADVERSARIAL) == 1.0
    assert score_answer("She went hiking", "hiking", ADVERSARIAL) == 0.0


def test_score_answer_rejects_bad_input():
    with pytest.raises(ValueError):
        score_answer("x", None, SINGLE_HOP)
    with pytest.raises(ValueError):
        score_answer("x", "x", 9)


# Expected scores computed by LoCoMo's task_eval/evaluation.py
# (eval_question_answering with NLTK's PorterStemmer).
@pytest.mark.parametrize(
    ("prediction", "gold", "category", "expected"),
    [
        ("7 May 2023", "7 May 2023", TEMPORAL, 1.0),
        ("On May 7th, 2023", "7 May 2023", TEMPORAL, 0.5714285714285715),
        ("She went to the beach", "beach", SINGLE_HOP, 0.4),
        ("Researching adoption agencies", "Adoption agencies", SINGLE_HOP, 0.8),
        ("Transgender woman", "transgender woman", SINGLE_HOP, 1.0),
        ("", "anything", SINGLE_HOP, 0.0),
        ("pottery, camping, painting", "Pottery, camping, painting, swimming", MULTI_HOP, 0.75),
        ("She paints and runs", "painting, running", MULTI_HOP, 0.5),
        ("Likely yes; she enjoys reading", "Likely yes; she likes reading", OPEN_DOMAIN, 0.5714285714285715),
        ("The answer is not mentioned in the conversation.", None, ADVERSARIAL, 1.0),
        ("No information available", None, ADVERSARIAL, 1.0),
        ("She went hiking", None, ADVERSARIAL, 0.0),
    ],
)
def test_matches_locomo_reference(prediction, gold, category, expected):
    assert score_answer(prediction, gold, category) == pytest.approx(expected)
