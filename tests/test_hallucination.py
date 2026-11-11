"""Hallucination: any claim in the answer that the retrieved chunks do not support.

Three checks, ordered from cheapest to most expensive:
key facts (string match), refusal (string match), faithfulness (judge model).
"""
import pytest

from app.chatbot import ask
from app.retriever import Retriever
from tests.helpers.facts import has_fact, is_refusal, load_golden
from tests.helpers.judge import faithfulness_score

pytestmark = pytest.mark.hallucination

ANSWERABLE = load_golden("answerable")
UNANSWERABLE = load_golden("unanswerable")
FAITHFULNESS_THRESHOLD = 0.8

# Answers written by hand with a false claim in each. They prove the real
# judge can give a low score; a judge that passes everything is worthless.
FALSE_ANSWERS = [
    ("false-refund", "How long do I have to return an item?", "You have 90 days from the delivery date to return an item."),
    ("false-restocking", "Is there a restocking fee for opened items?", "Yes, there is a 25% restocking fee for opened items."),
    ("false-password", "What is the minimum password length?", "Passwords must be at least 8 characters long."),
    ("false-cancel", "How long do I have to cancel an order?", "You have 24 hours to cancel an order, and cancelling costs a $5 fee."),
]


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda c: c["id"])
def test_answer_has_key_facts(case):
    answer = ask(case["question"])
    for fact in case["expected_facts"]:
        assert has_fact(answer.text, fact), f"Missing fact {fact!r} in: {answer.text}"
    for wrong in case["must_not_contain"]:
        assert not has_fact(answer.text, wrong), f"Wrong fact {wrong!r} in: {answer.text}"


@pytest.mark.parametrize("case", ANSWERABLE, ids=lambda c: c["id"])
def test_answer_is_grounded_in_chunks(case):
    answer = ask(case["question"])
    score = faithfulness_score(answer.text, answer.chunks)
    assert score >= FAITHFULNESS_THRESHOLD, f"Unsupported claims (score {score:.2f}) in: {answer.text}"


@pytest.mark.parametrize("case", UNANSWERABLE, ids=lambda c: c["id"])
def test_refuses_when_docs_have_no_answer(case):
    answer = ask(case["question"])
    assert is_refusal(answer.text), f"Expected a refusal, got: {answer.text}"


@pytest.fixture(scope="module")
def retriever():
    return Retriever(["data/kb"])


@pytest.mark.parametrize("question, false_answer", [c[1:] for c in FALSE_ANSWERS], ids=[c[0] for c in FALSE_ANSWERS])
def test_judge_gives_low_score_to_false_answer(retriever, question, false_answer):
    score = faithfulness_score(false_answer, retriever.search(question))
    assert score < FAITHFULNESS_THRESHOLD, f"Judge accepted a false answer (score {score:.2f}): {false_answer}"
