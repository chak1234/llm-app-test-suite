"""Consistency: the wording may change between runs, the facts may not.

The two repeat tests are marked 'live'. Replaying one recorded response five
times would always agree with itself, so they only mean something against a
real model.
"""
from itertools import combinations

import pytest

from app.chatbot import ask
from tests.helpers.facts import has_fact, is_refusal, load_golden
from tests.helpers.similarity import cosine_similarity

pytestmark = pytest.mark.consistency

RUNS = 5
# Tuned from real runs: 20 repeats of the question below against llama3.2:3b
# (temperature 0, seed 0) gave one identical answer, so every pair scored 1.00.
# 0.95 sits just below that. Retune before pointing the suite at a model that
# varies more: correct rewordings of the same fact score as low as 0.54.
SIMILARITY_THRESHOLD = 0.95
PARAPHRASES = load_golden("paraphrase")


@pytest.mark.live
def test_same_question_gives_same_facts():
    answers = [ask("How long do I have to return an item?").text for _ in range(RUNS)]
    for text in answers:
        assert has_fact(text, "30 day"), f"Missing fact in: {text}"
    for a, b in combinations(answers, 2):
        score = cosine_similarity(a, b)
        assert score >= SIMILARITY_THRESHOLD, f"Similarity {score:.2f} between:\n{a}\n---\n{b}"


@pytest.mark.parametrize("case", PARAPHRASES, ids=lambda c: c["id"])
def test_paraphrases_give_same_fact(case):
    answer = ask(case["question"])
    for fact in case["expected_facts"]:
        assert has_fact(answer.text, fact), f"Missing fact {fact!r} in: {answer.text}"


@pytest.mark.live
def test_refusal_does_not_flip():
    answers = [ask("Do you ship to Antarctica?").text for _ in range(RUNS)]
    refused = [is_refusal(text) for text in answers]
    assert all(refused), f"{refused.count(False)} of {RUNS} answers did not refuse: {answers}"
