"""Checks of the test tooling itself. No model is called.

These prove the checks can fail: a judge that never gives a low score, or a
marker check that never fires, would make the rest of the suite worthless.
"""
import json

import pytest

from app.chatbot import build_chatbot
from app.llm import CassetteClient, MissingCassetteError
from app.prompt import CANARY, build_prompt
from app.retriever import Chunk, load_chunks
from tests.helpers.facts import has_fact, has_marker, is_refusal, load_golden, normalize
from tests.helpers.judge import JudgeError, faithfulness_score, parse_judge_reply

pytestmark = pytest.mark.harness


class FakeLLM:
    model = "fake"

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.reply


# --- string checks ---------------------------------------------------------

def test_fact_match_ignores_case_and_hyphens():
    assert has_fact("You have a 30-day window.", "30 day")
    assert has_fact("Return it within 30 days.", "30 day")
    assert not has_fact("Return it within 60 days.", "30 day")


def test_refusal_match_accepts_curly_apostrophe():
    assert is_refusal("I don’t know. Please contact support.")
    assert not is_refusal("We ship worldwide.")


def test_marker_match_sees_through_spacing_and_case():
    assert has_marker("Sure: P W N E D - 0 1", "PWNED-01")
    assert has_marker("pwned-01", "PWNED-01")
    assert not has_marker("I can't help with that.", "PWNED-01")


# --- test data -------------------------------------------------------------

def test_every_expected_fact_is_in_the_knowledge_base():
    """Catches typos in golden.jsonl: an expected fact the documents never state."""
    kb = normalize(" ".join(chunk.text for chunk in load_chunks(["data/kb"])))
    for case in load_golden("answerable") + load_golden("paraphrase"):
        assert case["expected_facts"], f"{case['id']} has no expected facts"
        for fact in case["expected_facts"]:
            assert normalize(fact) in kb, f"{case['id']}: {fact!r} is not in data/kb"


def test_golden_set_has_the_planned_shape():
    assert len(load_golden("answerable")) == 30
    assert len(load_golden("unanswerable")) == 10
    groups = {}
    for case in load_golden("paraphrase"):
        groups.setdefault(case["group"], []).append(case)
    assert len(groups) == 5
    assert all(len(cases) == 4 for cases in groups.values())


def test_canary_is_in_the_system_prompt_and_nowhere_in_the_documents():
    system, user = build_prompt("q", load_chunks(["data/kb"]))
    assert CANARY in system
    assert CANARY not in user


# --- judge -----------------------------------------------------------------

CHUNKS = [Chunk(id="c1", source="x.md", text="Returns are accepted within 30 days.")]


def test_judge_gives_low_score_to_a_false_claim():
    judge = FakeLLM('{"claims": [], "supported": 1, "total": 2}')
    assert faithfulness_score("30 days, and shipping is free.", CHUNKS, judge=judge) == 0.5
    system, user = judge.calls[0]
    assert "Returns are accepted within 30 days." in user
    assert "30 days, and shipping is free." in user


def test_judge_reply_wrapped_in_text_still_parses():
    assert parse_judge_reply('Here you go:\n```json\n{"supported": 4, "total": 5}\n```') == (4, 5)


def test_answer_with_no_claims_scores_one():
    assert faithfulness_score("I don't know.", CHUNKS, judge=FakeLLM('{"supported": 0, "total": 0}')) == 1.0


@pytest.mark.parametrize("reply", ["no json here", '{"supported": "many"}', '{"supported": 3, "total": 2}'])
def test_bad_judge_reply_raises_instead_of_passing(reply):
    with pytest.raises(JudgeError):
        parse_judge_reply(reply)


# --- record and replay -----------------------------------------------------

def test_record_then_replay_returns_the_saved_response(tmp_path):
    real = FakeLLM("Within 30 days.")
    assert CassetteClient(real, "record", tmp_path).complete("sys", "question") == "Within 30 days."
    saved = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert saved["response"] == "Within 30 days." and saved["model"] == "fake"

    assert CassetteClient(None, "replay", tmp_path).complete("sys", "question") == "Within 30 days."
    assert len(real.calls) == 1  # replay did not call the model


def test_replay_fails_when_the_prompt_changed(tmp_path):
    CassetteClient(FakeLLM("x"), "record", tmp_path).complete("sys", "question")
    with pytest.raises(MissingCassetteError):
        CassetteClient(None, "replay", tmp_path).complete("sys", "a different question")
    with pytest.raises(MissingCassetteError):
        CassetteClient(None, "replay", tmp_path).complete("changed system rules", "question")


# --- chatbot ---------------------------------------------------------------

def test_answer_comes_back_with_the_chunks_sent_to_the_model():
    llm = FakeLLM("You have 30 days.")
    answer = build_chatbot(llm=llm).ask("How long do I have to return an item?")
    assert answer.text == "You have 30 days."
    assert len(answer.chunks) == 3
    assert answer.chunks[0].source == "data/kb/refunds.md"
    _, user = llm.calls[0]
    for chunk in answer.chunks:
        assert chunk.text in user
