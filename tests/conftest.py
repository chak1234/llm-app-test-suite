"""Fixtures, marker handling, and result logging."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from app.chatbot import Chatbot
from app.llm import llm_mode
from tests.helpers import results

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "results.json"
KNOWN_ISSUES_FILE = ROOT / "data" / "known_issues.yaml"


def pytest_report_header(config):
    return f"LLM_MODE: {llm_mode()}"


def pytest_collection_modifyitems(config, items):
    """In replay mode: skip 'live' tests and expect the recorded known issues to fail."""
    if llm_mode() != "replay":
        return
    skip_live = pytest.mark.skip(reason="needs a real model; LLM_MODE is replay")
    known = {
        issue["test"]: issue["reason"]
        for issue in yaml.safe_load(KNOWN_ISSUES_FILE.read_text(encoding="utf-8")) or []
    }
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
        if item.name in known:
            # AssertionError only: a missing cassette is a broken setup, not a known issue.
            item.add_marker(
                pytest.mark.xfail(reason=f"Known issue: {known[item.name]}", raises=AssertionError, strict=True)
            )


@pytest.fixture(autouse=True)
def log_chatbot_calls(request, monkeypatch):
    """Log question, chunks, and answer for every chatbot call in the test."""
    results.start_test(request.node.nodeid)
    original_ask = Chatbot.ask

    def logged_ask(self, question):
        answer = original_ask(self, question)
        results.record(
            "ask",
            question=question,
            chunks=[{"id": chunk.id, "text": chunk.text} for chunk in answer.chunks],
            answer=answer.text,
        )
        return answer

    monkeypatch.setattr(Chatbot, "ask", logged_ask)
    yield
    report = getattr(request.node, "call_report", None)
    results.finish_test(report.outcome if report else "error")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if report.when == "call":
        item.call_report = report


def pytest_sessionfinish(session, exitstatus):
    tests = results.all_tests()
    summary = {
        "mode": llm_mode(),
        "total_attempts": len(tests),
        "reruns": sum(1 for test in tests if test["attempt"] > 1),
        "tests": tests,
    }
    RESULTS_FILE.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
