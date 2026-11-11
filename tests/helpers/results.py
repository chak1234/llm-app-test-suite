"""Collects what each test asked, retrieved, answered, and scored.

conftest.py sets the current test and writes everything to results.json at
the end of the session, so a failure can be debugged without re-running.
"""
from __future__ import annotations

_current: dict | None = None
_tests: list[dict] = []


def start_test(test_id: str) -> None:
    global _current
    # A rerun of the same test adds a second entry; the attempt number tells them apart.
    attempt = 1 + sum(1 for entry in _tests if entry["test"] == test_id)
    _current = {"test": test_id, "attempt": attempt, "outcome": None, "events": []}
    _tests.append(_current)


def record(kind: str, **data) -> None:
    if _current is not None:
        _current["events"].append({"kind": kind, **data})


def finish_test(outcome: str) -> None:
    global _current
    if _current is not None:
        _current["outcome"] = outcome
    _current = None


def all_tests() -> list[dict]:
    return _tests
