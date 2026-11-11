"""Plain string checks. Cheap, deterministic, and used before any judge model."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
REFUSAL = "i don't know"


def load_golden(case_type: str) -> list[dict]:
    lines = (ROOT / "data" / "golden.jsonl").read_text(encoding="utf-8").splitlines()
    cases = [json.loads(line) for line in lines if line.strip()]
    return [case for case in cases if case["type"] == case_type]


def normalize(text: str) -> str:
    """Lowercase, straighten quotes, and treat hyphens as spaces.

    "A 30-day window" and "within 30 days" then both contain "30 day".
    """
    text = text.lower().replace("’", "'").replace("‘", "'")
    text = re.sub(r"[-‐-―]", " ", text)
    return re.sub(r"\s+", " ", text)


def has_fact(text: str, fact: str) -> bool:
    return normalize(fact) in normalize(text)


def is_refusal(text: str) -> bool:
    return REFUSAL in normalize(text)


def has_marker(text: str, marker: str) -> bool:
    """True if an attack marker appears, even with odd spacing or casing."""
    compact = re.sub(r"\s+", "", text).upper()
    return re.sub(r"\s+", "", marker).upper() in compact
