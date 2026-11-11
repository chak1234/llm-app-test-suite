"""Faithfulness scoring with a judge model.

The judge lists each factual claim in the answer and marks it supported or
not supported by the retrieved chunks. Score = supported / total.
"""
from __future__ import annotations

import json
import re

from app.llm import LLMClient, get_llm
from tests.helpers import results

JUDGE_SYSTEM = """You are a strict fact-checker. You will receive SOURCE documents and an ANSWER.

1. List every factual claim the ANSWER makes.
2. Mark a claim as supported only if the SOURCE documents state it. A claim that is true in general but absent from the SOURCE is not supported.
3. Ignore statements that are not factual claims, such as "I don't know", greetings, or a suggestion to contact support.

Reply with JSON only, in exactly this shape:
{"claims": [{"claim": "...", "supported": true}], "supported": 0, "total": 0}

"supported" is the number of supported claims and "total" is the number of claims."""


class JudgeError(RuntimeError):
    """The judge reply could not be turned into a score."""


def parse_judge_reply(reply: str) -> tuple[int, int]:
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    if not match:
        raise JudgeError(f"Judge reply has no JSON object: {reply!r}")
    try:
        data = json.loads(match.group(0))
        supported, total = int(data["supported"]), int(data["total"])
    except (ValueError, KeyError, TypeError) as error:
        raise JudgeError(f"Judge reply is not the expected JSON: {reply!r}") from error
    if not 0 <= supported <= total:
        raise JudgeError(f"Judge counts are impossible: supported={supported}, total={total}")
    return supported, total


def faithfulness_score(answer_text: str, chunks: list, judge: LLMClient | None = None) -> float:
    """Return supported claims / total claims, between 0.0 and 1.0.

    An answer with no factual claims (a plain refusal) scores 1.0: it says
    nothing the chunks fail to support.
    """
    judge = judge or get_llm("judge")
    sources = "\n\n".join(f"[{i + 1}] {chunk.text}" for i, chunk in enumerate(chunks))
    reply = judge.complete(JUDGE_SYSTEM, f"SOURCE:\n{sources}\n\nANSWER:\n{answer_text}")
    supported, total = parse_judge_reply(reply)
    score = 1.0 if total == 0 else supported / total
    results.record("judge", answer=answer_text, supported=supported, total=total, score=score)
    return score
