"""LLM client: local (Ollama), hosted (any OpenAI-compatible API), and replay.

Configuration comes from environment variables:

    LLM_MODE       replay (default) | record | live
    LLM_PROVIDER   ollama (default) | hosted
    LLM_MODEL      model name; pin an exact version
    OLLAMA_HOST    default http://localhost:11434
    LLM_BASE_URL   hosted only, e.g. https://api.example.com/v1
    LLM_API_KEY    hosted only

The judge model reads JUDGE_PROVIDER / JUDGE_MODEL / JUDGE_BASE_URL /
JUDGE_API_KEY and falls back to the LLM_* value for any that is unset.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Protocol

import requests

ROOT = Path(__file__).resolve().parent.parent
CASSETTE_DIR = ROOT / "tests" / "cassettes"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
TIMEOUT_SECONDS = 120


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str: ...


class OllamaClient:
    """Local model served by Ollama."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def complete(self, system: str, user: str) -> str:
        response = requests.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "options": {"temperature": 0, "seed": 0},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


class HostedClient:
    """Any provider that exposes an OpenAI-compatible /chat/completions route."""

    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def complete(self, system: str, user: str) -> str:
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


class MissingCassetteError(RuntimeError):
    pass


def prompt_key(system: str, user: str) -> str:
    """Hash of the full prompt. Any change to rules, chunks, or question changes it."""
    return hashlib.sha256(f"{system}\n\x00\n{user}".encode("utf-8")).hexdigest()


class CassetteClient:
    """Wraps a real client with saved responses in tests/cassettes/.

    record: call the real model and save the response.
    replay: return the saved response; fail if there is none.
    """

    def __init__(self, inner: LLMClient | None, mode: str, directory: Path = CASSETTE_DIR):
        if mode not in ("record", "replay"):
            raise ValueError(f"CassetteClient mode must be record or replay, got {mode!r}")
        if mode == "record" and inner is None:
            raise ValueError("record mode needs a real client")
        self.inner = inner
        self.mode = mode
        self.directory = directory
        self.model = inner.model if inner else "replay"

    def _path(self, key: str) -> Path:
        return self.directory / f"{key[:32]}.json"

    def complete(self, system: str, user: str) -> str:
        key = prompt_key(system, user)
        path = self._path(key)
        if self.mode == "replay":
            if not path.exists():
                raise MissingCassetteError(
                    f"No recorded response for this prompt ({path.name}). "
                    "The prompt, the documents, or the question changed since the last "
                    "recording. Re-record with LLM_MODE=record.\n"
                    f"Prompt ends with: ...{user[-200:]}"
                )
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        response = self.inner.complete(system, user)
        self.directory.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"key": key, "model": self.inner.model, "user": user, "response": response},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return response


def _setting(role: str, name: str, default: str | None = None) -> str | None:
    if role == "judge":
        value = os.environ.get(f"JUDGE_{name}")
        if value:
            return value
    return os.environ.get(f"LLM_{name}") or default


def _real_client(role: str) -> LLMClient:
    provider = _setting(role, "PROVIDER", "ollama")
    if provider == "ollama":
        return OllamaClient(
            model=_setting(role, "MODEL", DEFAULT_OLLAMA_MODEL),
            host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        )
    if provider == "hosted":
        missing = [n for n in ("MODEL", "BASE_URL", "API_KEY") if not _setting(role, n)]
        if missing:
            names = ", ".join(f"LLM_{n}" for n in missing)
            raise RuntimeError(f"Hosted provider needs these environment variables: {names}")
        return HostedClient(
            model=_setting(role, "MODEL"),
            base_url=_setting(role, "BASE_URL"),
            api_key=_setting(role, "API_KEY"),
        )
    raise RuntimeError(f"Unknown LLM provider {provider!r}; use 'ollama' or 'hosted'")


def llm_mode() -> str:
    mode = os.environ.get("LLM_MODE", "replay").lower()
    if mode not in ("replay", "record", "live"):
        raise RuntimeError(f"LLM_MODE must be replay, record, or live; got {mode!r}")
    return mode


def get_llm(role: str = "app") -> LLMClient:
    """Build the client for role 'app' (the chatbot) or 'judge'."""
    mode = llm_mode()
    if mode == "replay":
        return CassetteClient(None, "replay")
    client = _real_client(role)
    if mode == "record":
        return CassetteClient(client, "record")
    return client
