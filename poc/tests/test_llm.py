import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import llm


class FakeMessages:
    def __init__(self, reply, stop_reason="end_turn"):
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []

    def create(self, **kw):
        self.calls.append(kw)
        content = [NS(type="thinking", thinking=""), NS(type="text", text=self.reply)]
        return NS(content=content, stop_reason=self.stop_reason, stop_details=NS(category="cyber"))


def fake_client(monkeypatch, reply, stop_reason="end_turn"):
    msgs = FakeMessages(reply, stop_reason)
    monkeypatch.setattr(llm, "_claude_client", NS(beta=NS(messages=msgs)))
    return msgs


def test_claude_request_and_text(monkeypatch):
    msgs = fake_client(monkeypatch, "Hello")
    assert llm._claude("claude-opus-5-5", "Say hi", "Be brief.", json_mode=False) == "Hello"
    kw = msgs.calls[0]
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == "Be brief."
    assert kw["messages"] == [{"role": "user", "content": "Say hi"}]
    assert kw["output_config"] == {"effort": "medium"} and kw["fallbacks"] == "default"
    assert "temperature" not in kw  # rejected by Claude Opus 5.5


def test_claude_json_mode_extracts_object(monkeypatch):
    msgs = fake_client(monkeypatch, 'Here you go:\n```json\n{"tickers": ["AAPL"]}\n```')
    assert llm._claude("claude-opus-5-5", "plan", None, json_mode=True) == '{"tickers": ["AAPL"]}'
    assert "JSON object only" in msgs.calls[0]["system"]


def test_claude_refusal_raises(monkeypatch):
    fake_client(monkeypatch, "", stop_reason="refusal")
    with pytest.raises(RuntimeError, match="declined"):
        llm._claude("claude-opus-5-5", "x", None, json_mode=False)
