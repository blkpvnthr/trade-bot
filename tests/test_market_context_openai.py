"""Tests for the resilient OpenAI market-context classifier (obj 9).

The live crash was `OpenAI classification failed: Unterminated string ...` --
json.loads() on a response truncated by the max_tokens cap, with no truncation
detection, no app-level retry, and only an opaque error. These pin:
  * a truncated response (finish_reason='length') is DETECTED and retried
  * repeated truncation falls back DETERMINISTICALLY (never raises by default)
  * a malformed-JSON response is retried, then falls back
  * a valid response is parsed and tagged classification_source='openai'
  * the deterministic fallback is tagged classification_source='fallback'
  * core execution never depends on the LLM (fail-open by default)
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

import market_context_builder as mcb

_VALID_JSON = (
    '{"overall_market_bias":"bullish","trade_permission":"normal",'
    '"risk_level":"low","summary":"ok","decision_reasons":[]}'
)
# A response cut off mid-string -> json.loads raises "Unterminated string".
_TRUNCATED_JSON = '{"overall_market_bias":"bullish","summary":"the market is '


def _resp(content, finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason)])


class _FakeOpenAI:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


@pytest.fixture(autouse=True)
def _enable_openai(monkeypatch):
    # Force the OpenAI branch (not the missing-key / disabled early returns).
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(mcb, "MARKET_CONTEXT_DISABLE_OPENAI", False)
    monkeypatch.setattr(mcb, "MARKET_CONTEXT_FAIL_OPENAI_CLOSED", False)


def test_valid_response_is_parsed_and_tagged_openai():
    client = _FakeOpenAI(_resp(_VALID_JSON))
    out = mcb.classify_with_openai({}, client=client)
    assert out["overall_market_bias"] == "bullish"
    assert out["classification_source"] == "openai"
    assert "event_flags" in out                    # validate_event_flags ran
    assert len(client.calls) == 1


def test_truncated_then_complete_retries_and_succeeds():
    client = _FakeOpenAI(
        _resp(_TRUNCATED_JSON, finish_reason="length"),   # 1st: truncated
        _resp(_VALID_JSON, finish_reason="stop"),         # 2nd: complete
    )
    out = mcb.classify_with_openai({}, client=client)
    assert out["classification_source"] == "openai"
    assert len(client.calls) == 2
    # The retry escalated the output-token budget to avoid re-truncation.
    assert client.calls[1]["max_tokens"] > client.calls[0]["max_tokens"]


def test_persistent_truncation_falls_back_deterministically():
    client = _FakeOpenAI(
        _resp(_TRUNCATED_JSON, finish_reason="length"),
        _resp(_TRUNCATED_JSON, finish_reason="length"),
    )
    out = mcb.classify_with_openai({}, client=client)
    assert out["classification_source"] == "fallback"   # never raised
    assert out["overall_market_bias"] in {"bullish", "bearish", "mixed", "risk_off"}


def test_malformed_json_without_length_flag_is_retried_then_fallbacks():
    # finish_reason='stop' but still unparseable -> retry once, then fallback.
    client = _FakeOpenAI(
        _resp("not json at all", finish_reason="stop"),
        _resp("still not json", finish_reason="stop"),
    )
    out = mcb.classify_with_openai({}, client=client)
    assert out["classification_source"] == "fallback"
    assert len(client.calls) == 2


def test_api_exception_falls_back_immediately_no_retry():
    class _Boom:
        def __init__(self):
            self.calls = 0
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            self.calls += 1
            raise RuntimeError("auth error")

    client = _Boom()
    out = mcb.classify_with_openai({}, client=client)
    assert out["classification_source"] == "fallback"
    assert client.calls == 1          # non-JSON errors don't burn retries


def test_fail_closed_mode_raises_on_persistent_failure(monkeypatch):
    monkeypatch.setattr(mcb, "MARKET_CONTEXT_FAIL_OPENAI_CLOSED", True)
    client = _FakeOpenAI(
        _resp(_TRUNCATED_JSON, finish_reason="length"),
        _resp(_TRUNCATED_JSON, finish_reason="length"),
    )
    with pytest.raises(RuntimeError):
        mcb.classify_with_openai({}, client=client)
