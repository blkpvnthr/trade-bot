import os
import pytest
import trade


def test_streaming_and_terminal_env_defaults(monkeypatch):
    for k in ("STREAMING_ENGINE", "TERMINAL_MODE", "APCA_PAPER"):
        monkeypatch.delenv(k, raising=False)
    c = trade.build_default_config()
    assert c.enable_streaming_engine is False
    assert c.terminal_mode == "dashboard"
    assert c.paper is True


def test_streaming_env_enables(monkeypatch):
    monkeypatch.setenv("STREAMING_ENGINE", "1")
    monkeypatch.setenv("TERMINAL_MODE", "signals")
    c = trade.build_default_config()
    assert c.enable_streaming_engine is True
    assert c.terminal_mode == "signals"


def test_live_guard_refuses_without_authorization(monkeypatch):
    monkeypatch.delenv("ALLOW_LIVE_TRADING", raising=False)
    c = trade.build_default_config()
    c.paper = False  # simulate live request
    with pytest.raises(RuntimeError, match="ALLOW_LIVE_TRADING"):
        trade.validate_and_print_execution_mode(c)


def test_live_guard_allows_with_authorization(monkeypatch):
    monkeypatch.setenv("ALLOW_LIVE_TRADING", "1")
    c = trade.build_default_config()
    c.paper = False
    trade.validate_and_print_execution_mode(c)  # should not raise


def test_invalid_terminal_mode_rejected(monkeypatch):
    c = trade.build_default_config()
    c.terminal_mode = "bogus"
    with pytest.raises(RuntimeError, match="TERMINAL_MODE"):
        trade.validate_and_print_execution_mode(c)


def test_paper_banner_ok(monkeypatch):
    c = trade.build_default_config()
    trade.validate_and_print_execution_mode(c)  # paper -> no raise
