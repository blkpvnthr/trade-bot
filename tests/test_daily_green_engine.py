import trade


def test_scanner_config_has_daily_green_defaults():
    cfg = trade.ScannerConfig()
    assert cfg.daily_green_strategy_enabled is True
    assert cfg.daily_green_rr == 2.0
    assert cfg.daily_green_min_risk_pct == 0.0015
    assert cfg.daily_green_per_position_weight == 0.09
    assert cfg.daily_green_max_positions == 8
    assert cfg.daily_green_flatten_before_close_min == 5
    assert cfg.daily_green_reenter_after_stop is False


def test_build_default_config_uses_universe_txt_in_daily_green():
    cfg = trade.build_default_config().scanner
    assert cfg.daily_green_strategy_enabled is True
    # universe.txt is reached only when all other universe sources are off
    assert cfg.use_llm_sentiment_universe is False
    assert cfg.use_sp500_fallback is False
    assert cfg.broad_market_scan is False
    assert cfg.universe is None


def test_load_trading_universe_parses_file(tmp_path, monkeypatch):
    f = tmp_path / "universe.txt"
    f.write_text("AAPL, msft  nvda # comment\nAAPL\n")
    monkeypatch.setattr(trade, "UNIVERSE_FILE", f)
    assert trade.load_trading_universe() == ["AAPL", "MSFT", "NVDA"]


def test_load_trading_universe_falls_back_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(trade, "UNIVERSE_FILE", tmp_path / "nope.txt")
    out = trade.load_trading_universe()
    assert out == trade._FALLBACK_SCAN_UNIVERSE
    assert "QQQ" in out
