import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from daily_profit_engine import TradeOpportunity


ROOT = Path(__file__).resolve().parents[1]
RUN_V2 = ROOT / "run-v2.py"


def _load_run_v2():
    spec = importlib.util.spec_from_file_location("run_v2_for_tests", RUN_V2)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_live_mode_refused_by_default(monkeypatch):
    mod = _load_run_v2()
    monkeypatch.setenv("MODE", "live")
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_PAPER_ONLY", "true")

    assert mod.paper_mode_allowed() is False


def test_live_orders_require_live_mode_and_paper_only_false(monkeypatch):
    mod = _load_run_v2()
    cases = [
        ("paper", "true", False),
        ("paper", "false", False),
        ("live", "true", False),
        ("live", "false", True),
    ]
    for mode, paper_only, expected in cases:
        monkeypatch.setenv("MODE", mode)
        monkeypatch.setenv("DAILY_PROFIT_ENGINE_PAPER_ONLY", paper_only)
        assert mod.live_orders_allowed() is expected


def test_disabled_engine_calls_legacy_cycle(monkeypatch):
    mod = _load_run_v2()
    calls = {"legacy": 0}

    def legacy(*args, **kwargs):
        calls["legacy"] += 1
        return {"legacy": True}

    monkeypatch.setenv("DAILY_PROFIT_ENGINE_ENABLED", "false")
    monkeypatch.setattr(mod.run, "run_two_engine_cycle", legacy)
    monkeypatch.setattr(mod.run, "collect_daily_profit_opportunities", lambda **_: (_ for _ in ()).throw(AssertionError("collector called")))
    monkeypatch.setattr(mod, "DailyOpportunityRouter", lambda *a, **k: (_ for _ in ()).throw(AssertionError("router built")))
    monkeypatch.setattr(mod, "execute_approved", lambda *a, **k: (_ for _ in ()).throw(AssertionError("v2 entries executed")))

    result = mod.run_v2_cycle(
        alpha=None,
        bal=None,
        monitor=None,
        gateway=object(),
        risk=object(),
        waterfall=object(),
        market_context={},
    )

    assert result == {"legacy": True}
    assert calls["legacy"] == 1


def test_normal_main_calls_v2_trading_not_legacy_main(monkeypatch):
    mod = _load_run_v2()
    calls = {"v2": 0, "legacy_main": 0}
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_ENABLED", "true")
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_PAPER_ONLY", "true")
    monkeypatch.setattr(mod, "run_v2_trading", lambda: calls.__setitem__("v2", calls["v2"] + 1))
    monkeypatch.setattr(mod.run, "main", lambda: calls.__setitem__("legacy_main", calls["legacy_main"] + 1))

    assert mod.main([]) == 0
    assert calls == {"v2": 1, "legacy_main": 0}


def test_disabled_main_calls_legacy_run_main(monkeypatch):
    mod = _load_run_v2()
    calls = {"v2": 0, "legacy_main": 0}
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_ENABLED", "false")
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_PAPER_ONLY", "true")
    monkeypatch.setattr(mod, "run_v2_trading", lambda: calls.__setitem__("v2", calls["v2"] + 1))
    monkeypatch.setattr(mod.run, "main", lambda: calls.__setitem__("legacy_main", calls["legacy_main"] + 1) or 0)

    assert mod.main([]) == 0
    assert calls == {"v2": 0, "legacy_main": 1}


def test_normal_run_v2_cycle_calls_router(monkeypatch, tmp_path):
    mod = _load_run_v2()
    calls = {"route": 0}
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_ENABLED", "true")
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_PAPER_ONLY", "true")
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(mod.run, "ENABLE_MARKET_REGIME", False)
    monkeypatch.setattr(mod.run, "ENABLE_EMA_EXIT", False)
    monkeypatch.setattr(mod.run, "ENABLE_INSIDER_COPIER", False)
    monkeypatch.setattr(mod.run, "run_option_safety_exits", lambda *a, **k: 0)
    monkeypatch.setattr(
        mod.run,
        "collect_daily_profit_opportunities",
        lambda **_: {"opportunities": [TradeOpportunity(symbol="SPY", expected_r=2.0, probability=0.8, score=80, reference_price=500)]},
    )

    class SpyRouter:
        def __init__(self, *args, **kwargs):
            pass

        def route(self, *args, **kwargs):
            calls["route"] += 1
            from daily_profit_engine import DailyOpportunityRouter
            return DailyOpportunityRouter().route(*args, **kwargs)

        def write_funnel(self, *args, **kwargs):
            from daily_profit_engine import DailyOpportunityRouter
            return DailyOpportunityRouter().write_funnel(*args, **kwargs)

    class Gateway:
        def get_account(self):
            return {"equity": 100000, "cash": 100000}

        def get_positions_raw(self):
            return []

        def split_positions(self, raw):
            return [], {}, 0.0, 0.0

        def submit_market_order(self, *args, **kwargs):
            return None

    monkeypatch.setattr(mod, "DailyOpportunityRouter", SpyRouter)
    risk = SimpleNamespace(sync_account=lambda account: None, sync_exposure=lambda *a: None)

    mod.run_v2_cycle(None, None, None, Gateway(), risk, object(), {"overall_market_bias": "bullish"})

    assert calls["route"] == 1


def test_exits_execute_when_router_rejects_every_entry(monkeypatch, tmp_path):
    mod = _load_run_v2()
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_ENABLED", "true")
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_PAPER_ONLY", "true")
    monkeypatch.setenv("DAILY_MAX_NEW_ENTRIES", "0")
    monkeypatch.setenv("DAILY_PROFIT_ENGINE_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(mod.run, "ENABLE_MARKET_REGIME", False)
    monkeypatch.setattr(mod.run, "ENABLE_EMA_EXIT", False)
    monkeypatch.setattr(mod.run, "ENABLE_INSIDER_COPIER", False)

    calls = {"alpha_exit": 0, "bal_exit": 0, "safety_exit": 0}
    monkeypatch.setattr(mod.run, "run_alpha_exits", lambda *a, **k: calls.__setitem__("alpha_exit", calls["alpha_exit"] + 1) or 1)
    monkeypatch.setattr(mod.run, "run_bal_exits", lambda *a, **k: calls.__setitem__("bal_exit", calls["bal_exit"] + 1) or 1)
    monkeypatch.setattr(mod.run, "run_option_safety_exits", lambda *a, **k: calls.__setitem__("safety_exit", calls["safety_exit"] + 1) or 1)
    monkeypatch.setattr(
        mod.run,
        "collect_daily_profit_opportunities",
        lambda **_: {
            "opportunities": [
                TradeOpportunity(
                    symbol="REJECT",
                    expected_r=2.0,
                    probability=0.8,
                    score=90,
                    reference_price=10.0,
                )
            ],
            "stale_state": True,
        },
    )

    class Gateway:
        def get_account(self):
            return {"equity": 100000, "cash": 100000}

        def get_positions_raw(self):
            return []

        def split_positions(self, raw):
            return [], {}, 0.0, 0.0

        def submit_market_order(self, *args, **kwargs):
            raise AssertionError("entries must not execute")

    risk = SimpleNamespace(sync_account=lambda account: None, sync_exposure=lambda *a: None)

    summary = mod.run_v2_cycle(
        alpha=object(),
        bal=object(),
        monitor=None,
        gateway=Gateway(),
        risk=risk,
        waterfall=object(),
        market_context={"overall_market_bias": "bearish"},
    )

    assert calls == {"alpha_exit": 1, "bal_exit": 1, "safety_exit": 1}
    assert summary["v2_entries"] == 0
    data = json.loads(Path(summary["daily_profit_funnel_path"]).read_text())
    assert data["safe_mode"] is True
    assert data["approved_entries"] == 0


def test_demo_cycle_writes_funnel(tmp_path):
    env = dict(os.environ)
    env["DAILY_PROFIT_ENGINE_DEMO"] = "1"
    env["DAILY_PROFIT_ENGINE_PAPER_ONLY"] = "true"
    env["DAILY_PROFIT_ENGINE_LOG_DIR"] = str(tmp_path)

    proc = subprocess.run(
        [sys.executable, str(RUN_V2), "--demo-cycle"],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    marker = "FUNNEL_JSON="
    assert marker in proc.stdout
    funnel_path = Path(proc.stdout.split(marker, 1)[1].splitlines()[0].strip())
    assert funnel_path.exists()
    data = json.loads(funnel_path.read_text())
    assert data["candidates"] >= 1
    assert "approved_entries" in data
