#!/usr/bin/env python3
"""Read-only pre-market preflight for run.py.

Every check here is read-only: it never calls a broker order-submission
method and never writes to any shared state file (kill-switch, scanner
state, order CSVs). Safe to run at any time, including while a live run.py
process is running in this checkout.

Secret values (API keys, tokens, passwords) are never printed -- only
presence/absence. Non-secret config (broker base URL, data feed name,
universe size) is printed for operator visibility.

Exit code 0 = all checks passed. Exit code 1 = at least one check failed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import run  # noqa: E402
import universe  # noqa: E402

RESULTS: list[dict] = []


def check(name: str):
    def decorator(fn):
        try:
            detail = fn()
            RESULTS.append({"check": name, "status": "PASS", "detail": detail})
        except AssertionError as exc:
            RESULTS.append({"check": name, "status": "FAIL", "detail": str(exc)})
        except Exception as exc:
            RESULTS.append({"check": name, "status": "ERROR", "detail": repr(exc)})
        return fn
    return decorator


SECRET_ENV_NAMES = {
    "APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "ALPHA_VANTAGE_API_KEY",
    "NEWSAPI_KEY", "OPENAI_API_KEY", "FINNHUB_API_KEY", "FINNHUB_API",
    "SERPAPI_KEY", "DISCORD_BOT_TOKEN", "GH_TOKEN", "GITHUB_CLIENT_SECRET",
    "GOOGLE_CLIENT_SECRET", "SUPABASE_KEY", "WEBHOOK_SECRET", "SYNC_SECRET",
    "DASH_PASSWORD", "DATABENTO_API", "MASSIVE_API_KEY", "BITBUCKET_API",
    "FLASHALPHA_API", "SEC_USER_AGENT",
}


@check("required_env_vars_present")
def _():
    required = ["APCA_API_KEY_ID", "APCA_API_SECRET_KEY", "APCA_API_BASE_URL"]
    missing = [v for v in required if not os.getenv(v)]
    assert not missing, f"missing required env vars: {missing}"
    return {v: ("<set>" if v in SECRET_ENV_NAMES else os.getenv(v)) for v in required}


@check("broker_endpoint_is_paper")
def _():
    base_url = os.getenv("APCA_API_BASE_URL", "")
    assert "paper-api" in base_url, f"APCA_API_BASE_URL does not look like the paper endpoint: {base_url!r}"
    return {"APCA_API_BASE_URL": base_url}


# Single shared read-only broker instance for the account/clock checks below.
# dry_run=True is belt-and-suspenders -- preflight never calls a submit_*
# method regardless.
_broker = run.AlpacaBroker(run.ENV_PATH, paper=True, dry_run=True)


@check("account_readable")
def _():
    acct = _broker.get_account_snapshot()
    assert acct.equity >= 0, f"unexpected negative equity: {acct.equity}"
    return {"equity": acct.equity, "cash": acct.cash, "buying_power": acct.buying_power}


@check("market_clock_readable")
def _():
    info = _broker.get_clock_info()
    assert "is_open" in info, "clock payload missing is_open"
    return info


@check("positions_and_open_orders_readable")
def _():
    positions = _broker.get_all_open_positions()
    open_orders = _broker.list_open_orders()
    return {"open_position_count": len(positions), "open_order_count": len(open_orders)}


@check("data_feed_works")
def _():
    scanner = run.MomentumSetupScanner(_broker, run.ScannerConfig(), run.ExecutorConfig(paper=True, dry_run=True))
    bars = scanner._download_bars(["SPY"])
    assert bars is not None, "data feed returned None"
    return {"feed": os.getenv("ALPACA_DATA_FEED", "iex"), "spy_bar_rows": int(len(bars))}


@check("universe_valid_and_deduplicated")
def _():
    result = universe.build_universe(run.DEFAULT_SCAN_UNIVERSE_CANDIDATES, max_symbols=30)
    symbols = result.symbols
    assert len(symbols) == len(set(symbols)), "duplicate symbols survived universe construction"
    bad = [s for s in symbols if s in universe.KNOWN_INVALID_SYMBOLS or not universe.VALID_SYMBOL_RE.match(s)]
    assert not bad, f"invalid symbols survived universe construction: {bad}"
    return {
        "universe_size": len(symbols),
        "excluded_count": len(result.excluded),
        "excluded_reasons": [f"{e.symbol}: {e.reason}" for e in result.excluded],
    }


@check("required_files_and_dirs_exist")
def _():
    required_paths = [
        PROJECT_ROOT / ".env",
        PROJECT_ROOT / "run.py",
        PROJECT_ROOT / "universe.py",
        PROJECT_ROOT / "daily_system_outputs",
        PROJECT_ROOT / "logs" / "execution",
        PROJECT_ROOT / "tests",
    ]
    missing = [str(p) for p in required_paths if not p.exists()]
    assert not missing, f"missing required files/dirs: {missing}"
    return {"checked": [str(p) for p in required_paths]}


@check("kill_switch_state_readable")
def _():
    path = run.KILL_SWITCH_PATH
    if not path.exists():
        return {"path": str(path), "exists": False, "note": "no state file yet -- treated as a fresh/clean kill-switch state by run.py"}
    state = json.loads(path.read_text(encoding="utf-8"))
    return {"path": str(path), "exists": True, "kill_triggered": state.get("kill_triggered"), "date": state.get("date")}


@check("no_order_submitted_during_preflight")
def _():
    assert _broker.dry_run_order_attempts == 0, "preflight unexpectedly attempted an order submission"
    return {"dry_run_order_attempts": _broker.dry_run_order_attempts}


def main() -> int:
    print("=" * 78)
    print("[preflight] run.py pre-market preflight (read-only, no orders submitted)")
    print("=" * 78)
    failed = 0
    for r in RESULTS:
        marker = {"PASS": "OK  ", "FAIL": "FAIL", "ERROR": "ERR "}[r["status"]]
        print(f"[{marker}] {r['check']}: {r['detail']}")
        if r["status"] != "PASS":
            failed += 1
    print("=" * 78)
    print(f"[preflight] {len(RESULTS) - failed}/{len(RESULTS)} checks passed")
    out_path = PROJECT_ROOT / "daily_system_outputs" / "preflight_last_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(RESULTS, indent=2, default=str))
    print(f"[preflight] wrote results -> {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
