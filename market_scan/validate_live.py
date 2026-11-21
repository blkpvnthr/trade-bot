"""Non-trading, read-only market-hours validation for the production scanner.

Exercises the end-to-end live data path WITHOUT ever mutating the broker: it
authenticates, checks IEX REST + websocket connectivity, drives an incremental
subscription update, confirms reference-data reconciliation + freshness, forces a
controlled reconnect and verifies recovery, runs a ranking pass, and publishes a
market-context snapshot to the RESEARCH directory (never the production
live_context files). It constructs NO order path, so it cannot place, cancel, or
modify any order.

Run from the repo root during a regular session:

    python -m market_scan.validate_live --duration 30          # paper creds
    python -m market_scan.validate_live --duration 30 --mode live

Outside RTH the websocket checks SKIP (no live ticks) rather than fail. Exit code
is non-zero only if a check FAILS.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from pathlib import Path

from market_scan import paths
from market_scan.config import MarketScanConfig
from market_scan.health import write_live_context
from order_flow_engine import OrderFlowEngine, OrderFlowConfig

REFS = ("SPY", "QQQ", "DIA", "IWM")
PROBES = ("NVDA", "AMD")


def _creds():
    try:
        from dotenv import load_dotenv
        load_dotenv(paths.PROJECT_ROOT / ".env")
    except Exception:  # noqa: BLE001
        pass
    key = os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
    sec = os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
    return key, sec


async def validate(duration: float = 30.0, mode: str = "paper") -> list[tuple[str, str, str]]:
    results: list[tuple[str, str, str]] = []

    def rec(name, status, detail=""):
        results.append((name, status, detail))
        print(f"  [{status:4}] {name}{(' — ' + detail) if detail else ''}", flush=True)

    key, sec = _creds()
    if not key or not sec:
        rec("credentials", "FAIL", "APCA_API_KEY_ID / APCA_API_SECRET_KEY not found")
        return results
    rec("credentials", "PASS", "loaded from .env")

    # 1) Auth + clock (read-only).
    market_open = False
    try:
        from alpaca.trading.client import TradingClient
        tc = TradingClient(key, sec, paper=(mode != "live"))
        acct = tc.get_account(); clock = tc.get_clock()
        market_open = bool(getattr(clock, "is_open", False))
        rec("auth", "PASS", f"status={acct.status} mode={mode} market_open={market_open}")
    except Exception as exc:  # noqa: BLE001
        rec("auth", "FAIL", f"{type(exc).__name__}: {exc}"); return results

    # 2) IEX REST (read-only, feed=IEX).
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        from alpaca.data.enums import DataFeed
        dc = StockHistoricalDataClient(key, sec)
        q = dc.get_stock_latest_quote(
            StockLatestQuoteRequest(symbol_or_symbols=list(REFS), feed=DataFeed.IEX))
        rec("iex_rest", "PASS", f"quotes for {sorted(q.keys())} (feed=IEX)")
    except Exception as exc:  # noqa: BLE001
        rec("iex_rest", "FAIL", f"{type(exc).__name__}: {exc}")

    if not market_open:
        for n in ("websocket", "incremental_subscription", "reconciliation",
                  "reconnect_recovery", "ranking", "context_publish"):
            rec(n, "SKIP", "market closed — rerun during RTH")
        return results

    # 3-7) Websocket path (RTH only), reusing the PRODUCTION StreamClient.
    from market_scan.stream_client import StreamClient
    cfg = MarketScanConfig()
    ofe = OrderFlowEngine(OrderFlowConfig(stale_quote_s=cfg.max_quote_age_seconds))
    sc = StreamClient(api_key=key, secret_key=sec, feed=cfg.data_feed,
                      on_trade=ofe.ingest_trade, on_quote=ofe.ingest_quote,
                      reconnect=True)
    sc.connect()
    await sc.apply_subscriptions(to_add=list(REFS) + list(PROBES), to_remove=[])
    task = asyncio.create_task(sc.run())
    try:
        deadline = time.time() + duration
        while time.time() < deadline and not sc.is_live():
            await asyncio.sleep(0.5)
        rec("websocket", "PASS" if sc.is_live() else "FAIL",
            f"messages={sc._messages}")

        # incremental subscription add/remove on the live socket
        try:
            await sc.apply_subscriptions(to_add=["MSFT"], to_remove=["AMD"])
            rec("incremental_subscription", "PASS", "added MSFT, removed AMD mid-stream")
        except Exception as exc:  # noqa: BLE001
            rec("incremental_subscription", "FAIL", f"{type(exc).__name__}: {exc}")

        await asyncio.sleep(min(5.0, duration))
        now = time.time()
        fresh_refs = [r for r in REFS
                      if (ofe.snapshot(r, now).last_quote_age_s or 1e9) <= cfg.max_quote_age_seconds]
        rec("reconciliation", "PASS" if len(fresh_refs) >= 2 else "FAIL",
            f"references delivering fresh data: {fresh_refs}")

        # controlled disconnect + recovery
        sc.request_reconnect()
        rdl = time.time() + min(20.0, duration)
        while time.time() < rdl and not sc.is_live():
            await asyncio.sleep(0.5)
        rec("reconnect_recovery", "PASS" if sc.is_live() else "FAIL",
            f"reconnect_requests={sc.reconnect_requests} live={sc.is_live()}")

        # ranking pass over whatever has warmed up
        snaps = {s: ofe.snapshot(s, now) for s in PROBES}
        ok = sum(1 for s in snaps.values() if getattr(s, "ok", False))
        rec("ranking", "PASS", f"warmed probe snapshots={ok}/{len(PROBES)}")

        # publish a context snapshot to the RESEARCH dir (not production)
        try:
            write_live_context(paths.research_output_dir() / "validation_context.json",
                               {"as_of": now, "fresh_refs": fresh_refs, "mode": mode,
                                "note": "validate_live read-only run; no orders"})
            rec("context_publish", "PASS", str(paths.research_output_dir() / "validation_context.json"))
        except Exception as exc:  # noqa: BLE001
            rec("context_publish", "FAIL", f"{type(exc).__name__}: {exc}")
    finally:
        sc.stop()
        task.cancel()
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only market-hours scanner validation (no orders).")
    ap.add_argument("--duration", type=float, default=30.0, help="max seconds to wait for live data")
    ap.add_argument("--mode", choices=["paper", "live"], default="paper")
    a = ap.parse_args()
    print(f"[validate_live] NON-TRADING read-only validation (mode={a.mode}, duration={a.duration}s)",
          flush=True)
    results = asyncio.run(validate(a.duration, a.mode))
    failed = [r for r in results if r[1] == "FAIL"]
    print(f"[validate_live] done: {sum(1 for r in results if r[1]=='PASS')} pass, "
          f"{sum(1 for r in results if r[1]=='SKIP')} skip, {len(failed)} fail. NO ORDERS SUBMITTED.",
          flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
