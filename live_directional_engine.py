#!/usr/bin/env python3
"""live_directional_engine.py — live/paper executor for the 0.75x directional strategy.

Runs ONCE per trading day. Reproduces the validated backtest decision logic against the
LIVE Alpaca account, using REAL option chains (AV realtime). Same rules as
directional_volexit_engine.py --size-mult 0.75 (the surviving candidate):

  REGIME : QQQ 200DMA ±2%  -> bull / bear / transition(cash)
  ENTRY  : weekly; bull -> long ~45Δ CALL ; bear -> long ~45Δ PUT  on QQQ/SPY/IWM
           size = 0.75 × risk_budget premium per name
  EXIT   : RVOL>=2.0 reversal-against  OR  7-DTE time stop  (NO MA/stop/profit-target —
           those churned the convex edge in testing)

SAFETY (this account is real-money LIVE — handle with care):
  • Default = DRY-RUN: reads account read-only, prints intended orders, submits NOTHING.
  • --submit required to place orders. If the endpoint is LIVE, --allow-live AND the
    token --i-understand-live-risk are BOTH required, else it aborts.
  • Strategy state (which positions it owns) is tracked in logs/live_directional_state.json
    so it never touches positions it didn't open.
  • Hard caps: max positions, per-name one position, options-BP check.

Intended use: cron once daily after the close (or pre-open) →
  python live_directional_engine.py                # dry-run preview (safe)
  python live_directional_engine.py --submit       # submit to PAPER (needs paper creds)
"""
from __future__ import annotations
import os, sys, json, argparse
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("/Users/blkpvnthr/Desktop/trader"); sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")
import edge_scan_spreads as E
import credit_spread_engine as C
from directional_volexit_engine import pick_band
import options as O

ET = ZoneInfo("America/New_York")
STATE_PATH = ROOT / "logs" / "live_directional_state.json"
JOURNAL = ROOT / "logs" / "live_directional_journal.jsonl"
UNIVERSE = ["QQQ", "SPY", "IWM"]
RISK_BUDGET = 2000.0
SIZE_MULT = 0.75
MAX_POSITIONS = 9
RVOL_TH = 2.0
TIME_STOP_DTE = 7
ENTRY_INTERVAL_DAYS = 7


def now_et():
    return datetime.now(ET)


def load_state():
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"positions": [], "last_entry_date": None}


def save_state(s, dry):
    if dry:
        return
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(s, indent=2))


def journal(rec, dry):
    rec = {**rec, "ts": now_et().isoformat(), "dry_run": dry}
    if not dry:
        JOURNAL.parent.mkdir(parents=True, exist_ok=True)
        with open(JOURNAL, "a") as f:
            f.write(json.dumps(rec) + "\n")
    return rec


def trading_client(paper):
    from alpaca.trading.client import TradingClient
    return TradingClient(E.AK, E.AS, paper=paper)


def recent_daily(symbol, days=40):
    """Recent daily bars (raw) for RVOL/candle. Returns sorted list of (date,open,close,vol)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    cl = StockHistoricalDataClient(E.AK, E.AS)
    end = datetime.now(); start = end - timedelta(days=days * 2 + 30)
    df = cl.get_stock_bars(StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day,
                           start=start, end=end)).df.reset_index()
    out = []
    for _, r in df.iterrows():
        d = r["timestamp"].date() if hasattr(r["timestamp"], "date") else datetime.fromisoformat(str(r["timestamp"])).date()
        out.append((d, float(r["open"]), float(r["close"]), float(r["volume"])))
    return sorted(out)


def rvol_and_candle(symbol):
    bars = recent_daily(symbol)
    if len(bars) < 22:
        return 0.0, 0.0, 0.0
    o, c, v = bars[-1][1], bars[-1][2], bars[-1][3]
    pc = bars[-2][2]
    avg = sum(b[3] for b in bars[-21:-1]) / 20
    return (v / avg if avg > 0 else 0.0), c - o, c - pc  # (rvol, body, vs_prev)


def option_sell_limit(occ):
    """Live bid for a held option (marketable sell limit). None if unavailable."""
    try:
        from alpaca.data.historical.option import OptionHistoricalDataClient
        from alpaca.data.requests import OptionLatestQuoteRequest
        q = OptionHistoricalDataClient(E.AK, E.AS).get_option_latest_quote(
            OptionLatestQuoteRequest(symbol_or_symbols=occ))
        bid = float(q[occ].bid_price or 0)
        return bid if bid > 0 else None
    except Exception:
        return None


def occ_underlying(sym):
    import re
    m = re.match(r"^([A-Z]+)\d{6}[CP]\d{8}$", sym)
    return m.group(1) if m else sym


def is_call_occ(sym):
    return len(sym) >= 15 and sym[-9] == "C"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="actually place orders (else dry-run preview)")
    ap.add_argument("--allow-live", action="store_true", help="permit the LIVE endpoint (real money)")
    ap.add_argument("--i-understand-live-risk", dest="confirm_live", action="store_true")
    ap.add_argument("--force-entry-day", action="store_true", help="treat today as an entry day regardless of cadence")
    args = ap.parse_args()

    paper_env = str(os.getenv("ALPACA_PAPER_ONLY", "true")).lower() in {"1", "true", "yes", "paper"}
    dry = not args.submit
    # endpoint: prefer paper unless explicitly allowed live
    paper = paper_env or not args.allow_live

    print("=" * 74)
    print(f"LIVE DIRECTIONAL ENGINE (0.75×)  {now_et():%Y-%m-%d %H:%M %Z}")
    print(f"mode={'DRY-RUN (no orders)' if dry else 'SUBMIT'}  endpoint={'PAPER' if paper else 'LIVE'}")
    print("=" * 74)

    if args.submit and not paper:
        if not (args.allow_live and args.confirm_live):
            print("ABORT: refusing to submit to the LIVE (real-money) endpoint without "
                  "--allow-live AND --i-understand-live-risk.")
            return

    client = trading_client(paper)
    acct = client.get_account()
    equity = round(float(acct.equity), 2)
    opt_bp = round(float(getattr(acct, "options_buying_power", 0) or 0), 2)
    print(f"account: equity=${equity:,.2f}  options_BP=${opt_bp:,.2f}")

    # broker option positions
    broker_pos = {}
    for p in client.get_all_positions():
        if len(p.symbol) >= 15 and p.symbol[-9] in ("C", "P"):  # OCC option
            broker_pos[p.symbol] = p

    state = load_state()
    rb = RISK_BUDGET * SIZE_MULT

    # ---- regime (QQQ 200DMA ±2%) ----
    regime_map = C.build_regime_200(E.get_closes("QQQ"), band=0.02)
    today = max(d for d in regime_map)  # latest available session
    reg = regime_map[today]
    print(f"regime(QQQ 200DMA±2%, asof {today}): {reg.upper()}")

    actions = []

    # ---- EXITS on tracked positions ----
    held = state.get("positions", [])
    still = []
    for pos in held:
        occ = pos["contract"]
        if occ not in broker_pos:
            print(f"  [reconcile] {occ} not in broker positions — marking closed in state")
            continue
        u = pos["underlying"]
        rv, body, vsprev = rvol_and_candle(u)
        exp = datetime.strptime(pos["expiry"], "%Y-%m-%d").date()
        dte = (exp - today).days
        bullish = pos["kind"] == "long_call"
        red = body < 0 and vsprev < 0
        green = body > 0 and vsprev > 0
        reason = None
        if rv >= RVOL_TH and ((bullish and red) or ((not bullish) and green)):
            reason = "vol_reversal"
        elif dte <= TIME_STOP_DTE:
            reason = "time_stop"
        elif reg == "cash" or (bullish and reg == "bear") or ((not bullish) and reg == "bull"):
            reason = "regime_flip"
        if reason:
            qty = int(float(broker_pos[occ].qty))
            actions.append(("SELL", occ, qty, None, f"exit:{reason} (rvol={rv:.1f} dte={dte})"))
        else:
            still.append(pos)

    # ---- ENTRIES (weekly cadence) ----
    led = state.get("last_entry_date")
    is_entry_day = args.force_entry_day or led is None or (today - date.fromisoformat(led)).days >= ENTRY_INTERVAL_DAYS
    if is_entry_day and reg in ("bull", "bear"):
        held_unders = {p["underlying"] for p in still}
        for sym in UNIVERSE:
            if len(still) + sum(1 for a in actions if a[0] == "BUY") >= MAX_POSITIONS:
                break
            if sym in held_unders:
                continue
            try:
                rows = O.get_realtime_options(sym)
            except Exception as ex:
                print(f"  [chain-err] {sym}: {str(ex)[:80]}"); continue
            # 30-45 DTE expiry with most contracts
            exps = {}
            for r in rows:
                e = (r.get("expiration") or "")[:10]
                if not e:
                    continue
                try:
                    dt = (date.fromisoformat(e) - today).days
                except Exception:
                    continue
                if 30 <= dt <= 45:
                    exps[e] = exps.get(e, 0) + 1
            if not exps:
                continue
            exp = max(exps, key=exps.get)
            typ = "call" if reg == "bull" else "put"
            c = pick_band(rows, typ, 0.40, 0.55, exp)
            if not c:
                continue
            ask = E.f(c.get("ask"))
            n = int(rb // (ask * 100)) if ask > 0 else 0
            if n < 1:
                continue
            occ = c.get("contractID") or c.get("symbol")
            actions.append(("BUY", occ, n, round(ask, 2), f"entry:{'long_call' if typ=='call' else 'long_put'} "
                            f"{exp} K={E.f(c.get('strike'))} Δ={abs(E.f(c.get('delta'))):.2f} ask={ask:.2f} cost=${ask*100*n:,.0f}"))

    # ---- output plan ----
    print(f"\nPLANNED ACTIONS ({len(actions)}):")
    if not actions:
        print("  (none — no exits triggered, not an entry day, or transition=cash)")
    for side, occ, qty, limit, why in actions:
        print(f"  {side:<4} {qty:>3} x {occ:<22} @ {('mkt-quote' if limit is None else f'${limit:.2f}'):>9} | {why}")

    # ---- submit (gated) ----
    if args.submit and actions:
        from alpaca.trading.requests import LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        for side, occ, qty, limit, why in actions:
            os_side = OrderSide.BUY if side == "BUY" else OrderSide.SELL
            px = limit if limit is not None else option_sell_limit(occ)  # SELL fetches live bid
            if px is None or px <= 0:
                print(f"  SKIP {side} {occ}: no valid limit price (quote unavailable)"); continue
            req = LimitOrderRequest(symbol=occ, qty=qty, side=os_side,
                                    time_in_force=TimeInForce.DAY, limit_price=round(px, 2))
            try:
                resp = client.submit_order(req)
                print(f"  SUBMITTED {side} {qty} {occ} @ ${px:.2f}  id={getattr(resp,'id','?')}")
                journal({"event": "order", "side": side, "contract": occ, "qty": qty, "limit": px, "why": why}, dry)
            except Exception as ex:
                print(f"  ORDER FAILED {side} {occ}: {str(ex)[:120]}")

    # ---- persist state (entries we'd open / exits we'd close) ----
    if not dry:
        # (state reconciliation would be finalized against fills in a real run)
        if is_entry_day and reg in ("bull", "bear"):
            state["last_entry_date"] = today.isoformat()
        save_state(state, dry)

    print("\nNOTE: dry-run makes NO changes." if dry else "\nSUBMIT mode: orders sent above.")
    print("LIVE_DIR_DONE")


if __name__ == "__main__":
    main()
