#!/usr/bin/env python3
"""volarb_signal.py — daily vol-arb (IV>RV) iron-condor signal emitter.

Independent, monitor-only signal — does NOT trade or touch the deployed trend runtime.
Each run: compute ATM IV (live chain) / trailing-20d realized vol; if IV/RV > threshold (1.4,
the validated knee), flag QUALIFIES and emit the defined-risk iron condor (10Δ short strangle,
5Δ long wings) with strikes/credit/max-loss/breakevens. Writes:
  logs/structure/volarb_<YYYYMMDD>.json  +  volarb_latest.json

Validated (vol_arb_backtest.py): SPY, IV/RV>1.4 -> recent-regime Sharpe 2.16, bounded tail.
Usage:  python volarb_signal.py            # SPY (the validated underlying)
        python volarb_signal.py SPY QQQ --threshold 1.4
"""
from __future__ import annotations
import sys, json, math, argparse, statistics
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dotenv import load_dotenv
load_dotenv(HERE / ".env")
import warnings; warnings.filterwarnings("ignore")
import options as O
import structure_levels as SL
ET = ZoneInfo("America/New_York")
OUT = HERE / "logs" / "structure"


def f(x, d=0.0):
    try: return float(x)
    except: return d


def spot_and_rv(symbol, n=20):
    """Last close (spot) + trailing-n realized vol (annualized) via Alpaca daily (slim-safe)."""
    df = SL._daily_bars(symbol, days=90)
    if df is None or len(df) < n + 1: return None, None
    col = "close" if "close" in df.columns else "Close"
    c = [float(x) for x in df[col]]
    rs = [math.log(c[i] / c[i - 1]) for i in range(len(c) - n, len(c))]
    return c[-1], statistics.pstdev(rs) * math.sqrt(252)


def fetch_chain(symbol):
    """Full realtime chain (BOTH calls and puts) from REALTIME_OPTIONS — av_options_adapter is calls-only."""
    rows = O.get_realtime_options(symbol)
    out = []
    for r in rows:
        typ = str(r.get("type") or r.get("option_type") or r.get("contract_type") or "").lower()
        t = "call" if "call" in typ else ("put" if "put" in typ else None)
        if t is None: continue
        strike = f(r.get("strike") or r.get("strike_price"))
        exp = r.get("expiration") or r.get("expiration_date")
        if not strike or not exp: continue
        out.append({"type": t, "expiration_date": str(exp)[:10], "strike": strike,
                    "bid": f(r.get("bid") or r.get("bid_price")), "ask": f(r.get("ask") or r.get("ask_price")),
                    "delta": f(r.get("delta")), "implied_volatility": f(r.get("implied_volatility"))})
    return out


def pick(chain, typ, tgt_abs_delta, exp):
    best = None; bd = 9
    for r in chain:
        if r.get("type") != typ or r.get("expiration_date", "")[:10] != exp: continue
        if f(r.get("bid")) <= 0: continue
        dl = abs(f(r.get("delta")))
        if dl > 0 and abs(dl - tgt_abs_delta) < bd: best = r; bd = abs(dl - tgt_abs_delta)
    return best


def atm_iv(chain, spot, exp):
    best = None; bd = 1e9
    for r in chain:
        if r.get("expiration_date", "")[:10] != exp: continue
        iv = f(r.get("implied_volatility"))
        if iv <= 0: continue
        dd = abs(f(r.get("strike")) - spot)
        if dd < bd: best = iv; bd = dd
    return best


def signal(symbol, threshold=1.4, dte_lo=25, dte_hi=65, dte_target=45):
    spot, rv = spot_and_rv(symbol)
    out = {"symbol": symbol, "spot": round(spot, 2) if spot else None,
           "rv20": round(rv, 4) if rv else None, "threshold": threshold, "qualifies": False}
    if spot is None or rv is None:
        out["error"] = "no underlying data"; return out
    try:
        chain = fetch_chain(symbol)
    except Exception as e:
        out["error"] = f"chain fetch failed: {str(e)[:80]}"; return out
    if not chain:
        out["error"] = "empty chain (market closed / no realtime data)"; return out
    today = datetime.now(ET).date()
    exps = {}
    for r in chain:
        e = r.get("expiration_date", "")[:10]
        if not e: continue
        try: dt = (date.fromisoformat(e) - today).days
        except: continue
        if dte_lo <= dt <= dte_hi: exps[e] = abs(dt - dte_target)
    if not exps:
        out["error"] = "no expiry in DTE window"; return out
    exp = min(exps, key=exps.get)
    iv = atm_iv(chain, spot, exp)
    if not iv:
        out["error"] = "no ATM IV"; return out
    ratio = iv / rv
    out.update(atm_iv=round(iv, 4), expiry=exp, dte=(date.fromisoformat(exp) - today).days,
               iv_rv_ratio=round(ratio, 3), qualifies=bool(ratio > threshold))
    if not out["qualifies"]:
        out["note"] = f"IV/RV {ratio:.2f} <= {threshold} — premium not rich enough, no trade"
        return out
    # build defined-risk iron condor
    sc = pick(chain, "call", 0.10, exp); lc = pick(chain, "call", 0.05, exp)
    sp = pick(chain, "put", 0.10, exp); lp = pick(chain, "put", 0.05, exp)
    if not (sc and lc and sp and lp):
        out["note"] = "qualifies but condor legs incomplete in chain"; return out
    Kc_s, Kc_l = f(sc["strike"]), f(lc["strike"]); Kp_s, Kp_l = f(sp["strike"]), f(lp["strike"])
    credit = (f(sc["bid"]) - f(lc["ask"])) + (f(sp["bid"]) - f(lp["ask"]))
    maxloss = max(Kc_l - Kc_s, Kp_s - Kp_l) - credit
    out["condor"] = {
        "short_put": {"strike": Kp_s, "delta": round(f(sp["delta"]), 3), "bid": f(sp["bid"])},
        "long_put": {"strike": Kp_l, "delta": round(f(lp["delta"]), 3), "ask": f(lp["ask"])},
        "short_call": {"strike": Kc_s, "delta": round(f(sc["delta"]), 3), "bid": f(sc["bid"])},
        "long_call": {"strike": Kc_l, "delta": round(f(lc["delta"]), 3), "ask": f(lc["ask"])},
        "credit": round(credit, 2), "max_loss": round(maxloss, 2),
        "return_on_risk_pct": round(credit / maxloss * 100, 1) if maxloss > 0 else None,
        "breakeven_low": round(Kp_s - credit, 2), "breakeven_high": round(Kc_s + credit, 2),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbols", nargs="*", default=["SPY"])
    ap.add_argument("--threshold", type=float, default=1.4)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    syms = a.symbols or ["SPY"]
    now = datetime.now(ET); tag = now.strftime("%Y%m%d")
    res = [signal(s, a.threshold) for s in syms]
    payload = {"generated_at": now.isoformat(), "threshold": a.threshold, "signals": res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"volarb_{tag}.json").write_text(json.dumps(payload, indent=2))
    (OUT / "volarb_latest.json").write_text(json.dumps(payload, indent=2))
    if a.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"VOL-ARB SIGNAL  {now:%Y-%m-%d %H:%M %Z}  (IV/RV threshold {a.threshold})")
        for r in res:
            if r.get("error"):
                print(f"  {r['symbol']:<5} ERROR: {r['error']}"); continue
            tagq = "🔔 FIRE " if r["qualifies"] else "  no-go"
            line = f"  {tagq} {r['symbol']:<5} IV {r['atm_iv']*100:.0f}% / RV {r['rv20']*100:.0f}% = {r['iv_rv_ratio']:.2f}"
            if r["qualifies"] and r.get("condor"):
                cd = r["condor"]
                line += (f"  -> CONDOR {r['expiry']} ({r['dte']}d): "
                         f"{cd['long_put']['strike']:.0f}/{cd['short_put']['strike']:.0f}p "
                         f"{cd['short_call']['strike']:.0f}/{cd['long_call']['strike']:.0f}c "
                         f"credit ${cd['credit']} maxloss ${cd['max_loss']} ROR {cd['return_on_risk_pct']}% "
                         f"BE {cd['breakeven_low']}-{cd['breakeven_high']}")
            elif r["qualifies"]:
                line += f"  -> {r.get('note','')}"
            print(line)
        print(f"saved -> logs/structure/volarb_{tag}.json")
    print("VOLARB_SIGNAL_DONE")


if __name__ == "__main__":
    main()
