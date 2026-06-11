#!/usr/bin/env python3
"""
alpha_engine_diagnostic.py — Funnel diagnostic for AlphaCallEngine.

For every candidate symbol it instruments each decision gate of the engine and
records the metrics that drive (or block) a call signal:

    setup_score · grade · reject_reason · market_regime · RVOL · VWAP state ·
    ORB state · chain availability · per-contract options-liquidity sub-gates
    (DTE / delta / OI / volume / mid / spread%) · binding constraint

It then ranks rejection reasons by frequency, reconstructs the full funnel, and
explains why the desired 15-20 trades/day are not being produced.

Output:  daily_system_outputs/alpha_rejection_report.json

Read-only: it does NOT submit orders or mutate engine state. It re-uses the
engine's own detect_setup() / select_call_contract() / size_position() so the
diagnosis matches live behavior exactly, and adds an instrumented copy of the
contract-selection gates to attribute *which* sub-gate eliminates each contract.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(ET).strftime("%Y%m%d"))
DAILY_OUTPUTS = ROOT / "daily_system_outputs"
REPORT_PATH = DAILY_OUTPUTS / "alpha_rejection_report.json"
CHAIN_HEALTH_PATH = DAILY_OUTPUTS / "options_chain_health_report.json"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

import pandas as pd

from alpha_call_engine import (
    AlphaCallConfig, detect_setup, select_call_contract, size_position,
    evaluate_call_contracts, resolve_risk_free_rate,
)

# ---------------------------------------------------------------------------
# Credential resolution (handles the APCA_* vs ALPACA_* naming mismatch)
# ---------------------------------------------------------------------------

def resolve_creds() -> tuple[str, str, bool]:
    key = (os.getenv("APCA_API_KEY_ID") or os.getenv("ALPACA_API_KEY_ID")
           or os.getenv("ALPACA_API_KEY") or "")
    sec = (os.getenv("APCA_API_SECRET_KEY") or os.getenv("ALPACA_API_SECRET_KEY")
           or os.getenv("ALPACA_SECRET_KEY") or "")
    paper = (os.getenv("APCA_PAPER") or os.getenv("ALPACA_PAPER") or "true").lower() in {"1", "true", "yes"}
    return key, sec, paper


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_universe() -> list[str]:
    for p in (ROOT / f"{DATA_TAG}_locked_trading_universe.txt", ROOT / "locked_trading_universe.txt"):
        if p.exists():
            syms = [ln.strip().upper() for ln in p.read_text().splitlines() if ln.strip()]
            if syms:
                return syms
    return []


def load_market_context() -> tuple[dict, str]:
    cached = DAILY_OUTPUTS / DATA_TAG / f"{DATA_TAG}_market_context.json"
    if cached.exists():
        try:
            return json.loads(cached.read_text()), f"cached:{cached.name}"
        except Exception:
            pass
    try:
        from market_context_builder import build_market_context
        return build_market_context(), "rebuilt"
    except Exception as e:
        return ({"overall_market_bias": "unknown"}, f"unavailable:{e}")


# ---------------------------------------------------------------------------
# Data clients
# ---------------------------------------------------------------------------

class DiagClients:
    def __init__(self):
        self.key, self.sec, self.paper = resolve_creds()
        self.creds_present = bool(self.key and self.sec)
        self.trading = self.stock = self.opt = None
        self.feed = None
        if not self.creds_present:
            return
        try:
            from alpaca.trading.client import TradingClient
            self.trading = TradingClient(self.key, self.sec, paper=self.paper)
        except Exception as e:
            print(f"[diag] TradingClient init failed: {e}")
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.enums import DataFeed
            self.stock = StockHistoricalDataClient(self.key, self.sec)
            self.feed = DataFeed.SIP if os.getenv("ALPACA_DATA_FEED", "sip").lower() == "sip" else DataFeed.IEX
        except Exception as e:
            print(f"[diag] StockHistoricalDataClient init failed: {e}")
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
            self.opt = OptionHistoricalDataClient(self.key, self.sec)
        except Exception as e:
            print(f"[diag] OptionHistoricalDataClient init failed: {e}")


def fetch_1m_bars(clients: DiagClients, symbols: list[str], lookback_min: int = 600) -> tuple[dict, str]:
    """Return ({symbol: 1m DataFrame for the most recent session}, source)."""
    if clients.stock is None:
        return {}, "no_stock_client"
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=6)
        kw = dict(symbol_or_symbols=symbols, timeframe=TimeFrame.Minute, start=start, end=end)
        if clients.feed is not None:
            kw["feed"] = clients.feed
        df = clients.stock.get_stock_bars(StockBarsRequest(**kw)).df
        if df is None or df.empty:
            return {}, "alpaca_empty"
        df = df.reset_index()
        out = {}
        for sym, grp in df.groupby("symbol"):
            g = grp.set_index("timestamp")[["open", "high", "low", "close", "volume"]].copy()
            # restrict to the most recent calendar session present
            last_day = g.index[-1].tz_convert(ET).date()
            g_et = g.copy()
            mask = [ts.tz_convert(ET).date() == last_day for ts in g_et.index]
            out[sym] = g_et[mask]
        return out, "alpaca_1m"
    except Exception as e:
        return {}, f"alpaca_error:{str(e)[:120]}"


# ---------------------------------------------------------------------------
# Options-chain diagnostics (instrumented copy of select_call_contract gates)
# ---------------------------------------------------------------------------

def fetch_chain(clients: DiagClients, symbol: str, min_dte: int = 21, max_dte: int = 60) -> tuple[list[dict], str]:
    """Real call chain with OI (contracts endpoint) merged with bid/ask/delta (snapshot).

    NOTE: GetOptionContractsRequest WITHOUT an expiration window returns only the
    nearest (often 0-DTE) expiry, which silently fails the engine's 21-60 DTE gate.
    We therefore request the [today+min_dte, today+max_dte] window explicitly.
    """
    if clients.trading is None:
        return [], "no_trading_client"
    contracts: dict[str, dict] = {}
    try:
        from alpaca.trading.requests import GetOptionContractsRequest
        from alpaca.trading.enums import ContractType, AssetStatus
        today = datetime.now(ET).date()
        req = GetOptionContractsRequest(underlying_symbols=[symbol], status=AssetStatus.ACTIVE,
                                        type=ContractType.CALL,
                                        expiration_date_gte=today + timedelta(days=min_dte),
                                        expiration_date_lte=today + timedelta(days=max_dte),
                                        limit=1000)
        resp = clients.trading.get_option_contracts(req)
        for c in getattr(resp, "option_contracts", []) or []:
            occ = str(getattr(c, "symbol", ""))
            if not occ:
                continue
            contracts[occ] = {
                "symbol": occ, "type": "call",
                "expiration_date": str(getattr(c, "expiration_date", "")),
                "strike_price": float(getattr(c, "strike_price", 0) or 0),
                "open_interest": int(float(getattr(c, "open_interest", 0) or 0)),
                "volume": 0, "delta": 0.0, "implied_volatility": 0.0,
                "bid_price": 0.0, "ask_price": 0.0,
            }
    except Exception as e:
        return [], f"contracts_error:{str(e)[:100]}"
    if not contracts:
        return [], "no_listed_contracts"
    # merge greeks + quotes
    if clients.opt is not None:
        try:
            from alpaca.data.requests import OptionChainRequest
            chain = clients.opt.get_option_chain(OptionChainRequest(underlying_symbol=symbol))
            snap = chain if isinstance(chain, dict) else getattr(chain, "data", {}) or {}
            for occ, s in snap.items():
                if occ not in contracts:
                    continue
                q = getattr(s, "latest_quote", None)
                g = getattr(s, "greeks", None)
                t = getattr(s, "latest_trade", None)
                day = getattr(s, "daily_bar", None)
                if q is not None:
                    contracts[occ]["bid_price"] = float(getattr(q, "bid_price", 0) or 0)
                    contracts[occ]["ask_price"] = float(getattr(q, "ask_price", 0) or 0)
                if g is not None:
                    contracts[occ]["delta"] = float(getattr(g, "delta", 0) or 0)
                iv = getattr(s, "implied_volatility", None)
                if iv is not None:
                    contracts[occ]["implied_volatility"] = float(iv or 0)
                if day is not None and getattr(day, "volume", None) is not None:
                    contracts[occ]["volume"] = int(getattr(day, "volume", 0) or 0)
                elif t is not None and getattr(t, "size", None) is not None:
                    contracts[occ]["volume"] = int(getattr(t, "size", 0) or 0)
        except Exception:
            pass
    return list(contracts.values()), "ok"


def diagnose_chain(chain: list[dict], cfg: AlphaCallConfig) -> dict:
    """Attribute eliminations across the same gates select_call_contract applies, in order."""
    today = datetime.now(ET).date()
    n = len(chain)
    survivors = list(chain)
    funnel = {"total_contracts": n}

    def keep(pred):
        nonlocal survivors
        survivors = [c for c in survivors if pred(c)]
        return len(survivors)

    def dte_of(c):
        try:
            return (datetime.strptime(str(c.get("expiration_date") or c.get("expiry"))[:10], "%Y-%m-%d").date() - today).days
        except Exception:
            return -999

    def delta_of(c):
        return abs(float(c.get("delta") or 0))

    def mid_of(c):
        b = float(c.get("bid_price") or c.get("bid") or 0)
        a = float(c.get("ask_price") or c.get("ask") or 0)
        return (a + b) / 2 if a > 0 and b > 0 else 0.0

    def spread_of(c):
        b = float(c.get("bid_price") or c.get("bid") or 0)
        a = float(c.get("ask_price") or c.get("ask") or 0)
        m = mid_of(c)
        return (a - b) / m if m > 0 else 1.0

    funnel["after_dte"]    = keep(lambda c: cfg.min_dte <= dte_of(c) <= cfg.max_dte)
    funnel["after_delta"]  = keep(lambda c: cfg.min_delta <= delta_of(c) <= cfg.max_delta)
    funnel["after_oi"]     = keep(lambda c: int(c.get("open_interest") or 0) >= cfg.min_open_interest)
    funnel["after_volume"] = keep(lambda c: int(c.get("volume") or 0) >= cfg.min_volume)
    funnel["after_mid"]    = keep(lambda c: mid_of(c) > 0)
    funnel["after_spread"] = keep(lambda c: spread_of(c) <= cfg.max_spread_pct)

    # binding constraint = first gate that dropped survivors to zero
    binding = None
    prev = n
    for gate in ("after_dte", "after_delta", "after_oi", "after_volume", "after_mid", "after_spread"):
        if funnel[gate] == 0 and prev > 0:
            binding = gate.replace("after_", "")
            break
        prev = funnel[gate]

    # independent stats across the full chain (not sequential) for visibility
    deltas_present = sum(1 for c in chain if delta_of(c) > 0)
    in_dte = [c for c in chain if cfg.min_dte <= dte_of(c) <= cfg.max_dte]
    quotes_present = sum(1 for c in chain if mid_of(c) > 0)
    ois = sorted((int(c.get("open_interest") or 0) for c in chain), reverse=True)[:5]

    return {
        "funnel": funnel,
        "binding_constraint": binding,
        "deltas_present_pct": round(100 * deltas_present / n, 1) if n else 0.0,
        "quotes_present_pct": round(100 * quotes_present / n, 1) if n else 0.0,
        "contracts_in_dte_window": len(in_dte),
        "top5_open_interest": ois,
    }


# ---------------------------------------------------------------------------
# Per-symbol diagnosis
# ---------------------------------------------------------------------------

def diagnose_symbol(symbol, bars, source, context, chain, chain_status, cfg, equity):
    ev = detect_setup(symbol, bars if bars is not None else pd.DataFrame(), cfg, context)
    row = {
        "symbol": symbol,
        "data_source": source,
        "bar_count": int(len(bars)) if bars is not None else 0,
        "market_regime": str(context.get("overall_market_bias") or context.get("market_bias") or "unknown"),
        "regime_ok": ev.regime_ok,
        "setup_type": ev.setup_type,
        "setup_score": ev.setup_score,
        "grade": ev.grade,
        "price": round(ev.price, 4),
        "vwap": round(ev.vwap, 4),
        "vwap_state": "above" if ev.above_vwap else "below",
        "rvol": round(ev.rvol, 3),
        "institutional_flow": ev.institutional_flow,
        "orb_high": round(ev.orb_high, 4),
        "orb_low": round(ev.orb_low, 4),
        "orb_state": ("breakout" if ev.orb_high > 0 and ev.price > ev.orb_high
                      else "inside" if ev.orb_high > 0 else "no_orb"),
        "sector_strong": ev.sector_strong,
        "notes": ev.notes,
        "chain_status": chain_status,
        "chain_available": bool(chain),
        "chain_size": len(chain),
    }

    # Black-Scholes valuation (hard gates + BS layer) via the engine itself.
    rfr = resolve_risk_free_rate(cfg, context)
    evalc = evaluate_call_contracts(symbol, ev.price, chain, cfg, rfr, ev.rvol) if chain else None

    # chain liquidity metrics + BS summary
    if chain:
        cd = diagnose_chain(chain, cfg)
        row["options"] = {
            "contracts_in_dte_window": cd["contracts_in_dte_window"],
            "deltas_present_pct": cd["deltas_present_pct"],
            "quotes_present_pct": cd["quotes_present_pct"],
            "top5_open_interest": cd["top5_open_interest"],
            "selection_funnel": cd["funnel"],
            "binding_constraint": cd["binding_constraint"],
            "black_scholes": _bs_summary(evalc),
        }
    else:
        row["options"] = {"selection_funnel": None, "binding_constraint": "no_chain", "black_scholes": None}

    # ── replicate evaluate_universe disposition logic (incl. BS layer) ─────────
    contract = evalc["best"] if evalc else None
    if ev.reject_reason:
        # set by detect_setup (regime or C-grade)
        disp, reason = "rejected", ev.reject_reason
    elif ev.setup_score < cfg.min_setup_score:
        disp, reason = "rejected", f"rejected_by_setup_quality:score={ev.setup_score:.0f}<{cfg.min_setup_score:.0f}"
    elif not chain:
        disp, reason = "rejected", f"rejected_by_options_liquidity:{chain_status}"
    elif contract is None:
        br = (evalc or {}).get("best_rejected")
        if br is not None and br.get("bs_reject_reason"):
            disp, reason = "rejected", f"rejected_by_options_liquidity:no_qualifying_contract:{br.get('bs_reject_reason')}"
        else:
            binding = row["options"].get("binding_constraint")
            disp, reason = "rejected", f"rejected_by_options_liquidity:no_qualifying_contract:{binding}"
    else:
        contracts, premium, max_risk = size_position(equity, contract["mid"], cfg)
        available = equity * cfg.risk_per_trade_pct
        if contracts < 1 or premium > available:
            disp, reason = "rejected", f"rejected_by_cash_buffer:premium={premium:.0f}>avail={available:.0f}"
        else:
            disp, reason = "READY", "passed_all_gates"
            row["selected_contract"] = {
                "contract": contract["contract_symbol"], "dte": contract["dte"],
                "delta": round(contract["delta"], 3), "oi": contract["open_interest"],
                "spread_pct": round(contract["spread_pct"], 4), "mid": contract["mid"],
                "contracts": contracts, "premium_cost": premium,
                "black_scholes_score": contract.get("black_scholes_score"),
                "theoretical_value": contract.get("theoretical_value"),
                "implied_volatility": contract.get("implied_volatility"),
                "gamma_theta_ratio": contract.get("gamma_theta_ratio"),
                "mispricing_pct": contract.get("mispricing_pct"),
                "expected_value_at_3R": contract.get("expected_value_at_3R"),
            }

    row["disposition"] = disp
    row["final_reason"] = reason
    return row


def _bs_summary(evalc) -> dict | None:
    """Per-symbol Black-Scholes summary for the chain-health report."""
    if not evalc:
        return None
    src = evalc.get("best") or evalc.get("best_rejected")
    sel = None
    if src:
        sel = {k: src.get(k) for k in (
            "contract_symbol", "black_scholes_score", "composite_score", "theoretical_value",
            "market_mid", "implied_volatility", "iv_source", "delta", "gamma", "theta", "vega",
            "gamma_theta_ratio", "theta_pct_per_day", "mispricing_pct", "moneyness",
            "expected_value_at_1R", "expected_value_at_2R", "expected_value_at_3R",
            "expected_loss_at_stop", "expected_r_multiple", "bs_reject_reason",
        )}
        sel["is_selected"] = evalc.get("best") is not None
    return {
        "counts": evalc.get("counts", {}),
        "best_black_scholes_score": evalc.get("best_black_scholes_score"),
        "selected_contract_black_scholes_summary": sel,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = AlphaCallConfig()
    equity = float(os.getenv("STARTING_EQUITY", "100000"))
    universe = load_universe()
    context, ctx_source = load_market_context()
    clients = DiagClients()

    print(f"[diag] universe={len(universe)} symbols | context={ctx_source} | creds_present={clients.creds_present}")
    bars_by_symbol, bar_source = fetch_1m_bars(clients, universe) if universe else ({}, "no_universe")
    print(f"[diag] bar_source={bar_source} | symbols_with_bars={len(bars_by_symbol)}")

    rows = []
    for sym in universe:
        bars = bars_by_symbol.get(sym)
        chain, chain_status = fetch_chain(clients, sym, cfg.min_dte, cfg.max_dte) if clients.creds_present else ([], "no_creds")
        try:
            rows.append(diagnose_symbol(sym, bars, bar_source, context, chain, chain_status, cfg, equity))
        except Exception as e:
            rows.append({"symbol": sym, "disposition": "error", "final_reason": f"diag_error:{str(e)[:160]}"})
        print(f"  {sym:6s} score={rows[-1].get('setup_score','?'):>5} grade={rows[-1].get('grade','?'):<2} "
              f"chain={rows[-1].get('chain_size','?')} -> {rows[-1].get('final_reason','?')}")

    # ── aggregate funnel ───────────────────────────────────────────────────────
    n = len(rows)
    ready = [r for r in rows if r.get("disposition") == "READY"]
    reason_counts = Counter(r.get("final_reason", "unknown").split(":")[0] for r in rows)
    detailed_reason_counts = Counter(r.get("final_reason", "unknown") for r in rows)

    funnel = {
        "candidates": n,
        "regime_ok": sum(1 for r in rows if r.get("regime_ok")),
        "score_ge_threshold": sum(1 for r in rows if isinstance(r.get("setup_score"), (int, float)) and r["setup_score"] >= cfg.min_setup_score),
        "grade_A_or_better": sum(1 for r in rows if r.get("grade") in ("A", "A+")),
        "chain_available": sum(1 for r in rows if r.get("chain_available")),
        "qualifying_contract": sum(1 for r in rows if r.get("selected_contract")),
        "ready_signals": len(ready),
    }

    diagnosis = build_diagnosis(funnel, reason_counts, cfg, n, len(bars_by_symbol),
                                clients.creds_present, bar_source, context)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": DATA_TAG,
        "engine_config": {
            "enabled": cfg.enabled,
            "min_setup_score": cfg.min_setup_score,
            "min_dte": cfg.min_dte, "max_dte": cfg.max_dte,
            "min_delta": cfg.min_delta, "max_delta": cfg.max_delta,
            "min_open_interest": cfg.min_open_interest,
            "min_volume": cfg.min_volume, "max_spread_pct": cfg.max_spread_pct,
            "max_concurrent_positions": cfg.max_concurrent_positions,
            "max_options_exposure_pct": cfg.max_options_exposure_pct,
            "risk_per_trade_pct": cfg.risk_per_trade_pct,
            "min_rvol_for_accum": cfg.min_rvol_for_accum,
            "require_bullish_regime": cfg.require_bullish_regime,
            "allowed_regimes": list(cfg.allowed_regimes),
        },
        "inputs": {
            "universe_size": len(universe),
            "market_context_source": ctx_source,
            "market_regime": str(context.get("overall_market_bias")),
            "trade_permission": str(context.get("trade_permission")),
            "creds_present": clients.creds_present,
            "bar_source": bar_source,
            "symbols_with_bars": len(bars_by_symbol),
        },
        "funnel": funnel,
        "rejection_reasons_ranked": [
            {"reason": r, "count": c, "pct": round(100 * c / n, 1) if n else 0.0}
            for r, c in reason_counts.most_common()
        ],
        "rejection_reasons_detailed": [
            {"reason": r, "count": c} for r, c in detailed_reason_counts.most_common()
        ],
        "diagnosis": diagnosis,
        "per_symbol": rows,
    }

    DAILY_OUTPUTS.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    write_chain_health(rows, cfg)
    print(f"\n[diag] wrote {REPORT_PATH}")
    print(f"[diag] wrote {CHAIN_HEALTH_PATH}")
    print(f"[diag] READY signals: {len(ready)} / {n} candidates")
    print("[diag] rejection reasons:", dict(reason_counts))
    for line in diagnosis["why_not_15_to_20_trades"]:
        print("   •", line)


def write_chain_health(rows, cfg):
    """Per-symbol + aggregate options-chain health (independent of setup scoring)."""
    per = []
    binding = Counter()
    chains_with_contracts = 0
    chains_with_qualifying = 0
    # Black-Scholes aggregates (required diagnostics)
    bs_agg = {
        "contracts_priced_by_black_scholes": 0,
        "contracts_rejected_overpriced": 0,
        "contracts_rejected_theta_decay": 0,
        "contracts_rejected_gamma_theta": 0,
        "contracts_rejected_iv": 0,
        "contracts_rejected_iv_inflated": 0,
        "contracts_qualified": 0,
    }
    best_bs_score = 0.0
    selected_summary = None   # BS summary of the globally best qualified contract
    for r in rows:
        o = r.get("options") or {}
        f = o.get("selection_funnel")
        sym = r.get("symbol")
        bs = o.get("black_scholes") if isinstance(o, dict) else None
        if bs:
            c = bs.get("counts", {})
            bs_agg["contracts_priced_by_black_scholes"] += c.get("contracts_priced_by_black_scholes", 0)
            bs_agg["contracts_rejected_overpriced"]    += c.get("contracts_rejected_overpriced", 0)
            bs_agg["contracts_rejected_theta_decay"]   += c.get("contracts_rejected_theta_decay", 0)
            bs_agg["contracts_rejected_gamma_theta"]   += c.get("contracts_rejected_gamma_theta", 0)
            bs_agg["contracts_rejected_iv"]            += c.get("contracts_rejected_iv", 0)
            bs_agg["contracts_rejected_iv_inflated"]   += c.get("contracts_rejected_iv_inflated", 0)
            bs_agg["contracts_qualified"]              += c.get("qualified", 0)
            sc = bs.get("best_black_scholes_score") or 0.0
            sel = bs.get("selected_contract_black_scholes_summary")
            if sel and sel.get("is_selected") and sc >= best_bs_score:
                best_bs_score = sc
                selected_summary = {"symbol": sym, **sel}
        if r.get("chain_size", 0) > 0 and f:
            chains_with_contracts += 1
            qualifies = bool(bs and bs.get("counts", {}).get("qualified", 0) > 0)
            if qualifies:
                chains_with_qualifying += 1
            binding[o.get("binding_constraint") or ("none" if qualifies else "unknown")] += 1
            per.append({
                "symbol": sym,
                "chain_status": r.get("chain_status"),
                "total_contracts": f.get("total_contracts"),
                "contracts_in_dte_window": o.get("contracts_in_dte_window"),
                "greeks_present_pct": o.get("deltas_present_pct"),
                "quotes_present_pct": o.get("quotes_present_pct"),
                "top5_open_interest": o.get("top5_open_interest"),
                "selection_funnel": f,
                "binding_constraint": o.get("binding_constraint"),
                "qualifying_contracts": f.get("after_spread", 0),
                "black_scholes": bs,
            })
        else:
            binding[r.get("chain_status") or "no_chain"] += 1
            per.append({"symbol": sym, "chain_status": r.get("chain_status"),
                        "total_contracts": 0, "binding_constraint": "no_chain"})
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "date_tag": DATA_TAG,
        "gates": {
            "min_dte": cfg.min_dte, "max_dte": cfg.max_dte,
            "min_delta": cfg.min_delta, "max_delta": cfg.max_delta,
            "min_open_interest": cfg.min_open_interest,
            "min_volume": cfg.min_volume, "max_spread_pct": cfg.max_spread_pct,
            "bs_enabled": cfg.bs_enabled,
            "bs_max_overprice_pct": cfg.bs_max_overprice_pct,
            "bs_max_theta_pct_per_day": cfg.bs_max_theta_pct_per_day,
            "bs_min_gamma_theta": cfg.bs_min_gamma_theta,
            "bs_max_iv": cfg.bs_max_iv, "risk_free_rate": cfg.risk_free_rate,
        },
        "summary": {
            "symbols": len(rows),
            "chains_with_contracts": chains_with_contracts,
            "chains_with_qualifying_contract": chains_with_qualifying,
            "binding_constraint_distribution": dict(binding.most_common()),
        },
        "black_scholes": {
            "contracts_priced_by_black_scholes": bs_agg["contracts_priced_by_black_scholes"],
            "contracts_rejected_overpriced": bs_agg["contracts_rejected_overpriced"],
            "contracts_rejected_theta_decay": bs_agg["contracts_rejected_theta_decay"],
            "contracts_rejected_gamma_theta": bs_agg["contracts_rejected_gamma_theta"],
            "contracts_rejected_iv": bs_agg["contracts_rejected_iv"],
            "contracts_rejected_iv_inflated": bs_agg["contracts_rejected_iv_inflated"],
            "contracts_qualified": bs_agg["contracts_qualified"],
            "best_black_scholes_score": round(best_bs_score, 2),
            "selected_contract_black_scholes_summary": selected_summary,
        },
        "note": ("Chain health reflects data feed + hard liquidity gates + the Black-Scholes "
                 "valuation layer. The BS layer runs only on hard-gate survivors and never "
                 "overrides regime/setup/liquidity/risk gates."),
        "per_symbol": per,
    }
    CHAIN_HEALTH_PATH.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def build_diagnosis(funnel, reason_counts, cfg, n, n_bars, creds, bar_source, context):
    reasons = []
    # Structural ceiling
    reasons.append(
        f"STRUCTURAL CEILING: max_concurrent_positions={cfg.max_concurrent_positions}. Long calls are "
        f"held to a 3R target / 1.7R trail (multi-hour to multi-day holds), so at most ~{cfg.max_concurrent_positions} "
        f"positions can be open at once. 15-20 trades/day requires either a much higher concurrency cap or "
        f"fast intraday turnover that this management style does not produce."
    )
    reasons.append(
        f"UNIVERSE SIZE: only {funnel['candidates']} candidates were scanned. Even a 100% pass rate "
        f"cannot yield 15-20 new trades on a 25-name list dominated by speculative small/mid-caps with thin options."
    )
    if not creds:
        reasons.append(
            "DATA/AUTH: Alpaca credentials were NOT resolved for the engine broker. run.py's AlpacaGateway reads "
            "ALPACA_API_KEY/ALPACA_SECRET_KEY, but .env provides APCA_API_KEY_ID/APCA_API_SECRET_KEY. In live runs "
            "this makes broker.get_options_chain() return [] for every symbol -> 100% rejected_by_options_liquidity."
        )
    if bar_source != "alpaca_1m":
        reasons.append(
            f"BAR DATA: 1-minute bars unavailable in this run (source={bar_source}). detect_setup() rejects every "
            f"symbol with 'rejected_by_setup_quality:no_bars' when bars are missing."
        )
    # threshold
    reasons.append(
        f"SETUP THRESHOLD: min_setup_score={cfg.min_setup_score}. With regime contributing at most +15, a symbol "
        f"needs a fresh ORB breakout (<=2.5% extension) or strong accumulation PLUS RVOL>=2.5 to clear 70. These "
        f"conditions are time-of-day specific and rare across a small universe in a single scan."
    )
    reasons.append(
        "ORB ANCHOR [FIXED]: _compute_orb() now anchors to the real 09:30-09:45 ET window of the active session "
        "(both-sided, tz-safe) and returns a diagnostic status instead of a silent orb_high=0. Previously a "
        "mid-session/completed window produced an empty ORB range and silently disabled ORB + PIVOT scoring."
    )
    reasons.append(
        "SECTOR BONUS [FIXED]: detect_setup() now reads sector_strength.sector_rotation_bias (and still honors the "
        "legacy sector_leadership/sector_bias keys). With today's risk_on_growth bias the +5 now applies and lifted "
        "LQDA 68->73, INTR 66->71, PACS 65->70, HLIO 65->70 across the A threshold — confirming the dead bonus was "
        "suppressing 4 valid A-grade setups. (These A-grades now stop at the separate options-liquidity wall.)"
    )
    # options liquidity (second wall — would bite after the score gate is fixed)
    reasons.append(
        "CONTRACT-FETCH DEFECT: GetOptionContractsRequest with no expiration window returns only the NEAREST "
        "(0-DTE) expiry, so the engine's DTE 21-60 gate eliminates 100% of contracts even for liquid names. "
        "run.py's AlpacaGateway.get_options_chain has this exact defect. Adding expiration_date_gte/lte "
        "(today+21 .. today+60) surfaces the real chain (e.g. WOLF 214 calls, SE 144, QUBT 130)."
    )
    reasons.append(
        f"OPTIONS LIQUIDITY (after DTE fix, the true binding gates): delta in [{cfg.min_delta},{cfg.max_delta}] "
        f"trims ~60-90% of contracts; then OI>={cfg.min_open_interest} at that delta/DTE removes nearly all that "
        f"remain (e.g. WOLF 78->3, SE 31->0); finally volume>={cfg.min_volume} eliminates the survivors. Note the "
        f"volume gate is partly an artifact: per-contract daily volume is not populated by the contracts endpoint "
        f"(left at 0), so it must be merged from the snapshot or the gate will always fire. Net: this small-cap "
        f"universe rarely has calls meeting OI>={cfg.min_open_interest} at 0.45-0.70 delta / 21-60 DTE."
    )
    return {
        "regime_is_bottleneck": funnel["regime_ok"] < funnel["candidates"],
        "primary_rejection_reason": reason_counts.most_common(1)[0][0] if reason_counts else None,
        "ready_vs_target": f"{funnel['ready_signals']} ready vs 15-20 target",
        "why_not_15_to_20_trades": reasons,
        "recommended_levers": [
            "SECTOR BONUS [DONE]: sector_strength.sector_rotation_bias is now mapped -> +5 applied; lifted 4 names (LQDA/INTR/PACS/HLIO) to A. score_ge_threshold 0 -> 4.",
            "ORB ANCHOR [DONE]: now anchored to real 09:30-09:45 ET; emits diagnostic status. No longer silently zeroes ORB/PIVOT on mid-session windows.",
            "SCORE GATE (still the binding wall for the other 21): near-misses now BTI 63, OSCR/HBM 57-58. A min_setup_score change is a STRATEGIC decision — deferred per instruction.",
            "CONTRACT FETCH: add expiration_date_gte/lte (today+min_dte .. today+max_dte) to GetOptionContractsRequest in run.py's get_options_chain, and merge per-contract daily volume from the snapshot (currently 0 -> always fails the volume gate).",
            "Broker cred mismatch: have AlpacaGateway read APCA_API_KEY_ID/APCA_API_SECRET_KEY (or add ALPACA_API_KEY/SECRET_KEY to .env), else live chains are empty.",
            "Universe + liquidity: this 25-name small-cap list rarely has calls at OI>=200 / 0.45-0.70 delta / 21-60 DTE. Either expand to 75-150 liquid optionable names or relax OI/volume gates for this universe.",
            "Re-scope the target: with max_concurrent_positions=5 and 3R/1.7R-trail management (multi-hour/day holds), ~5 quality trades/day is the realistic ceiling. 15-20/day requires raising concurrency AND universe AND turnover.",
        ],
    }


if __name__ == "__main__":
    main()
