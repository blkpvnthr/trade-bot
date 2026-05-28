"""
scripts/analyze_execution_alpha_blocks.py

Identify high-quality setups blocked by execution_alpha_buy_blocked and compute
counterfactual outcomes using intraday 1-minute bars.

Analysis only — does NOT change any thresholds, orders, or trading state.

Default filters (all must pass):
  stream_confirmed=True, pivot_buy_signal=True,
  trade_confidence >= 0.8, risk_reward_ratio >= 2.0, rel_volume >= 1.2

Outputs:
  daily_system_outputs/execution_alpha_block_analysis.json
  daily_system_outputs/execution_alpha_block_analysis.csv

Usage:
  python scripts/analyze_execution_alpha_blocks.py
  python scripts/analyze_execution_alpha_blocks.py --date 20260521
  python scripts/analyze_execution_alpha_blocks.py --strategy pivot_points --min-confidence 0.75
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")

try:
    import pandas as pd
except ImportError:
    print("[error] pandas not installed. Run: pip install pandas")
    sys.exit(1)


def _load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _parse_ts(ts_str: str) -> datetime:
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def _load_bars(date_tag: str) -> pd.DataFrame | None:
    bars_path = PROJECT_ROOT / "system-data" / date_tag / f"{date_tag}_intraday_bars_1m.csv"
    if not bars_path.exists():
        return None
    try:
        df = pd.read_csv(bars_path, parse_dates=["timestamp"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception as exc:
        print(f"[warn] could not load bars: {exc}")
        return None


def _compute_mfe_mae(
    bars: pd.DataFrame,
    symbol: str,
    entry_ts: datetime,
    entry_price: float,
    stop_price: float,
    target_price: float,
    horizons: list[int],
) -> dict:
    result: dict = {"bars_available": False}
    sym_bars = bars[bars["symbol"] == symbol].copy()
    if sym_bars.empty:
        return result

    entry_ts_utc = entry_ts.astimezone(timezone.utc)
    # Use bars starting from the entry bar onwards
    forward_bars = sym_bars[sym_bars["timestamp"] >= entry_ts_utc].sort_values("timestamp").reset_index(drop=True)
    if forward_bars.empty:
        return result

    result["bars_available"] = True
    target_hit = False
    stop_hit = False
    target_hit_first: bool | None = None

    for h in horizons:
        window = forward_bars[forward_bars["timestamp"] <= entry_ts_utc + timedelta(minutes=h)]
        if window.empty:
            result[f"mfe_{h}m"] = None
            result[f"mae_{h}m"] = None
            continue
        high_max = window["high"].max()
        low_min = window["low"].min()
        mfe = round((high_max - entry_price) / entry_price * 100, 4) if entry_price > 0 else None
        mae = round((low_min - entry_price) / entry_price * 100, 4) if entry_price > 0 else None
        result[f"mfe_{h}m"] = mfe
        result[f"mae_{h}m"] = mae

        if h == max(horizons):
            # Check if target or stop was touched in this window
            if target_price > entry_price:
                target_hit = high_max >= target_price
                stop_hit = low_min <= stop_price
            else:
                target_hit = low_min <= target_price
                stop_hit = high_max >= stop_price

            # Determine which bar first crossed target vs stop
            if target_hit or stop_hit:
                for _, bar in window.iterrows():
                    t_hit = bar["high"] >= target_price if target_price > entry_price else bar["low"] <= target_price
                    s_hit = bar["low"] <= stop_price if stop_price < entry_price else bar["high"] >= stop_price
                    if t_hit and s_hit:
                        target_hit_first = None  # same bar — ambiguous
                        break
                    elif t_hit:
                        target_hit_first = True
                        break
                    elif s_hit:
                        target_hit_first = False
                        break

    result["target_hit"] = target_hit
    result["stop_hit"] = stop_hit
    result["target_hit_first"] = target_hit_first
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze execution-alpha blocked setups")
    parser.add_argument("--date", default="", help="YYYYMMDD (default: today ET)")
    parser.add_argument("--strategy", default=os.getenv("STRATEGY_ID", "tux_mtf_original"))
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--min-rr", type=float, default=2.0, help="Min risk/reward ratio")
    parser.add_argument("--min-rvol", type=float, default=1.2, help="Min relative volume")
    args = parser.parse_args()

    date_tag = args.date or os.getenv("RL_DATA_DATE", "") or datetime.now(ET).strftime("%Y%m%d")
    strategy_id = args.strategy

    events_path = PROJECT_ROOT / "logs" / "execution" / strategy_id / f"{date_tag}_execution_events.jsonl"
    if not events_path.exists():
        print(f"[error] events file not found: {events_path}")
        return 1

    print(f"[analyze] date={date_tag} strategy={strategy_id}")
    print(f"[analyze] events={events_path}")

    events = _load_events(events_path)

    # Extract pivot_buy_orders_generated rows keyed by (symbol, cycle_ts)
    generated_map: dict[str, list[dict]] = {}
    for ev in events:
        if ev.get("event") != "pivot_buy_orders_generated":
            continue
        ev_ts = _parse_ts(ev["timestamp_utc"])
        for row in ev.get("payload", {}).get("rows", []):
            sym = row.get("symbol", "")
            if sym not in generated_map:
                generated_map[sym] = []
            generated_map[sym].append({**row, "_ts": ev_ts})

    # Extract execution_alpha_buy_blocked events
    blocked_events = [
        {"_ts": _parse_ts(ev["timestamp_utc"]), **ev.get("payload", {})}
        for ev in events
        if ev.get("event") == "execution_alpha_buy_blocked"
    ]

    print(f"[analyze] pivot_buy rows found: {sum(len(v) for v in generated_map.values())}")
    print(f"[analyze] blocked events found: {len(blocked_events)}")

    # Load bars for counterfactual analysis
    bars = _load_bars(date_tag)
    if bars is None:
        print("[warn] intraday bars not available — MFE/MAE will be null")

    horizons = [5, 15, 30, 60]
    records: list[dict] = []

    for blocked in blocked_events:
        sym = blocked.get("symbol", "")
        blocked_ts = blocked["_ts"]
        exec_score = float(blocked.get("execution_alpha_score") or 0)
        threshold = float(blocked.get("threshold") or 0)

        # Join with most recent generated row for this symbol within 5 seconds
        candidates = generated_map.get(sym, [])
        matched: dict | None = None
        for gen in reversed(candidates):
            delta = abs((blocked_ts - gen["_ts"]).total_seconds())
            if delta <= 5.0:
                matched = gen
                break

        if matched is None:
            continue

        # Apply filters
        stream_confirmed = bool(matched.get("stream_confirmed"))
        pivot_buy_signal = bool(matched.get("pivot_buy_signal"))
        trade_confidence = float(matched.get("trade_confidence") or 0)
        rr = float(matched.get("risk_reward_ratio") or 0)
        rel_volume = float(matched.get("rel_volume") or 0)

        if not stream_confirmed:
            continue
        if not pivot_buy_signal:
            continue
        if trade_confidence < args.min_confidence:
            continue
        if rr < args.min_rr:
            continue
        if rel_volume < args.min_rvol:
            continue

        entry_price = float(matched.get("entry_price") or matched.get("price") or 0)
        stop_price = float(matched.get("stop_price") or 0)
        target_price = float(matched.get("target_price") or 0)

        record: dict = {
            "symbol": sym,
            "timestamp_et": blocked_ts.astimezone(ET).isoformat(),
            "execution_alpha_score": round(exec_score, 4),
            "threshold": round(threshold, 4),
            "score_deficit": round(threshold - exec_score, 4),
            "trade_confidence": round(trade_confidence, 4),
            "risk_reward_ratio": round(rr, 4),
            "rel_volume": round(rel_volume, 4),
            "setup_reason": str(matched.get("entry_type") or matched.get("reason") or ""),
            "entry_price": round(entry_price, 4),
            "stop_price": round(stop_price, 4),
            "target_price": round(target_price, 4),
            "stop_risk_pct": round((entry_price - stop_price) / entry_price * 100, 4) if entry_price > 0 else None,
            "target_reward_pct": round((target_price - entry_price) / entry_price * 100, 4) if entry_price > 0 else None,
        }

        if bars is not None and entry_price > 0:
            mfe_mae = _compute_mfe_mae(bars, sym, blocked_ts, entry_price, stop_price, target_price, horizons)
            record.update(mfe_mae)
        else:
            record["bars_available"] = False
            for h in horizons:
                record[f"mfe_{h}m"] = None
                record[f"mae_{h}m"] = None
            record["target_hit"] = None
            record["stop_hit"] = None
            record["target_hit_first"] = None

        records.append(record)

    print(f"[analyze] high-quality blocked setups (after filter): {len(records)}")

    daily_out = PROJECT_ROOT / "daily_system_outputs"
    daily_out.mkdir(parents=True, exist_ok=True)

    json_path = daily_out / "execution_alpha_block_analysis.json"
    csv_path = daily_out / "execution_alpha_block_analysis.csv"

    output = {
        "date": date_tag,
        "strategy_id": strategy_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "stream_confirmed": True,
            "pivot_buy_signal": True,
            "min_trade_confidence": args.min_confidence,
            "min_risk_reward_ratio": args.min_rr,
            "min_rel_volume": args.min_rvol,
        },
        "total_blocked": len(blocked_events),
        "high_quality_blocked": len(records),
        "setups": records,
    }
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    if records:
        df = pd.DataFrame(records)
        df.to_csv(csv_path, index=False)
        print(f"\n{'Symbol':<8} {'Score':>6} {'Thresh':>6} {'Deficit':>7} {'RR':>5} {'RVol':>5} {'Target?':>7} {'Stop?':>6}")
        print("-" * 60)
        for r in records:
            print(
                f"{r['symbol']:<8} {r['execution_alpha_score']:>6.4f} {r['threshold']:>6.4f} "
                f"{r['score_deficit']:>7.4f} {r['risk_reward_ratio']:>5.2f} {r['rel_volume']:>5.2f} "
                f"{str(r.get('target_hit','?')):>7} {str(r.get('stop_hit','?')):>6}"
            )
    else:
        pd.DataFrame().to_csv(csv_path, index=False)

    print(f"\n[analyze] written to {json_path}")
    print(f"[analyze] written to {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
