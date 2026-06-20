"""Organize raw TradingView trade exports into a canonical, machine-ingestible
structure. Outcome (win/loss/flat) is NEVER taken from folder names -- it is
derived from the Net PnL columns inside each CSV. The WINS/LOSSES/FLAT folder
label is recorded for traceability only.

Canonical layout:
    {dest}/{strategy_id}/{symbol}/{timeframe}/
        {strategy_id}__{symbol}__{timeframe}__{date_range}.csv

Pure functions here; the CLI in scripts/organize_tradingview_exports.py wires
them with --dry-run (default) / --apply / --move.
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path

import pandas as pd

from .tv_import import parse_tv_csv

LABELS = {"WINS", "LOSSES", "FLAT"}
EXCHANGES = {"NASDAQ", "NYSE", "NYSEARCA", "ARCA", "AMEX", "BATS", "OTC",
             "OTCMKTS", "TVC", "CBOE", "CME", "SP", "DJ", "FX", "OANDA",
             "BINANCE", "COINBASE", "PYTH", "FOREXCOM", "CRYPTO"}
_STOP = LABELS | EXCHANGES | {"WIN", "LOSS", "FLAT"}


# -- normalization / inference --------------------------------------------
def normalize_strategy_id(name: str) -> str:
    """Lowercase snake_case id. 'Linear-Regression' -> 'linear_regression'."""
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def infer_symbol(stem: str):
    """Infer a ticker from a filename stem, else None. Handles
    EXCHANGE_TICKER, a token before a trailing date, or a bare ticker token."""
    # 1) token right before a trailing YYYY-MM-DD (the dominant TV pattern here)
    m = re.search(r"[_-]([A-Z0-9.]{1,6})_(\d{4}-\d{2}-\d{2})$", stem)
    if m:
        return m.group(1)
    # 2) EXCHANGE_TICKER anywhere
    for ex in EXCHANGES:
        m = re.search(rf"\b{ex}[_-]([A-Z0-9.]{{1,6}})\b", stem)
        if m:
            return m.group(1)
    # 3) a bare uppercase ticker-looking token (prefer the last one)
    for tok in reversed(re.split(r"[^A-Za-z0-9.]+", stem)):
        if re.fullmatch(r"[A-Z]{1,5}", tok) and tok not in _STOP:
            return tok
    return None


def _tf_from_filename(stem: str):
    s = stem.lower()
    pats = [("1Day", r"1day|1d|daily"), ("1Hour", r"1hour|1h|60min|60m"),
            ("15Min", r"15min|15m"), ("5Min", r"5min|5m")]
    for tf, body in pats:
        if re.search(rf"(?:^|[_\s,.-])(?:{body})(?:$|[_\s,.-])", s):
            return tf
    for tf, num in [("15Min", "15"), ("1Hour", "60"), ("5Min", "5")]:
        if re.search(rf"(?:^|[_\s,.-]){num}(?:$|[_\s,.-])", s):
            return tf
    return None


def _tf_from_timestamps(dts) -> str | None:
    """Recover an intraday timeframe from the minute-granularity of trade times."""
    if dts is None or len(dts) == 0:
        return None
    if (dts.dt.hour.eq(0) & dts.dt.minute.eq(0)).all():
        return "1Day"
    mins = {int(m) for m in dts.dt.minute.unique()}
    if mins <= {0}:
        return "1Hour"
    if mins <= {0, 15, 30, 45}:
        return "15Min"
    if all(m % 5 == 0 for m in mins):
        return "5Min"
    return None


def infer_timeframe(stem: str, dts=None) -> str:
    return _tf_from_filename(stem) or _tf_from_timestamps(dts) or "unknown_timeframe"


def strategy_from_parts(rel_parts: tuple):
    """(strategy_name, strategy_id, source_folder_label) from path components.
    Prefers the strategy FOLDER (more reliable than the filename)."""
    label = next((p.upper() for p in rel_parts[:-1] if p.upper() in LABELS), None)
    dirs = [p for p in rel_parts[:-1] if p.upper() not in LABELS]
    if dirs:
        name = dirs[-1]
    else:
        name = _strategy_from_filename(rel_parts[-1])
    return name, normalize_strategy_id(name), label


def _strategy_from_filename(fname: str) -> str:
    stem = Path(fname).stem
    stem = re.sub(r"^(?:win|loss|flat)[-_p]*", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[_-][A-Z0-9.]{1,6}_\d{4}-\d{2}-\d{2}$", "", stem)  # drop _SYM_date
    return stem or "unknown_strategy"


# -- csv inspection --------------------------------------------------------
def file_hash(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_datetimes(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return pd.Series([], dtype="datetime64[ns]"), 0
    headers = {h: (h or "").strip().lower() for h in rows[0]}
    dtcol = next((o for o, lo in headers.items() if "date" in lo or "time" in lo), None)
    if dtcol is None:
        return pd.Series([], dtype="datetime64[ns]"), len(rows)
    dts = pd.to_datetime(pd.Series([r.get(dtcol) for r in rows]),
                         errors="coerce", format="mixed").dropna()
    return dts, len(rows)


def summarize_csv(path) -> dict:
    """row count, trade count, net pnl, date range, and a timestamp-derived tf."""
    dts, row_count = _read_datetimes(path)
    if len(dts):
        earliest, latest = dts.min(), dts.max()
        date_range = f"{earliest:%Y%m%d}_{latest:%Y%m%d}"
        earliest_iso, latest_iso = earliest.isoformat(), latest.isoformat()
    else:
        date_range, earliest_iso, latest_iso = "unknown_range", None, None
    try:
        trades = parse_tv_csv(path)
    except Exception:
        trades = []
    net = round(sum(t.pnl for t in trades), 2)
    net_pct = round(sum(t.return_pct for t in trades) * 100.0, 4)
    return {
        "row_count": row_count, "trade_count": len(trades),
        "total_net_pnl": net, "total_net_pnl_pct": net_pct,
        "date_range": date_range, "earliest_trade_time": earliest_iso,
        "latest_trade_time": latest_iso, "ts_timeframe": _tf_from_timestamps(dts),
    }


# -- planning / applying ---------------------------------------------------
def dest_path(dest_root: Path, sid: str, symbol, timeframe: str,
              date_range: str) -> Path:
    sym = symbol or "UNKNOWN"
    sym_dir = symbol or "_unknown_symbol"
    fname = f"{sid}__{sym}__{timeframe}__{date_range}.csv"
    return Path(dest_root) / sid / sym_dir / timeframe / fname


def build_plan(source_root, dest_root) -> list:
    source_root, dest_root = Path(source_root), Path(dest_root)
    plan = []
    for src in sorted(source_root.rglob("*.csv")):
        rel = src.relative_to(source_root)
        name, sid, label = strategy_from_parts(rel.parts)
        symbol = infer_symbol(src.stem)
        summ = summarize_csv(src)
        tf = infer_timeframe(src.stem, None) if summ["ts_timeframe"] is None \
            else (_tf_from_filename(src.stem) or summ["ts_timeframe"])
        dest = dest_path(dest_root, sid, symbol, tf, summ["date_range"])
        plan.append({
            "source_path": str(src), "dest_path": str(dest),
            "strategy_id": sid, "strategy_name": name, "symbol": symbol,
            "timeframe": tf, "date_range": summ["date_range"],
            "earliest_trade_time": summ["earliest_trade_time"],
            "latest_trade_time": summ["latest_trade_time"],
            "row_count": summ["row_count"], "trade_count": summ["trade_count"],
            "total_net_pnl": summ["total_net_pnl"],
            "total_net_pnl_pct": summ["total_net_pnl_pct"],
            "source_folder_label": label, "copied_or_moved": "planned",
            "file_hash": file_hash(src),
        })
    return plan


def apply_plan(plan: list, move: bool = False) -> dict:
    """Copy (default) or move each planned file into place. Collision-safe:
    identical content -> skip; different content -> append an 8-char hash."""
    stats = {"organized": 0, "skipped": 0, "collisions": 0}
    for rec in plan:
        src = Path(rec["source_path"])
        dest = Path(rec["dest_path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if file_hash(dest) == rec["file_hash"]:
                rec["copied_or_moved"] = "skipped"
                stats["skipped"] += 1
                continue
            dest = dest.with_name(f"{dest.stem}__{rec['file_hash'][:8]}.csv")
            rec["dest_path"] = str(dest)
            stats["collisions"] += 1
        if move:
            shutil.move(str(src), str(dest))
            rec["copied_or_moved"] = "moved"
        else:
            shutil.copy2(str(src), str(dest))
            rec["copied_or_moved"] = "copied"
        stats["organized"] += 1
    return stats


def write_manifest(plan: list, dest_root) -> Path:
    import json
    path = Path(dest_root) / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"count": len(plan), "records": plan}, indent=2))
    return path


def _demo():
    assert normalize_strategy_id("Linear-Regression") == "linear_regression"
    assert normalize_strategy_id("Pin_Bar_Magic_v1") == "pin_bar_magic_v1"
    assert infer_symbol("WIN-Pin_Bar_Magic_v1_NASDAQ_GOOG_2026-06-19") == "GOOG"
    assert infer_symbol("NASDAQ_AAPL") == "AAPL"
    assert infer_symbol("NYSE_TSM") == "TSM"
    assert infer_timeframe("foo_15Min") == "15Min"
    assert infer_timeframe("foo") == "unknown_timeframe"
    n, sid, label = strategy_from_parts(("WINS", "Pin_Bar_Magic_v1", "x.csv"))
    assert sid == "pin_bar_magic_v1" and label == "WINS"
    print("export_organizer._demo OK")


if __name__ == "__main__":
    _demo()
