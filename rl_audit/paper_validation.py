#!/usr/bin/env python3
"""
rl_audit/paper_validation.py — paper-trading validation framework for the
alpha_entry RL agent.

Logs every RL recommendation with the context needed to later confirm (or
refute) a genuine edge on REAL paper fills, and computes rolling statistics at
trade-count checkpoints (100/250/500/1000).

SAFETY: read/write to a local sqlite DB only. No Alpaca, no orders. The agent is
NOT wired live; this is the instrument that decides whether it ever should be.

Schema (table `recommendations`):
    id            INTEGER PK
    timestamp     TEXT (UTC ISO)        — when the recommendation was made
    ticker        TEXT
    features      TEXT (JSON)           — the observation feature dict (audit trail)
    regime        TEXT                  — market regime tag at decision time
    decision      TEXT                  — enter_now | wait | skip | reduce_size
    expected_R    REAL                  — model/engine ex-ante expectation
    actual_R      REAL                  — realized R once the trade closes (NULL until then)
    outcome       TEXT                  — win | loss | scratch | open
    label_source  TEXT                  — paper_fill | backtest_oos | proxy (provenance)
    model_version TEXT
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path(__file__).resolve().parent.parent / "paper_validation.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recommendations (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp     TEXT NOT NULL,
    ticker        TEXT NOT NULL,
    features      TEXT,
    regime        TEXT,
    decision      TEXT NOT NULL,
    expected_R    REAL,
    actual_R      REAL,
    outcome       TEXT,
    label_source  TEXT NOT NULL DEFAULT 'paper_fill',
    model_version TEXT
);
CREATE INDEX IF NOT EXISTS idx_rec_ts ON recommendations(timestamp);
CREATE INDEX IF NOT EXISTS idx_rec_src ON recommendations(label_source);
CREATE INDEX IF NOT EXISTS idx_rec_decision ON recommendations(decision);
"""


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    return conn


def log_recommendation(conn: sqlite3.Connection, *, timestamp: str, ticker: str,
                       features: dict[str, Any], regime: str, decision: str,
                       expected_R: Optional[float], actual_R: Optional[float] = None,
                       outcome: str = "open", label_source: str = "paper_fill",
                       model_version: str = "alpha_entry_53f_v1") -> int:
    cur = conn.execute(
        "INSERT INTO recommendations (timestamp,ticker,features,regime,decision,expected_R,"
        "actual_R,outcome,label_source,model_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (timestamp, ticker, json.dumps(features, default=str), regime, decision,
         expected_R, actual_R, outcome, label_source, model_version))
    conn.commit()
    return cur.lastrowid


def _metrics(rs: list[float]) -> dict[str, float]:
    import math
    rs = [r for r in rs if r is not None]
    n = len(rs)
    if n == 0:
        return {"n": 0}
    mean = sum(rs) / n
    wins = [r for r in rs if r > 0]; losses = [r for r in rs if r < 0]
    sd = (sum((r - mean) ** 2 for r in rs) / max(1, n - 1)) ** 0.5
    dd_peak = 0.0; cum = 0.0; maxdd = 0.0
    for r in rs:
        cum += r; dd_peak = max(dd_peak, cum); maxdd = max(maxdd, dd_peak - cum)
    downside = (sum(min(0.0, r) ** 2 for r in rs) / n) ** 0.5
    gross_w = sum(wins); gross_l = abs(sum(losses))
    return {
        "n": n,
        "expectancy": round(mean, 4),
        "win_rate": round(len(wins) / n, 4),
        "profit_factor": round(gross_w / gross_l, 4) if gross_l > 0 else (9999.0 if gross_w > 0 else 0.0),
        "sharpe": round((mean / sd) * (252 ** 0.5), 4) if sd > 0 else 0.0,
        "sortino": round((mean / downside) * (252 ** 0.5), 4) if downside > 0 else 0.0,
        "calmar": round(mean / maxdd, 4) if maxdd > 0 else 0.0,
        "max_drawdown": round(maxdd, 4),
    }


def checkpoint_stats(conn: sqlite3.Connection, checkpoints=(100, 250, 500, 1000),
                     label_source: Optional[str] = None) -> dict[str, Any]:
    """Rolling stats over the first N CLOSED entered trades (actual_R not null)."""
    q = ("SELECT actual_R FROM recommendations WHERE decision='enter_now' "
         "AND actual_R IS NOT NULL")
    params: tuple = ()
    if label_source:
        q += " AND label_source=?"; params = (label_source,)
    q += " ORDER BY id ASC"
    rs = [row[0] for row in conn.execute(q, params).fetchall()]
    out = {"closed_entered_trades_available": len(rs), "checkpoints": {}}
    for c in checkpoints:
        out["checkpoints"][str(c)] = _metrics(rs[:c]) if len(rs) >= 1 else {"n": 0}
        out["checkpoints"][str(c)]["reached"] = len(rs) >= c
    return out


if __name__ == "__main__":
    conn = connect()
    print(f"paper_validation.db ready at {DB_PATH}")
    print(json.dumps(checkpoint_stats(conn), indent=2))
