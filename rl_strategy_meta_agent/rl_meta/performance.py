"""Metrics computation, rolling-window engine, and the performance database.

Conventions (documented so they are reproducible, not magic):
  * Returns are per-trade fractional returns on capital-at-risk.
  * Sharpe/Sortino are annualised:  per_trade_ratio * sqrt(trades_per_year),
    where trades_per_year = trade_count / years_spanned (>= 1 trade/yr floor).
  * Max drawdown is computed on the cumulative-PnL equity curve, returned as a
    positive fraction of peak equity (0.20 == 20%).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np

from .registry import DATA_DIR
from .types import Metrics, RollingMetrics, TradeRecord

DB_PATH = DATA_DIR / "strategy_performance.db"
INITIAL_EQUITY = 10_000.0          # explicit starting equity for the equity curve
START_EQUITY = INITIAL_EQUITY      # backwards-compatible alias


def _years_span(trades: list) -> float:
    if not trades:
        return 0.0
    d0 = min(date.fromisoformat(t.entry_date) for t in trades)
    d1 = max(date.fromisoformat(t.exit_date) for t in trades)
    return max((d1 - d0).days, 1) / 365.25


def compute_metrics(trades: list, initial_equity: float = INITIAL_EQUITY) -> Metrics:
    if not trades:
        return Metrics(initial_equity=initial_equity, ending_equity=initial_equity,
                       min_equity=initial_equity)

    pnls = np.array([t.pnl for t in trades], dtype=float)
    rets = np.array([t.return_pct for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    n = len(pnls)
    win_rate = len(wins) / n
    gross_win = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else (
        float("inf") if gross_win > 0 else 0.0)
    expectancy = float(pnls.mean())

    years = _years_span(trades)
    tpy = max(n / years, 1.0) if years > 0 else 1.0
    ann = np.sqrt(tpy)

    std = rets.std(ddof=1) if n > 1 else 0.0
    sharpe = float(rets.mean() / std * ann) if std > 0 else 0.0
    downside = rets[rets < 0]
    dstd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    sortino = float(rets.mean() / dstd * ann) if dstd > 0 else 0.0

    # Valid equity curve: chronological (by exit date), from an explicit initial
    # equity. Drawdown from the running PEAK, not raw cumulative PnL. If equity
    # ever hits <=0 the strategy blew up -> drawdown capped at 100%.
    order = sorted(range(n), key=lambda i: (trades[i].exit_date, trades[i].entry_date))
    ord_pnls = pnls[order]
    curve = np.concatenate([[initial_equity], initial_equity + np.cumsum(ord_pnls)])
    peak = np.maximum.accumulate(curve)
    min_equity = float(curve.min())
    ending_equity = float(curve[-1])
    blown_up = min_equity <= 0.0
    if blown_up:
        max_dd = 1.0
    else:
        safe_peak = np.where(peak > 0, peak, np.nan)
        max_dd = float(np.nanmax((peak - curve) / safe_peak))

    total_return = float((ending_equity - initial_equity) / initial_equity)
    if years >= 0.08 and ending_equity > 0:
        cagr = float((ending_equity / initial_equity) ** (1 / years) - 1)
    else:
        cagr = -1.0 if ending_equity <= 0 else 0.0   # wiped out / too-short span

    return Metrics(
        trade_count=n,
        win_rate=round(win_rate, 4),
        profit_factor=round(profit_factor, 4) if np.isfinite(profit_factor) else profit_factor,
        expectancy=round(expectancy, 2),
        sharpe=round(sharpe, 4),
        sortino=round(sortino, 4),
        max_drawdown=round(max_dd, 4),
        avg_win=round(float(wins.mean()), 2) if len(wins) else 0.0,
        avg_loss=round(float(losses.mean()), 2) if len(losses) else 0.0,
        avg_hold_days=round(float(np.mean([t.hold_days for t in trades])), 2),
        total_return=round(total_return, 4),
        cagr=round(cagr, 4),
        initial_equity=round(initial_equity, 2),
        ending_equity=round(ending_equity, 2),
        min_equity=round(min_equity, 2),
        blown_up=blown_up,
    )


def rolling_metrics(trades: list, as_of: str | None = None,
                    windows=(30, 60, 90)) -> RollingMetrics:
    """Metrics over trailing windows ending at `as_of` (default: last exit)."""
    if not trades:
        return RollingMetrics()
    as_of_d = date.fromisoformat(as_of) if as_of else max(
        date.fromisoformat(t.exit_date) for t in trades)

    def window(days: int) -> Metrics:
        cutoff = as_of_d.toordinal() - days
        sub = [t for t in trades
               if cutoff <= date.fromisoformat(t.exit_date).toordinal() <= as_of_d.toordinal()]
        return compute_metrics(sub)

    w = {d: window(d) for d in windows}
    return RollingMetrics(window_30=w.get(30, Metrics()),
                          window_60=w.get(60, Metrics()),
                          window_90=w.get(90, Metrics()))


class PerformanceDB:
    """Thin sqlite store. One row per (strategy_id, date) snapshot."""

    def __init__(self, path: Path = DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        return sqlite3.connect(self.path)

    def _init(self):
        with self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS performance (
                    strategy_id TEXT NOT NULL,
                    date        TEXT NOT NULL,
                    lifetime    TEXT NOT NULL,
                    rolling     TEXT NOT NULL,
                    rank        INTEGER,
                    tier        TEXT,
                    status      TEXT,
                    PRIMARY KEY (strategy_id, date)
                )""")
            # Richer per-(strategy, symbol, timeframe, split) backtest table.
            c.execute("""
                CREATE TABLE IF NOT EXISTS backtests (
                    strategy_id      TEXT NOT NULL,
                    symbol           TEXT NOT NULL,   -- 'AGGREGATE' for pooled
                    timeframe        TEXT NOT NULL,
                    split            TEXT NOT NULL,   -- full|train|validation|oos
                    date             TEXT NOT NULL,
                    metrics          TEXT NOT NULL,
                    rolling_30       TEXT,
                    rolling_60       TEXT,
                    rolling_90       TEXT,
                    tier             TEXT,
                    eligible         INTEGER,
                    rejection_reason TEXT,
                    source           TEXT,   -- evaluation source / per-symbol source
                    confidence       REAL,   -- 0..100 trustworthiness (aggregate row)
                    PRIMARY KEY (strategy_id, symbol, timeframe, split, date)
                )""")
            # Imported TradingView trades (the evaluation source of truth).
            c.execute("""
                CREATE TABLE IF NOT EXISTS imported_trades (
                    strategy_id  TEXT NOT NULL,
                    symbol       TEXT NOT NULL,
                    timeframe    TEXT NOT NULL,
                    version_hash TEXT NOT NULL,
                    source_file  TEXT NOT NULL,
                    trade_idx    INTEGER NOT NULL,
                    entry_date   TEXT, exit_date TEXT,
                    entry_price  REAL, exit_price REAL,
                    pnl REAL, return_pct REAL, hold_days REAL,
                    raw_qty REAL, normalized_qty REAL, scale_factor REAL,
                    normalized INTEGER, raw_pnl_usd REAL,
                    PRIMARY KEY (strategy_id, symbol, timeframe, version_hash,
                                 source_file, trade_idx)
                )""")
            # Dedup ledger keyed by CSV content hash -> protects re-imports.
            c.execute("""
                CREATE TABLE IF NOT EXISTS import_log (
                    content_hash TEXT PRIMARY KEY,
                    strategy_id TEXT, symbol TEXT, timeframe TEXT,
                    version_hash TEXT, source_file TEXT,
                    n_trades INTEGER, imported_at TEXT
                )""")

    def record(self, strategy_id: str, on_date: str, lifetime: Metrics,
               rolling: RollingMetrics, rank: int, tier: str, status: str):
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO performance VALUES (?,?,?,?,?,?,?)",
                (strategy_id, on_date,
                 json.dumps(lifetime.to_dict()), json.dumps(rolling.to_dict()),
                 int(rank), tier, status))

    def latest(self, strategy_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT strategy_id,date,lifetime,rolling,rank,tier,status "
                "FROM performance WHERE strategy_id=? ORDER BY date DESC LIMIT 1",
                (strategy_id,)).fetchone()
        return _row_to_dict(row) if row else None

    def all_latest(self) -> list:
        with self._conn() as c:
            rows = c.execute("""
                SELECT p.strategy_id,p.date,p.lifetime,p.rolling,p.rank,p.tier,p.status
                FROM performance p
                JOIN (SELECT strategy_id, MAX(date) d FROM performance GROUP BY strategy_id) m
                ON p.strategy_id=m.strategy_id AND p.date=m.d
                ORDER BY p.rank ASC""").fetchall()
        return [_row_to_dict(r) for r in rows]


    def record_cell(self, strategy_id: str, symbol: str, timeframe: str,
                    split: str, on_date: str, metrics: Metrics,
                    rolling: RollingMetrics | None = None, tier: str = "",
                    eligible: bool = False, rejection_reason: str = "",
                    source: str = "", confidence: float = 0.0):
        rolling = rolling or RollingMetrics()
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO backtests VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (strategy_id, symbol, timeframe, split, on_date,
                 json.dumps(metrics.to_dict()),
                 json.dumps(rolling.window_30.to_dict()),
                 json.dumps(rolling.window_60.to_dict()),
                 json.dumps(rolling.window_90.to_dict()),
                 tier, int(eligible), rejection_reason, source, float(confidence)))

    def cells(self, strategy_id: str | None = None,
              split: str | None = None) -> list:
        q = ("SELECT strategy_id,symbol,timeframe,split,date,metrics,"
             "tier,eligible,rejection_reason,source FROM backtests")
        clauses, args = [], []
        if strategy_id:
            clauses.append("strategy_id=?"); args.append(strategy_id)
        if split:
            clauses.append("split=?"); args.append(split)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        return [{
            "strategy_id": r[0], "symbol": r[1], "timeframe": r[2], "split": r[3],
            "date": r[4], "metrics": json.loads(r[5]), "tier": r[6],
            "eligible": bool(r[7]), "rejection_reason": r[8], "source": r[9],
        } for r in rows]


    def has_import(self, content_hash: str) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM import_log WHERE content_hash=?",
                             (content_hash,)).fetchone() is not None

    def record_import(self, strategy_id: str, symbol: str, timeframe: str,
                      version_hash: str, source_file: str, content_hash: str,
                      trades: list, on_date: str, force: bool = False) -> str:
        """Persist imported trades. Returns 'imported' | 'duplicate' | 'replaced'.

        Duplicate protection: the same CSV content (content_hash) is never
        imported twice unless force=True. Re-importing the same source_file
        replaces its prior rows so trade counts never double up."""
        if self.has_import(content_hash) and not force:
            return "duplicate"
        with self._conn() as c:
            existing = c.execute(
                "SELECT COUNT(*) FROM imported_trades WHERE strategy_id=? AND "
                "symbol=? AND timeframe=? AND version_hash=? AND source_file=?",
                (strategy_id, symbol, timeframe, version_hash, source_file)
            ).fetchone()[0]
            # Clear prior rows for this (strategy,symbol,tf,version,file).
            c.execute("DELETE FROM imported_trades WHERE strategy_id=? AND "
                      "symbol=? AND timeframe=? AND version_hash=? AND source_file=?",
                      (strategy_id, symbol, timeframe, version_hash, source_file))
            c.execute("DELETE FROM import_log WHERE strategy_id=? AND symbol=? AND "
                      "timeframe=? AND source_file=?",
                      (strategy_id, symbol, timeframe, source_file))
            c.executemany(
                "INSERT INTO imported_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(strategy_id, symbol, timeframe, version_hash, source_file, i,
                  t.entry_date, t.exit_date, t.entry_price, t.exit_price,
                  t.pnl, t.return_pct, t.hold_days,
                  t.raw_qty, t.normalized_qty, t.scale_factor,
                  int(t.normalized), t.raw_pnl_usd) for i, t in enumerate(trades)])
            c.execute("INSERT OR REPLACE INTO import_log VALUES (?,?,?,?,?,?,?,?)",
                      (content_hash, strategy_id, symbol, timeframe, version_hash,
                       source_file, len(trades), on_date))
        return "replaced" if existing else "imported"

    def imported_trades(self, strategy_id: str, timeframe: str,
                        version_hash: str) -> dict:
        """{symbol: [TradeRecord]} of imported trades matching this version.
        The reconstructed `pnl` is the stored (normalized) effective PnL."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT symbol,entry_date,exit_date,entry_price,exit_price,pnl,"
                "return_pct,hold_days,raw_qty,normalized_qty,scale_factor,"
                "normalized,raw_pnl_usd FROM imported_trades WHERE strategy_id=? AND "
                "timeframe=? AND version_hash=? ORDER BY symbol,exit_date",
                (strategy_id, timeframe, version_hash)).fetchall()
        out: dict = {}
        for r in rows:
            out.setdefault(r[0], []).append(TradeRecord(
                entry_date=r[1], exit_date=r[2], entry_price=r[3], exit_price=r[4],
                pnl=r[5], return_pct=r[6], hold_days=r[7],
                raw_qty=r[8] or 0.0, normalized_qty=r[9] or 0.0,
                scale_factor=r[10] if r[10] is not None else 1.0,
                normalized=bool(r[11]), raw_pnl_usd=r[12] or 0.0))
        return out

    def imported_strategies(self, timeframe: str | None = None) -> list:
        """Distinct (strategy_id, version_hash) pairs with imported trades.
        Lets the backtester evaluate TradingView strategies that have no
        pinescript .txt (they enter only via import)."""
        q = "SELECT DISTINCT strategy_id, version_hash FROM imported_trades"
        args = []
        if timeframe:
            q += " WHERE timeframe=?"; args.append(timeframe)
        with self._conn() as c:
            return [(r[0], r[1]) for r in c.execute(q, args).fetchall()]


def aggregate_metrics(trades_by_symbol: dict) -> Metrics:
    """Pool ALL trades across symbols, then compute once. PF/Sharpe do NOT
    average across symbols, so pooling is the only correct aggregation."""
    pooled = [t for ts in trades_by_symbol.values() for t in ts]
    return compute_metrics(pooled)


def _row_to_dict(row) -> dict:
    return {
        "strategy_id": row[0], "date": row[1],
        "lifetime": json.loads(row[2]), "rolling": json.loads(row[3]),
        "rank": row[4], "tier": row[5], "status": row[6],
    }


def _demo():
    trades = [
        TradeRecord("2023-01-02", "2023-01-05", 100, 110, 1000.0, 0.10, 3),
        TradeRecord("2023-02-02", "2023-02-04", 110, 105, -500.0, -0.045, 2),
        TradeRecord("2023-03-02", "2023-03-09", 105, 113, 800.0, 0.076, 7),
    ]
    m = compute_metrics(trades)
    assert m.trade_count == 3
    assert abs(m.win_rate - 2 / 3) < 1e-3
    assert abs(m.profit_factor - (1800 / 500)) < 1e-3
    assert m.max_drawdown >= 0.0
    r = rolling_metrics(trades, as_of="2023-03-09")
    assert r.window_30.trade_count == 1  # only the March trade in trailing 30d
    print("performance._demo OK", m.profit_factor)


if __name__ == "__main__":
    _demo()
