#!/usr/bin/env python3
"""
Durable execution ledger for the V2.9.8 live engine (SQLite).
==============================================================

The single source of truth for what the engine has decided, intended, submitted,
and had filled. It is the substrate for THREE hard requirements:

* Idempotency (write-ahead): an order row is written with status ``intent`` BEFORE
  it is submitted and updated to ``submitted`` after the broker acks, so a crash
  between the two is recoverable. ``client_order_id`` is UNIQUE, so a re-run of
  the same decision cannot create a duplicate logical order.

* Crash-recovery state machine: each rebalance is a row in ``rebalance_runs`` with
  an explicit state (PLANNED -> SELLING -> SELLS_RECONCILED -> BUYING ->
  RECONCILING -> COMPLETE/FAILED). On restart the engine inspects this + the
  broker before doing anything.

* Single active rebalance: a one-row ``active_rebalance`` lock (UNIQUE id=1)
  enforced by the DATABASE, not an in-memory flag, so two processes cannot
  execute the same session concurrently.

Order completion is decided by ``cumulative_filled_qty`` vs the desired quantity,
NEVER by "an order exists" or "reached a terminal state" -- a terminally
partially-filled leg still has a residual to trade (see ``is_terminal``).

DB file defaults to ``live_trading.db`` (gitignored via ``*.db``). Uses WAL +
busy_timeout for safe concurrent access, and explicit transactions.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DB_PATH = os.environ.get("LIVE_TRADING_DB", str(PROJECT_ROOT / "live_trading.db"))

# Rebalance run states (crash-recovery state machine).
RUN_STATES = ("PLANNED", "SELLING", "SELLS_RECONCILED", "BUYING",
              "RECONCILING", "COMPLETE", "FAILED")

# Alpaca order lifecycle. Terminal = no further fills will occur. A terminal
# order does NOT imply the requested quantity was achieved.
_TERMINAL = {"filled", "canceled", "cancelled", "expired", "rejected",
             "done_for_day", "stopped", "suspended", "failed"}


def is_terminal(status: str) -> bool:
    return str(status).lower() in _TERMINAL


_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_state (
    name               TEXT PRIMARY KEY,
    prev_target        TEXT NOT NULL,          -- JSON list of weights (risk_assets order)
    last_rebalance_date TEXT,                  -- ISO date or NULL
    last_decision_hash TEXT,
    updated_at         REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_hash   TEXT PRIMARY KEY,
    as_of_close     TEXT NOT NULL,
    target_session  TEXT,
    forecast_source TEXT NOT NULL,
    recovery_state  TEXT,
    severe_bear     INTEGER,
    leverage_cap    REAL,
    growth_ceiling  REAL,
    vol_budget      REAL,
    ex_ante_vol     REAL,
    gross           REAL,
    growth          REAL,
    defensive       REAL,
    target_weights  TEXT NOT NULL,             -- JSON {symbol: weight}
    mode            TEXT,
    rebalance_reason TEXT,                      -- scheduled | loss_replacement | ...
    rotation_events TEXT,                       -- JSON list of exceptional-exit events
    created_at      REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rebalance_runs (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_hash TEXT NOT NULL,
    mode          TEXT NOT NULL,
    state         TEXT NOT NULL,
    note          TEXT,
    started_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS active_rebalance (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    run_id      INTEGER NOT NULL,
    pid         INTEGER,
    acquired_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    client_order_id      TEXT PRIMARY KEY,
    run_id               INTEGER,
    decision_hash        TEXT,
    symbol               TEXT NOT NULL,
    side                 TEXT NOT NULL,
    order_type           TEXT NOT NULL,
    intent_qty           REAL,
    intent_notional      REAL,
    submitted_qty        REAL,
    submitted_notional   REAL,
    broker_order_id      TEXT,
    status               TEXT NOT NULL,
    cumulative_filled_qty REAL NOT NULL DEFAULT 0,
    avg_fill_price       REAL,
    note                 TEXT,
    requested_at         REAL NOT NULL,
    submitted_at         REAL,
    updated_at           REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client_order_id TEXT NOT NULL,
    broker_order_id TEXT,
    filled_qty      REAL,
    filled_avg_price REAL,
    status          TEXT,
    ts              REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL NOT NULL,
    mode              TEXT,
    equity            REAL,
    cash              REAL,
    buying_power      REAL,
    portfolio_value   REAL,
    long_market_value REAL
);

CREATE TABLE IF NOT EXISTS high_water_mark (
    id         INTEGER PRIMARY KEY CHECK (id = 1),
    hwm_equity REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS kill_switch_events (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    ts     REAL NOT NULL,
    kind   TEXT NOT NULL,
    active INTEGER NOT NULL,
    reason TEXT
);

CREATE TABLE IF NOT EXISTS loss_replacement_cooldown (
    symbol         TEXT PRIMARY KEY,
    last_exit_date TEXT NOT NULL,          -- ISO date of the most recent overlay exit
    updated_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS position_entry (
    symbol           TEXT PRIMARY KEY,
    first_entry_date TEXT NOT NULL,        -- ISO date the position was first established
    updated_at       REAL NOT NULL
);
"""


class RebalanceLockError(RuntimeError):
    """Raised when another process already holds the single-active-rebalance lock."""


class ExecutionStore:
    def __init__(self, db_path: str | None = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Additive, idempotent column migrations for pre-existing DBs."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(decisions)")}
        for col in ("rebalance_reason", "rotation_events"):
            if col not in cols:
                self._conn.execute(f"ALTER TABLE decisions ADD COLUMN {col} TEXT")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def tx(self, immediate: bool = True):
        """Explicit transaction (BEGIN IMMEDIATE acquires the write lock at once)."""
        self._conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield self._conn
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    # ---- strategy cross-day state ---------------------------------------
    def get_strategy_state(self, name: str = "v298") -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM strategy_state WHERE name=?", (name,)).fetchone()
        if row is None:
            return None
        return {"prev_target": json.loads(row["prev_target"]),
                "last_rebalance_date": row["last_rebalance_date"],
                "last_decision_hash": row["last_decision_hash"]}

    def set_strategy_state(self, prev_target: list[float],
                           last_rebalance_date: str | None,
                           last_decision_hash: str | None,
                           name: str = "v298") -> None:
        with self.tx():
            self._conn.execute(
                "INSERT INTO strategy_state(name, prev_target, last_rebalance_date,"
                " last_decision_hash, updated_at) VALUES(?,?,?,?,?)"
                " ON CONFLICT(name) DO UPDATE SET prev_target=excluded.prev_target,"
                " last_rebalance_date=excluded.last_rebalance_date,"
                " last_decision_hash=excluded.last_decision_hash,"
                " updated_at=excluded.updated_at",
                (name, json.dumps(list(prev_target)), last_rebalance_date,
                 last_decision_hash, time.time()))

    # ---- loss-replacement cooldown (overlay churn protection) -----------
    def get_cooldowns(self) -> dict[str, str]:
        """symbol -> ISO date of its most recent loss-replacement exit."""
        rows = self._conn.execute(
            "SELECT symbol, last_exit_date FROM loss_replacement_cooldown").fetchall()
        return {r["symbol"]: r["last_exit_date"] for r in rows}

    def set_cooldown(self, symbol: str, last_exit_date: str) -> None:
        """Record/refresh the cooldown clock for an exited symbol (idempotent)."""
        with self.tx():
            self._conn.execute(
                "INSERT INTO loss_replacement_cooldown(symbol, last_exit_date, updated_at)"
                " VALUES(?,?,?) ON CONFLICT(symbol) DO UPDATE SET"
                " last_exit_date=excluded.last_exit_date, updated_at=excluded.updated_at",
                (symbol.upper(), last_exit_date, time.time()))

    # ---- position first-entry ledger (days_held for adaptive rotation) --
    def get_entries(self) -> dict[str, str]:
        """symbol -> ISO date the position was first established (for days_held)."""
        rows = self._conn.execute(
            "SELECT symbol, first_entry_date FROM position_entry").fetchall()
        return {r["symbol"]: r["first_entry_date"] for r in rows}

    def set_entry(self, symbol: str, first_entry_date: str) -> None:
        """Stamp first-entry date on a flat->held transition (no-op if already set)."""
        with self.tx():
            self._conn.execute(
                "INSERT INTO position_entry(symbol, first_entry_date, updated_at)"
                " VALUES(?,?,?) ON CONFLICT(symbol) DO NOTHING",
                (symbol.upper(), first_entry_date, time.time()))

    def clear_entry(self, symbol: str) -> None:
        """Clear first-entry on a full exit."""
        with self.tx():
            self._conn.execute("DELETE FROM position_entry WHERE symbol=?",
                               (symbol.upper(),))

    # ---- decisions ------------------------------------------------------
    def record_decision(self, d: dict) -> None:
        """Idempotent on decision_hash. `d` carries the PortfolioDecision fields."""
        with self.tx():
            self._conn.execute(
                "INSERT OR REPLACE INTO decisions(decision_hash, as_of_close,"
                " target_session, forecast_source, recovery_state, severe_bear,"
                " leverage_cap, growth_ceiling, vol_budget, ex_ante_vol, gross,"
                " growth, defensive, target_weights, mode, rebalance_reason,"
                " rotation_events, created_at)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (d["decision_hash"], d["as_of_close"], d.get("target_session"),
                 d["forecast_source"], d.get("recovery_state"),
                 int(bool(d.get("severe_bear"))), d.get("leverage_cap"),
                 d.get("growth_ceiling"), d.get("vol_budget"), d.get("ex_ante_vol"),
                 d.get("gross"), d.get("growth"), d.get("defensive"),
                 json.dumps(d["target_weights"]), d.get("mode"),
                 d.get("rebalance_reason"),
                 json.dumps(d.get("rotation_events") or []), time.time()))

    def get_decision(self, decision_hash: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM decisions WHERE decision_hash=?", (decision_hash,)).fetchone()
        return dict(row) if row else None

    # ---- rebalance run state machine + single-active lock ---------------
    def begin_or_resume_run(self, decision_hash: str, mode: str,
                            pid: int | None = None) -> tuple[int, str, bool]:
        """
        Acquire the single-active-rebalance lock and return (run_id, state, resumed).

        If a COMPLETE run for this decision_hash+mode exists -> (run_id, 'COMPLETE',
        True) without re-locking (nothing to do). If an incomplete run exists ->
        resume it. Otherwise create a PLANNED run. Raises RebalanceLockError if a
        DIFFERENT run currently holds the lock.
        """
        pid = pid if pid is not None else os.getpid()
        with self.tx():
            done = self._conn.execute(
                "SELECT run_id FROM rebalance_runs WHERE decision_hash=? AND mode=?"
                " AND state='COMPLETE' ORDER BY run_id DESC LIMIT 1",
                (decision_hash, mode)).fetchone()
            if done is not None:
                return int(done["run_id"]), "COMPLETE", True

            existing = self._conn.execute(
                "SELECT run_id, state FROM rebalance_runs WHERE decision_hash=?"
                " AND mode=? AND state NOT IN ('COMPLETE','FAILED')"
                " ORDER BY run_id DESC LIMIT 1", (decision_hash, mode)).fetchone()

            active = self._conn.execute(
                "SELECT run_id, pid FROM active_rebalance WHERE id=1").fetchone()

            if existing is not None:
                run_id, state, resumed = int(existing["run_id"]), existing["state"], True
            else:
                now = time.time()
                cur = self._conn.execute(
                    "INSERT INTO rebalance_runs(decision_hash, mode, state, started_at,"
                    " updated_at) VALUES(?,?,?,?,?)",
                    (decision_hash, mode, "PLANNED", now, now))
                run_id, state, resumed = int(cur.lastrowid), "PLANNED", False

            if active is not None and int(active["run_id"]) != run_id:
                raise RebalanceLockError(
                    f"another rebalance run {active['run_id']} (pid={active['pid']}) "
                    f"holds the active lock")

            self._conn.execute("DELETE FROM active_rebalance WHERE id=1")
            self._conn.execute(
                "INSERT INTO active_rebalance(id, run_id, pid, acquired_at)"
                " VALUES(1,?,?,?)", (run_id, pid, time.time()))
            return run_id, state, resumed

    def set_run_state(self, run_id: int, state: str, note: str | None = None) -> None:
        if state not in RUN_STATES:
            raise ValueError(f"unknown run state {state!r}")
        with self.tx():
            self._conn.execute(
                "UPDATE rebalance_runs SET state=?, note=COALESCE(?, note),"
                " updated_at=? WHERE run_id=?", (state, note, time.time(), run_id))

    def finish_run(self, run_id: int, state: str, note: str | None = None) -> None:
        """Set a terminal state (COMPLETE/FAILED) and release the active lock."""
        if state not in ("COMPLETE", "FAILED"):
            raise ValueError("finish_run requires COMPLETE or FAILED")
        with self.tx():
            self._conn.execute(
                "UPDATE rebalance_runs SET state=?, note=COALESCE(?, note),"
                " updated_at=? WHERE run_id=?", (state, note, time.time(), run_id))
            self._conn.execute(
                "DELETE FROM active_rebalance WHERE id=1 AND run_id=?", (run_id,))

    def force_release_lock(self) -> None:
        with self.tx():
            self._conn.execute("DELETE FROM active_rebalance WHERE id=1")

    def active_run(self) -> dict | None:
        row = self._conn.execute(
            "SELECT ar.run_id, ar.pid, ar.acquired_at, r.decision_hash, r.mode, r.state"
            " FROM active_rebalance ar JOIN rebalance_runs r ON r.run_id=ar.run_id"
            " WHERE ar.id=1").fetchone()
        return dict(row) if row else None

    def get_run(self, run_id: int) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM rebalance_runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    # ---- orders (write-ahead + fill reconciliation) ---------------------
    def record_order_intent(self, client_order_id: str, run_id: int | None,
                            decision_hash: str | None, symbol: str, side: str,
                            order_type: str, intent_qty: float | None,
                            intent_notional: float | None) -> bool:
        """
        Write-ahead the order intent. Returns True if a NEW intent row was created,
        False if this client_order_id already exists (idempotent no-op).
        """
        now = time.time()
        with self.tx():
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO orders(client_order_id, run_id, decision_hash,"
                " symbol, side, order_type, intent_qty, intent_notional, status,"
                " requested_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (client_order_id, run_id, decision_hash, symbol.upper(), side.lower(),
                 order_type, intent_qty, intent_notional, "intent", now, now))
            return cur.rowcount > 0

    def mark_order_submitted(self, client_order_id: str, broker_order_id: str,
                             submitted_qty: float | None,
                             submitted_notional: float | None,
                             status: str = "submitted") -> None:
        with self.tx():
            self._conn.execute(
                "UPDATE orders SET broker_order_id=?, submitted_qty=?,"
                " submitted_notional=?, status=?, submitted_at=?, updated_at=?"
                " WHERE client_order_id=?",
                (broker_order_id, submitted_qty, submitted_notional, status,
                 time.time(), time.time(), client_order_id))

    def update_order_fill(self, client_order_id: str, status: str,
                          cumulative_filled_qty: float,
                          avg_fill_price: float | None,
                          broker_order_id: str | None = None) -> None:
        with self.tx():
            self._conn.execute(
                "UPDATE orders SET status=?, cumulative_filled_qty=?, avg_fill_price=?,"
                " broker_order_id=COALESCE(?, broker_order_id), updated_at=?"
                " WHERE client_order_id=?",
                (status, cumulative_filled_qty, avg_fill_price, broker_order_id,
                 time.time(), client_order_id))
            self._conn.execute(
                "INSERT INTO fills(client_order_id, broker_order_id, filled_qty,"
                " filled_avg_price, status, ts) VALUES(?,?,?,?,?,?)",
                (client_order_id, broker_order_id, cumulative_filled_qty,
                 avg_fill_price, status, time.time()))

    def mark_order_failed(self, client_order_id: str, reason: str) -> None:
        with self.tx():
            self._conn.execute(
                "UPDATE orders SET status='failed', note=?, updated_at=?"
                " WHERE client_order_id=?", (reason, time.time(), client_order_id))

    def get_order(self, client_order_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE client_order_id=?", (client_order_id,)).fetchone()
        return dict(row) if row else None

    def orders_for_run(self, run_id: int) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE run_id=? ORDER BY requested_at", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def open_orders_for_run(self, run_id: int) -> list[dict]:
        """Orders whose status is NOT terminal (still working)."""
        return [o for o in self.orders_for_run(run_id) if not is_terminal(o["status"])]

    # ---- equity / high-water-mark / kill switch -------------------------
    def record_equity(self, mode: str, equity: float, cash: float,
                      buying_power: float, portfolio_value: float,
                      long_market_value: float) -> None:
        with self.tx():
            self._conn.execute(
                "INSERT INTO equity_snapshots(ts, mode, equity, cash, buying_power,"
                " portfolio_value, long_market_value) VALUES(?,?,?,?,?,?,?)",
                (time.time(), mode, equity, cash, buying_power, portfolio_value,
                 long_market_value))

    def get_high_water_mark(self) -> float | None:
        row = self._conn.execute(
            "SELECT hwm_equity FROM high_water_mark WHERE id=1").fetchone()
        return float(row["hwm_equity"]) if row else None

    def update_high_water_mark(self, equity: float) -> float:
        """Raise the HWM to `equity` if higher (or seed it). Returns the HWM."""
        with self.tx():
            row = self._conn.execute(
                "SELECT hwm_equity FROM high_water_mark WHERE id=1").fetchone()
            hwm = max(float(row["hwm_equity"]), equity) if row else equity
            self._conn.execute(
                "INSERT INTO high_water_mark(id, hwm_equity, updated_at) VALUES(1,?,?)"
                " ON CONFLICT(id) DO UPDATE SET hwm_equity=excluded.hwm_equity,"
                " updated_at=excluded.updated_at", (hwm, time.time()))
            return hwm

    def record_kill_switch(self, kind: str, active: bool, reason: str) -> None:
        with self.tx():
            self._conn.execute(
                "INSERT INTO kill_switch_events(ts, kind, active, reason)"
                " VALUES(?,?,?,?)", (time.time(), kind, int(bool(active)), reason))

    def latest_kill_switch(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM kill_switch_events ORDER BY id DESC LIMIT 1").fetchone()
        return dict(row) if row else None


def demo() -> None:
    """Self-check: idempotent intents, run state machine, single-active lock."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        s = ExecutionStore(os.path.join(td, "t.db"))
        # write-ahead idempotency
        assert s.record_order_intent("v298-20260101-SPY-B-abc", 1, "h", "SPY",
                                     "buy", "market", None, 1000.0) is True
        assert s.record_order_intent("v298-20260101-SPY-B-abc", 1, "h", "SPY",
                                     "buy", "market", None, 1000.0) is False
        # run + lock
        rid, st, resumed = s.begin_or_resume_run("h", "paper")
        assert st == "PLANNED" and not resumed
        # second process cannot start a different run while lock held
        try:
            s.begin_or_resume_run("h2", "paper")
            raise AssertionError("expected lock error")
        except RebalanceLockError:
            pass
        # same decision resumes the same run
        rid2, st2, resumed2 = s.begin_or_resume_run("h", "paper")
        assert rid2 == rid and resumed2
        s.finish_run(rid, "COMPLETE")
        assert s.active_run() is None
        # completed decision short-circuits
        rid3, st3, resumed3 = s.begin_or_resume_run("h", "paper")
        assert st3 == "COMPLETE" and resumed3
        # hwm
        assert s.update_high_water_mark(1000.0) == 1000.0
        assert s.update_high_water_mark(900.0) == 1000.0
        assert s.update_high_water_mark(1200.0) == 1200.0
        s.close()
    print("execution_store demo: OK")


if __name__ == "__main__":
    demo()
