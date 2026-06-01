"""
trading_day_monitor_agent.py — QuantumQuant
Intraday monitor: tracks live setups, active RL positions, risk exposure,
daily metrics, and fires alerts for key events.

Integrates:
  • PivotReversalScanner   — live label generation
  • ScalpingExecutionLayer — pattern tracking
  • InstitutionalScreener  — setup scoring
  • Risk controls from run.py environment vars
"""

from __future__ import annotations

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MONITOR_DAILY_LOSS_LIMIT  = float(os.environ.get("DAILY_LOSS_LIMIT",  -500.0))
MONITOR_MAX_DRAWDOWN      = float(os.environ.get("MAX_DRAWDOWN",      -0.05))   # -5 % equity
MONITOR_ALERT_SCORE_MIN   = float(os.environ.get("ALERT_SCORE_MIN",    0.65))
MONITOR_LOG_PATH          = os.environ.get("MONITOR_LOG_PATH",        "logs/monitor.log")
MONITOR_SESSION_CSV       = os.environ.get("MONITOR_SESSION_CSV",     "logs/session_summary.csv")


# ---------------------------------------------------------------------------
# Alert types
# ---------------------------------------------------------------------------

class AlertType:
    SETUP_FOUND     = "SETUP_FOUND"
    ENTRY_SIGNAL    = "ENTRY_SIGNAL"
    STOP_HIT        = "STOP_HIT"
    TARGET_HIT      = "TARGET_HIT"
    KILL_SWITCH     = "KILL_SWITCH"
    DAILY_LOSS_LIMIT= "DAILY_LOSS_LIMIT"
    DRAWDOWN_ALERT  = "DRAWDOWN_ALERT"


@dataclass
class Alert:
    ts:      str
    symbol:  str
    type:    str
    message: str
    data:    Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Monitor agent
# ---------------------------------------------------------------------------

class TradingDayMonitorAgent:
    """
    Instantiate once per trading session.
    Call update(symbol, bar) for every new bar on every symbol.
    Alerts fire via the alert_callback (default: log).
    """

    def __init__(
        self,
        symbols:         List[str],
        starting_equity: float            = 100_000.0,
        alert_callback:  Optional[Callable[[Alert], None]] = None,
        session_csv:     str              = MONITOR_SESSION_CSV,
    ):
        self.symbols         = symbols
        self.equity          = starting_equity
        self.starting_equity = starting_equity
        self.alert_cb        = alert_callback or self._default_alert
        self.session_csv     = session_csv

        # Per-symbol trackers
        from pivot_reversal_scanner      import PivotReversalScanner
        from scalping_execution_layer    import ScalpingExecutionLayer
        from tradingview_institutional_screener import InstitutionalScreener

        self._scanners:   Dict[str, PivotReversalScanner]   = {
            s: PivotReversalScanner(symbol=s) for s in symbols
        }
        self._scalpers:   Dict[str, ScalpingExecutionLayer] = {
            s: ScalpingExecutionLayer(symbol=s, order_callback=None) for s in symbols
        }
        self._screener   = InstitutionalScreener(min_score=MONITOR_ALERT_SCORE_MIN)
        self._bar_cache: Dict[str, List[Dict]] = {s: [] for s in symbols}

        # Risk state
        self._daily_pnl  = 0.0
        self._open_pos:   Dict[str, bool] = {s: False for s in symbols}
        self._kill_switch_fired = False

        # Session metrics
        self._alerts:    List[Alert]  = []
        self._trades_today:  int      = 0
        self._wins_today:    int      = 0

        # Setup CSV header
        os.makedirs(os.path.dirname(session_csv) if os.path.dirname(session_csv) else ".", exist_ok=True)
        if not os.path.exists(session_csv):
            with open(session_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "date", "symbol", "n_trades", "wins", "pnl",
                    "daily_loss_limit", "kill_switch",
                ])

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------

    def update(self, symbol: str, bar: Dict) -> List[Alert]:
        """
        Call once per bar per symbol.
        Returns any alerts fired during this bar.
        """
        if self._kill_switch_fired:
            return []

        fired: List[Alert] = []

        # ---- Pivot scan -------------------------------------------------
        lbl  = self._scanners[symbol].push(bar)
        self._bar_cache[symbol].append(bar)
        if len(self._bar_cache[symbol]) > 200:
            self._bar_cache[symbol].pop(0)

        # ---- Setup detection / scoring ----------------------------------
        bars_map = {symbol: self._bar_cache[symbol]}
        results  = self._screener.screen(bars_map)
        for res in results:
            if res.score >= MONITOR_ALERT_SCORE_MIN:
                a = Alert(
                    ts     = str(bar.get("timestamp", "")),
                    symbol = symbol,
                    type   = AlertType.SETUP_FOUND,
                    message= (f"{symbol} {res.setup_type} score={res.score:.2f}  "
                              f"close={res.close:.4f}  sup={res.support:.4f}  "
                              f"tags={res.tags}"),
                    data   = dict(score=res.score, setup=res.setup_type, tags=res.tags),
                )
                fired.append(a)
                self._fire(a)

        # ---- Scalping execution -----------------------------------------
        trade_result = self._scalpers[symbol].update(bar)
        if trade_result:
            self._trades_today += 1
            won = "win" in trade_result.outcome
            if won: self._wins_today += 1
            pnl_est = (trade_result.mfe_pct if won else -trade_result.mae_pct) * trade_result.entry_price
            self._daily_pnl += pnl_est
            self.equity     += pnl_est

            a_type  = AlertType.TARGET_HIT if won else AlertType.STOP_HIT
            a = Alert(
                ts     = str(bar.get("timestamp", "")),
                symbol = symbol,
                type   = a_type,
                message= (f"{symbol} trade closed: {trade_result.outcome}  "
                          f"entry={trade_result.entry_price:.4f}  "
                          f"MFE={trade_result.mfe_pct*100:.1f}%  "
                          f"MAE={trade_result.mae_pct*100:.1f}%"),
                data   = dict(outcome=trade_result.outcome,
                              mfe=trade_result.mfe_pct, mae=trade_result.mae_pct),
            )
            fired.append(a)
            self._fire(a)

        # ---- Risk checks ------------------------------------------------
        fired += self._risk_check(bar)

        self._alerts.extend(fired)
        return fired

    # ------------------------------------------------------------------
    # Risk
    # ------------------------------------------------------------------

    def _risk_check(self, bar: Dict) -> List[Alert]:
        alerts: List[Alert] = []

        # Daily loss limit
        if self._daily_pnl <= MONITOR_DAILY_LOSS_LIMIT and not self._kill_switch_fired:
            self._kill_switch_fired = True
            self._flatten_all(bar)
            a = Alert(
                ts      = str(bar.get("timestamp", "")),
                symbol  = "ALL",
                type    = AlertType.DAILY_LOSS_LIMIT,
                message = f"Daily loss limit hit: {self._daily_pnl:.2f}",
                data    = dict(daily_pnl=self._daily_pnl, limit=MONITOR_DAILY_LOSS_LIMIT),
            )
            alerts.append(a); self._fire(a)

        # Max drawdown
        dd = (self.equity - self.starting_equity) / (self.starting_equity + 1e-8)
        if dd <= MONITOR_MAX_DRAWDOWN and not self._kill_switch_fired:
            self._kill_switch_fired = True
            self._flatten_all(bar)
            a = Alert(
                ts      = str(bar.get("timestamp", "")),
                symbol  = "ALL",
                type    = AlertType.DRAWDOWN_ALERT,
                message = f"Max drawdown breached: {dd*100:.1f}%",
                data    = dict(drawdown=dd, threshold=MONITOR_MAX_DRAWDOWN),
            )
            alerts.append(a); self._fire(a)

        return alerts

    def _flatten_all(self, bar: Dict):
        logger.critical("KILL SWITCH — flattening all positions")
        for sym, scalper in self._scalpers.items():
            scalper.force_flatten(bar)

    # ------------------------------------------------------------------

    def session_summary(self) -> Dict:
        wr = self._wins_today / max(self._trades_today, 1)
        return dict(
            date         = datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            n_trades     = self._trades_today,
            wins         = self._wins_today,
            win_rate     = wr,
            daily_pnl    = self._daily_pnl,
            equity       = self.equity,
            drawdown_pct = (self.equity - self.starting_equity) / (self.starting_equity + 1e-8),
            kill_switch  = self._kill_switch_fired,
            n_alerts     = len(self._alerts),
        )

    def flush_session(self):
        s = self.session_summary()
        with open(self.session_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                s["date"], "ALL", s["n_trades"], s["wins"], s["daily_pnl"],
                MONITOR_DAILY_LOSS_LIMIT, s["kill_switch"],
            ])
        logger.info("Session flushed: %s", s)

    # ------------------------------------------------------------------

    @staticmethod
    def _default_alert(a: Alert):
        logger.info("[%s] %s — %s", a.type, a.symbol, a.message)

    def _fire(self, a: Alert):
        try:
            self.alert_cb(a)
        except Exception as e:
            logger.error("Alert callback error: %s", e)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import random
    random.seed(9)

    symbols = ["CRM", "RKLB"]
    monitor = TradingDayMonitorAgent(symbols, starting_equity=50_000, session_csv="/tmp/session_test.csv")

    for i in range(100):
        for sym in symbols:
            price = (190.0 if sym == "CRM" else 140.0) + random.gauss(0, 0.3)
            if i == 40: price *= 0.985
            if i == 42: price *= 1.015
            bar = dict(
                timestamp   = f"bar_{i:04d}",
                open        = price,
                high        = price + random.uniform(0, 0.3),
                low         = price - random.uniform(0, 0.4),
                close       = price + random.gauss(0, 0.1),
                volume      = random.uniform(3e4, 3e5) * (2.0 if i == 40 else 1.0),
                volume_ma20 = 1e5,
                vwap        = price * 0.999,
                ema9        = price * 0.998,
                ema20       = price * 0.996,
            )
            bar["close"] = max(bar["close"], bar["low"])
            alerts = monitor.update(sym, bar)

    monitor.flush_session()
    s = monitor.session_summary()
    logger.info("Session summary: %s", s)
    logger.info("trading_day_monitor_agent.py — OK")