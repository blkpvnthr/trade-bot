"""
market_state_engine.py — Centralized Market Truth Source
========================================================

Builds ONE shared `market_state` dict every cycle, consumed by Alpha, OOPS,
Probability, Buffett, Black-Litterman, and the Adaptive Capital Allocator so they
all reason from the same market picture instead of re-deriving it independently.

Inputs (all optional — missing data degrades confidence, never crashes):
    * existing market_context (the authoritative regime/permission source)
    * StreamingIndicatorEngine snapshot
    * PremarketLiveFeed snapshot
    * SPY/QQQ bars, sector-ETF bars
    * VIX data, breadth data

Outputs:
    daily_system_outputs/YYYYMMDD/market_state.json
    daily_system_outputs/YYYYMMDD/market_state_diagnostic.json

Rules:
    * MarketStateEngine is ADVISORY. It cannot place orders.
    * Missing data lowers `confidence`; it does not raise.
    * If state can't be built, it falls back to the existing market_context.
    * risk_on and risk_off are mutually exclusive (never both true).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Shared atomic JSON writer (single implementation in opportunity_utils); guarded so
# this module stays import-safe standalone. Falls back to an atomic tmp+rename write.
try:
    from opportunity_utils import write_json_safe as _write_json_safe
except Exception:
    def _write_json_safe(path, data, log_fn=None) -> bool:
        import json as _json, os as _os
        from pathlib import Path as _P
        p = _P(path); tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(_json.dumps(data, indent=2, default=str), encoding="utf-8")
        _os.replace(tmp, p)
        return True


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_TAG = os.getenv("RUN_DATA_DATE", datetime.now(timezone.utc).strftime("%Y%m%d"))
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"

STATE_KEYS: List[str] = [
    "regime", "trend_strength", "risk_on", "risk_off", "volatility_regime",
    "sector_rotation", "breadth", "spy_trend", "qqq_trend", "vix_state",
    "premarket_risk_state", "confidence",
]


def _f(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return None if v != v else v
    except Exception:
        return None


def _trend_from_indicators(ind: Optional[dict]) -> Optional[str]:
    """up / down / sideways from sma_20 vs sma_50 (None if not warm/available)."""
    if not ind:
        return None
    s20, s50 = _f(ind.get("sma_20")), _f(ind.get("sma_50"))
    if s20 is None or s50 is None or s50 == 0:
        return None
    diff = (s20 - s50) / abs(s50)
    if diff > 0.003:
        return "up"
    if diff < -0.003:
        return "down"
    return "sideways"


def _trend_from_bars(df: Any) -> Optional[str]:
    """up / down / sideways from a bars DataFrame (last close vs 20-bar mean)."""
    try:
        if df is None or not hasattr(df, "__len__") or len(df) < 5:
            return None
        closes = list(df["close"]) if hasattr(df, "columns") else list(df)
        last = float(closes[-1])
        n = min(20, len(closes))
        avg = sum(float(c) for c in closes[-n:]) / n
        if avg == 0:
            return None
        diff = (last - avg) / abs(avg)
        if diff > 0.003:
            return "up"
        if diff < -0.003:
            return "down"
        return "sideways"
    except Exception:
        return None


class MarketStateEngine:
    """Produces the shared `market_state` dict. Advisory only; never raises."""

    def __init__(self):
        self.last_state: Dict[str, Any] = {}
        self.last_diagnostic: Dict[str, Any] = {}
        self.data_quality: Dict[str, Any] = {}

    def build(self, market_context: Optional[dict] = None,
              indicator_snapshot: Optional[dict] = None,
              premarket_snapshot: Optional[dict] = None,
              bars_by_symbol: Optional[dict] = None,
              vix: Any = None, breadth: Any = None,
              alpaca_quote_count: int = 0, alpaca_bar_count: int = 0,
              alpaca_data_age: Any = None, feed_type: Any = None) -> Dict[str, Any]:
        """Build the market_state. Always returns a dict (falls back on any failure)."""
        # Phase 4 data-quality provenance (advisory; never gates regime/permission).
        self.data_quality = {
            "alpaca_quote_count": int(alpaca_quote_count or 0),
            "alpaca_bar_count": int(alpaca_bar_count or 0),
            "alpaca_data_age": alpaca_data_age,
            "feed_type": feed_type,
        }
        try:
            return self._build(market_context or {}, indicator_snapshot or {},
                               premarket_snapshot or {}, bars_by_symbol or {}, vix, breadth)
        except Exception as exc:
            logger.debug("MarketStateEngine.build failed (%s) — falling back to market_context", exc)
            return self._fallback(market_context or {}, reason=f"build_error:{str(exc)[:80]}")

    def _fallback(self, mc: dict, reason: str) -> Dict[str, Any]:
        state = {k: None for k in STATE_KEYS}
        state.update({
            "regime": str(mc.get("overall_market_bias") or mc.get("regime") or "unknown").lower(),
            "risk_on": False, "risk_off": False,
            "trend_strength": 0.0, "confidence": 0.1,
            "source": "market_context_fallback", "fallback_reason": reason,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        self.last_state = state
        self.last_diagnostic = {"reason": reason, "inputs_available": {}}
        return state

    def _build(self, mc: dict, ind_snap: dict, pm_snap: dict, bars: dict,
               vix: Any, breadth: Any) -> Dict[str, Any]:
        diag: Dict[str, Any] = {"inputs_available": {}, "penalties": {}}
        avail = diag["inputs_available"]
        confidence = 1.0

        def penalize(name: str, amount: float):
            nonlocal confidence
            confidence -= amount
            diag["penalties"][name] = amount

        avail["market_context"] = bool(mc)
        if not mc:
            penalize("no_market_context", 0.20)

        regime = str(mc.get("overall_market_bias") or mc.get("regime") or "unknown").lower()

        # ── SPY / QQQ trend (prefer bars, then streaming indicators) ───────────
        def trend_for(sym: str) -> Optional[str]:
            t = _trend_from_bars((bars or {}).get(sym))
            if t is not None:
                return t
            snap = (ind_snap.get("quotes", ind_snap) or {}).get(sym) if ind_snap else None
            ind = (snap or {}).get("indicators") if isinstance(snap, dict) else None
            return _trend_from_indicators(ind)

        spy_trend = trend_for("SPY")
        qqq_trend = trend_for("QQQ")
        avail["spy_qqq_trend"] = bool(spy_trend or qqq_trend)
        if spy_trend is None and qqq_trend is None:
            penalize("no_index_trend", 0.20)

        # ── trend strength (agreement of available index trends) ───────────────
        trends = [t for t in (spy_trend, qqq_trend) if t]
        ups = sum(1 for t in trends if t == "up")
        downs = sum(1 for t in trends if t == "down")
        if trends:
            trend_strength = round((ups - downs) / len(trends), 3)  # -1..1
        else:
            trend_strength = 0.0

        # ── VIX / volatility regime ────────────────────────────────────────────
        vix_level = _f(vix)
        if vix_level is None:
            vix_level = _f(mc.get("vix") or mc.get("vix_level"))
        avail["vix"] = vix_level is not None
        if vix_level is None:
            penalize("no_vix", 0.15)
            vix_state = str(mc.get("vix_regime") or mc.get("volatility_regime") or "unknown").lower()
        elif vix_level < 15:
            vix_state = "calm"
        elif vix_level < 20:
            vix_state = "normal"
        elif vix_level < 30:
            vix_state = "elevated"
        else:
            vix_state = "high"
        volatility_regime = str(mc.get("volatility_regime") or vix_state or "unknown").lower()

        # ── breadth (explicit, else fraction of streaming names in uptrend) ─────
        breadth_val = _f(breadth)
        if breadth_val is None:
            breadth_val = _f(mc.get("breadth") or mc.get("advance_decline"))
        quotes = ind_snap.get("quotes", ind_snap) if ind_snap else {}
        if breadth_val is None and isinstance(quotes, dict) and quotes:
            ups_b = downs_b = 0
            for v in quotes.values():
                ind = v.get("indicators") if isinstance(v, dict) else None
                t = _trend_from_indicators(ind)
                if t == "up":
                    ups_b += 1
                elif t == "down":
                    downs_b += 1
            tot = ups_b + downs_b
            breadth_val = round(ups_b / tot, 3) if tot else None
        avail["breadth"] = breadth_val is not None
        if breadth_val is None:
            penalize("no_breadth", 0.10)

        # ── sector rotation ────────────────────────────────────────────────────
        sector_rotation = str(
            mc.get("sector_rotation")
            or (mc.get("sector_strength") or {}).get("sector_rotation_bias")
            or mc.get("sector_rotation_bias") or "unknown"
        ).lower() if isinstance(mc.get("sector_strength"), dict) or mc.get("sector_rotation") \
            else str(mc.get("sector_rotation_bias") or "unknown").lower()

        # ── premarket risk state ────────────────────────────────────────────────
        pm_quotes = (pm_snap or {}).get("quotes") or {}
        avail["premarket"] = bool(pm_quotes)
        premarket_risk_state = "unknown"
        if pm_quotes:
            gaps = [_f(q.get("premarket_change_pct")) for q in pm_quotes.values()
                    if not q.get("stale") and _f(q.get("premarket_change_pct")) is not None]
            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                if avg_gap > 0.3:
                    premarket_risk_state = "risk_on"
                elif avg_gap < -0.3:
                    premarket_risk_state = "risk_off"
                else:
                    premarket_risk_state = "neutral"
        else:
            penalize("no_premarket", 0.10)

        # ── risk_on / risk_off (mutually exclusive) ────────────────────────────
        bullish_regime = any(w in regime for w in ("bull", "risk_on", "uptrend", "constructive"))
        bearish_regime = any(w in regime for w in ("bear", "risk_off", "crisis", "crash", "downtrend"))
        score = 0.0
        score += trend_strength                                   # -1..1
        score += 0.5 if bullish_regime else (-0.5 if bearish_regime else 0.0)
        score += {"calm": 0.3, "normal": 0.1, "elevated": -0.2, "high": -0.5}.get(vix_state, 0.0)
        score += {"risk_on": 0.3, "risk_off": -0.3}.get(premarket_risk_state, 0.0)
        if breadth_val is not None:
            score += (breadth_val - 0.5) * 0.6
        risk_on = score > 0.25
        risk_off = score < -0.25
        if risk_on and risk_off:           # safety — should never happen
            risk_on, risk_off = False, False

        confidence = max(0.1, min(1.0, round(confidence, 3)))
        state = {
            "regime": regime,
            "trend_strength": trend_strength,
            "risk_on": bool(risk_on),
            "risk_off": bool(risk_off),
            "volatility_regime": volatility_regime,
            "sector_rotation": sector_rotation,
            "breadth": breadth_val,
            "spy_trend": spy_trend or "unknown",
            "qqq_trend": qqq_trend or "unknown",
            "vix_state": vix_state,
            "premarket_risk_state": premarket_risk_state,
            "confidence": confidence,
            "risk_score": round(score, 3),
            "source": "market_state_engine",
            "alpaca_quote_count": self.data_quality.get("alpaca_quote_count", 0),
            "alpaca_bar_count": self.data_quality.get("alpaca_bar_count", 0),
            "alpaca_data_age": self.data_quality.get("alpaca_data_age"),
            "feed_type": self.data_quality.get("feed_type"),
            "market_state_confidence": confidence,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "date_tag": DATA_TAG,
        }
        diag.update({
            "regime": regime, "risk_score": round(score, 3), "confidence": confidence,
            "vix_level": vix_level, "trend_strength": trend_strength,
            "spy_trend": spy_trend, "qqq_trend": qqq_trend,
            "generated_at_utc": state["generated_at_utc"],
        })
        self.last_state = state
        self.last_diagnostic = diag
        return state

    # ── output ────────────────────────────────────────────────────────────────
    def write_report(self, out_dir: Any = None) -> Dict[str, Path]:
        target = Path(out_dir) if out_dir else (DAILY_OUTPUTS / DATA_TAG)
        paths: Dict[str, Path] = {}
        try:
            target.mkdir(parents=True, exist_ok=True)
            sp = target / "market_state.json"
            _write_json_safe(sp, self.last_state)
            paths["state"] = sp
            dp = target / "market_state_diagnostic.json"
            _write_json_safe(dp, self.last_diagnostic)
            paths["diagnostic"] = dp
            qp = target / "market_state_data_quality.json"
            _write_json_safe(qp, {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(), "date_tag": DATA_TAG,
                "alpaca_quote_count": self.data_quality.get("alpaca_quote_count", 0),
                "alpaca_bar_count": self.data_quality.get("alpaca_bar_count", 0),
                "alpaca_data_age": self.data_quality.get("alpaca_data_age"),
                "feed_type": self.data_quality.get("feed_type"),
                "market_state_confidence": self.last_state.get("confidence"),
            })
            paths["data_quality"] = qp
        except Exception as exc:
            logger.debug("MarketStateEngine.write_report failed: %s", exc)
        return paths


_ENGINE: Optional[MarketStateEngine] = None


def get_engine() -> MarketStateEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = MarketStateEngine()
    return _ENGINE
