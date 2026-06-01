"""
rl_common.py — Shared offline RL utilities for the two-engine architecture.
===========================================================================

Single home for the pieces every RL file needs, so the dataset builder, the four
Gymnasium environments, the trainer, the evaluator, and shadow mode do not each
re-implement them:

    * dependency shims (gymnasium spaces / Env base when gymnasium is absent)
    * research-event + daily-output loading from daily_system_outputs/
    * feature extraction per agent
    * the reward functions (Step 4)
    * walk-forward metrics + the deployment gate (Step 9)
    * atomic JSON output (via opportunity_utils)

OFFLINE ONLY. Nothing here imports AlpacaGateway or submits orders.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DAILY_OUTPUTS = PROJECT_ROOT / "daily_system_outputs"
RL_DATA_DIR = PROJECT_ROOT / "data" / "rl_training"
RL_MODEL_DIR = PROJECT_ROOT / "rl_models" / "rl_v2"
RL_BACKFILL_DIR = PROJECT_ROOT / "data" / "rl_backfill"

# ── action vocabularies (advisory; engines + RiskManager remain final authority) ──
ENTRY_ACTIONS = ["enter_now", "wait", "skip", "reduce_size"]
EXIT_ACTIONS = ["hold", "tighten_trail", "partial_exit", "full_exit", "no_overnight", "allow_overnight"]
DCA_ACTIONS = ["no_dca", "small_dca", "normal_dca", "aggressive_dca", "defer"]
ALLOC_BUCKETS = ["alpha", "buffett", "cash"]

MIN_DEPLOY_SAMPLES = int(os.getenv("RL_MIN_DEPLOY_SAMPLES", "200"))
# Trade-activity floor for the overfit guard (paper §11.5: 100% win rate on ~1 trade).
MIN_ACTIVE_TRADES = int(os.getenv("RL_MIN_ACTIVE_TRADES", "30"))
MIN_ACTIVITY_RATIO = float(os.getenv("RL_MIN_ACTIVITY_RATIO", "0.05"))
# Tran et al. (2023): Sharpe is the primary selection objective. Allowed relative Sharpe drop
# vs. baseline before deployment is blocked (0.10 = a >10% risk-adjusted regression objects).
SHARPE_REGRESSION_TOL = float(os.getenv("RL_SHARPE_REGRESSION_TOL", "0.10"))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_tag() -> str:
    try:
        from opportunity_utils import get_data_tag
        return get_data_tag()
    except Exception:
        return os.getenv("RUN_DATA_DATE") or datetime.now(timezone.utc).strftime("%Y%m%d")


def to_f(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        return default if v != v else v
    except Exception:
        return default


def write_json(path: Path, payload: Any) -> bool:
    """Atomic JSON write (shared opportunity_utils impl; guarded fallback)."""
    try:
        from opportunity_utils import write_json_safe
        return write_json_safe(Path(path), payload)
    except Exception:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            os.replace(tmp, p)
            return True
        except Exception as exc:
            logger.debug("rl_common.write_json failed: %s", exc)
            return False


# ── dependency shims ──────────────────────────────────────────────────────────────
try:
    import gymnasium as _gym
    from gymnasium import spaces as _spaces
    GYM_AVAILABLE = True
    EnvBase = _gym.Env
    spaces = _spaces
except Exception:  # pragma: no cover - exercised only when gymnasium absent
    GYM_AVAILABLE = False

    class _Discrete:
        def __init__(self, n): self.n = int(n)
        def sample(self): return 0

    class _Box:
        def __init__(self, low=0.0, high=1.0, shape=(1,), dtype=None):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

        def sample(self):
            import numpy as np
            return np.zeros(self.shape, dtype=self.dtype or float)

    class spaces:  # type: ignore
        Discrete = _Discrete
        Box = _Box

    class EnvBase:  # minimal Gymnasium-compatible protocol
        metadata: Dict[str, Any] = {}
        def reset(self, *a, **k): raise NotImplementedError
        def step(self, action): raise NotImplementedError
        def close(self): pass


def sb3_available() -> bool:
    try:
        import stable_baselines3  # noqa: F401
        return True
    except Exception:
        return False


# ── research-event + daily-output loading ──────────────────────────────────────────
def load_research_events(pattern: str = None) -> List[dict]:
    """Read all research_events.jsonl across daily_system_outputs (safe on missing)."""
    pattern = pattern or str(DAILY_OUTPUTS / "*" / "research_events.jsonl")
    out: List[dict] = []
    for f in sorted(glob.glob(pattern)):
        try:
            for line in Path(f).read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        except Exception:
            continue
    return out


def load_backfill_events() -> List[dict]:
    """Read replayed research events from data/rl_backfill/**/research_events.jsonl (safe)."""
    return load_research_events(str(RL_BACKFILL_DIR / "*" / "research_events.jsonl"))


def load_daily_json(filename: str) -> List[dict]:
    """Load every daily_system_outputs/*/<filename> JSON (list of parsed payloads)."""
    out = []
    for f in sorted(glob.glob(str(DAILY_OUTPUTS / "*" / filename))):
        try:
            out.append({"_path": f, "_data": json.loads(Path(f).read_text(encoding="utf-8"))})
        except Exception:
            continue
    return out


def load_dataset(name: str, data_dir: Path = None) -> List[dict]:
    """Load a built RL dataset (parquet → csv → json) as a list of row dicts. Safe."""
    d = Path(data_dir) if data_dir else RL_DATA_DIR
    for ext, reader in (("parquet", "parquet"), ("csv", "csv"), ("json", "json")):
        p = d / f"{name}.{ext}"
        if not p.exists():
            continue
        try:
            if ext == "json":
                return json.loads(p.read_text(encoding="utf-8"))
            import pandas as pd
            df = pd.read_parquet(p) if ext == "parquet" else pd.read_csv(p)
            return df.to_dict("records")
        except Exception as exc:
            logger.debug("load_dataset(%s) failed: %s", name, exc)
    return []


# ── observation normalization (Yasin & Gill 2024, §8) ──────────────────────────────
# Their work shows inconsistent feature scaling degrades the RL model. We apply a
# DETERMINISTIC, STATELESS squash so training and inference normalise identically with
# no fitted params to persist. tanh bounds every feature to (-1, 1) and tames the few
# unbounded ones (expected_R, atr_expansion, unrealized_R) without distorting the rest.
OBS_NORM = os.getenv("RL_OBS_NORM", "tanh").lower()      # tanh | clip | none
OBS_CLIP = float(os.getenv("RL_OBS_CLIP", "5.0"))


def normalize_obs(vec: List[float], method: Optional[str] = None) -> List[float]:
    """Stateless per-feature normalization of an observation vector. Never raises."""
    m = (method or OBS_NORM)
    try:
        if m == "none":
            return [to_f(x) for x in vec]
        if m == "clip":
            c = OBS_CLIP or 1.0
            return [max(-1.0, min(1.0, to_f(x) / c)) for x in vec]
        # default: tanh squash → (-1, 1)
        return [math.tanh(to_f(x)) for x in vec]
    except Exception:
        return [to_f(x) for x in vec]


def _vol_code(s: Any) -> float:
    s = str(s or "").lower()
    return {"calm": 0.0, "normal": 0.25, "low": 0.25, "elevated": 0.6,
            "high": 1.0, "crisis": 1.0}.get(s, 0.5)


def _band_code(s: Any) -> float:
    s = str(s or "")
    return {"80-100": 0.9, "70-79": 0.75, "60-69": 0.65, "50-59": 0.55,
            "40-49": 0.45, "0-39": 0.2}.get(s, 0.5)


def _tod_frac(ts: Any) -> float:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        mins = dt.hour * 60 + dt.minute
        return max(0.0, min(1.0, (mins - 13 * 60 - 30) / (6.5 * 60)))  # 0 at open .. 1 at close (UTC/EDT)
    except Exception:
        return 0.5


# ── multi-timeframe (MTF) trend features (Roy 2024 MSc thesis, TDH-DQN) ──────────────
# The thesis's central robustness idea: judge trend across MULTIPLE timeframes (the bots
# read 5s/5min/daily/weekly bars and a price-vs-EMA/SMA relation per timeframe, then trade
# WITH the alignment). We encode that as the signed price-vs-moving-average over fast/mid/
# slow windows on whatever bar series is available, plus an alignment score in [-1, 1].
# Stateless; absent series → neutral zeros (fully back-compatible with old events).
MTF_FEATURES = ["mtf_trend_fast", "mtf_trend_mid", "mtf_trend_slow", "mtf_alignment"]

# Bar-context enrichment fields (rl_alpaca_bar_enrichment) — relative volume, VWAP distance,
# ATR %, and market/sector regime. Appended to the alpha observation sets; neutral when the
# enrichment had no bars (rel_volume defaults to 1.0 = average → feature value 0.0).
ENRICH_EXTRA_FEATURES = ["rel_volume", "vwap_distance", "atr_pct", "spy_regime", "sector_regime"]


def _sma_last(xs: List[float], w: int) -> Optional[float]:
    if not xs or w <= 0:
        return None
    seg = xs[-w:] if len(xs) >= w else xs
    return sum(seg) / len(seg) if seg else None


def mtf_trend_features(closes, windows: Tuple[int, int, int] = (10, 50, 200)) -> Dict[str, float]:
    """Price-vs-SMA trend across fast/mid/slow windows + an alignment score in [-1, 1].

    A multi-timeframe trend proxy (Roy 2024): a long SMA window captures the higher
    timeframe, a short window the lower one. Returns neutral zeros when the series is thin
    so callers without bar history (e.g. trade-replay CSVs) stay back-compatible. Never raises.
    """
    out = {k: 0.0 for k in MTF_FEATURES}
    try:
        cs = [float(c) for c in closes if c is not None]
    except Exception:
        return out
    if len(cs) < 2:
        return out
    price = cs[-1]
    if price <= 0:
        return out
    trends: List[float] = []
    for key, w in zip(("mtf_trend_fast", "mtf_trend_mid", "mtf_trend_slow"), windows):
        sma = _sma_last(cs, min(int(w), len(cs)))
        if sma and sma > 0:
            t = (price - sma) / sma                       # signed trend strength vs that timeframe
            out[key] = round(t, 6)
            trends.append(1.0 if t > 0 else (-1.0 if t < 0 else 0.0))
    if trends:
        out["mtf_alignment"] = round(sum(trends) / len(trends), 4)   # -1 all-down .. +1 all-up
    return out


# ── "Generating Alpha" hybrid technical + sentiment block ───────────────────────────
# Pillai, Ajith & Sumesh (ComSIA 2026), "Generating Alpha: A Hybrid AI-Driven Trading
# System": fuse trend (EMA50/200, MACD), mean-reversion (RSI, Bollinger), per-stock regime
# (SMA20 of returns) and FinBERT-style sentiment into a multi-modal CONFLUENCE score, and
# treat sentiment + regime as GATES (suppress / exit), never as standalone buy signals.
# We encode the same interpretable features, computed from price/sentiment context. All
# functions are pure + stateless; thin/absent series degrade to neutral (never raise).
SENT_GATE_MIN = float(os.getenv("RL_SENT_GATE_MIN", "-0.70"))   # paper §D: suspend/exit below this
MIN_CONFLUENCE = int(os.getenv("RL_MIN_CONFLUENCE", "2"))       # paper Alg.1: entry needs score ≥ 2

# Interpretable interaction features appended to the alpha-entry observation (neutral 0.0
# when absent → fully back-compatible with events that predate the enrichment).
GA_FEATURES = ["ga_ema_ratio", "ga_macd_hist", "ga_rsi", "ga_bollinger_pctb",
               "ga_regime_switch", "ga_sentiment_st"]


def ema(values: List[float], period: int) -> Optional[float]:
    """Last EMA(period) of a value series (None if not computable). Pure."""
    try:
        xs = [float(v) for v in values if v is not None]
    except Exception:
        return None
    if not xs or period <= 0:
        return None
    k = 2.0 / (period + 1.0)
    e = xs[0]
    for v in xs[1:]:
        e = v * k + e * (1.0 - k)
    return e


def ema50_200_ratio(closes: List[float]) -> float:
    """EMA50/EMA200 − 1 interaction term (paper §C). >0 ≈ bullish trend structure. 0.0 if thin."""
    e50, e200 = ema(closes, 50), ema(closes, 200)
    if e50 is None or e200 is None or e200 == 0:
        return 0.0
    return round(e50 / e200 - 1.0, 6)


def macd_histogram(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> float:
    """MACD histogram (MACD − signal) as a FRACTION of last price (paper §C). 0.0 if thin."""
    try:
        cs = [float(c) for c in closes if c is not None]
    except Exception:
        return 0.0
    if len(cs) < slow + signal or cs[-1] <= 0:
        return 0.0
    macd_line = []
    for i in range(slow, len(cs) + 1):
        seg = cs[:i]
        ef, es = ema(seg, fast), ema(seg, slow)
        if ef is None or es is None:
            continue
        macd_line.append(ef - es)
    if len(macd_line) < signal:
        return 0.0
    sig = ema(macd_line, signal)
    if sig is None:
        return 0.0
    return round((macd_line[-1] - sig) / cs[-1], 6)


def rsi(closes: List[float], period: int = 14) -> float:
    """Wilder RSI(period) in [0, 100] (paper §C). 50.0 (neutral) if not computable."""
    try:
        cs = [float(c) for c in closes if c is not None]
    except Exception:
        return 50.0
    if len(cs) <= period:
        return 50.0
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = cs[i] - cs[i - 1]
        gains += max(0.0, d)
        losses += max(0.0, -d)
    avg_gain, avg_loss = gains / period, losses / period
    for i in range(period + 1, len(cs)):
        d = cs[i] - cs[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(0.0, d)) / period
        avg_loss = (avg_loss * (period - 1) + max(0.0, -d)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 4)


def bollinger_pctb(closes: List[float], period: int = 20, k: float = 2.0) -> float:
    """Bollinger %b in [0,1]-ish: position within the bands (paper §C). 0.5 (mid) if thin."""
    try:
        cs = [float(c) for c in closes if c is not None]
    except Exception:
        return 0.5
    if len(cs) < period:
        return 0.5
    seg = cs[-period:]
    mid = sum(seg) / period
    var = sum((x - mid) ** 2 for x in seg) / period
    sd = math.sqrt(var)
    if sd == 0:
        return 0.5
    upper, lower = mid + k * sd, mid - k * sd
    return round((cs[-1] - lower) / (upper - lower), 4)


def regime_switch(closes: List[float], window: int = 20) -> float:
    """Per-stock regime (paper §F): sign of SMA_window(pct_change). +1 bullish / −1 bearish /
    0 neutral|thin. Used as a long-only ON/OFF switch, never as a buy signal."""
    try:
        cs = [float(c) for c in closes if c is not None]
    except Exception:
        return 0.0
    if len(cs) < 3:
        return 0.0
    rets = [(cs[i] - cs[i - 1]) / cs[i - 1] for i in range(1, len(cs)) if cs[i - 1] > 0]
    if not rets:
        return 0.0
    seg = rets[-window:] if len(rets) >= window else rets
    avg = sum(seg) / len(seg)
    return 1.0 if avg > 0 else (-1.0 if avg < 0 else 0.0)


def sentiment_to_st(score: Any) -> float:
    """Map a per-symbol sentiment score to the paper's St ∈ [−1, 1] (S_t = mean(P_pos − P_neg)).
    Accepts an already-signed [−1,1] score (passed through, clamped) or a [0,1] LLM/probability
    score (rescaled 2·s−1). None → 0.0 (neutral; missing data never reads as bearish)."""
    if score is None:
        return 0.0
    s = to_f(score, 0.0)
    if s < 0.0:                       # already a signed [-1,1] score
        return max(-1.0, s)
    if s > 1.0:                       # out of range → clamp signed
        return min(1.0, s)
    return round(2.0 * s - 1.0, 6)    # [0,1] LLM score → [-1,1]


def generating_alpha_confluence(row: Dict[str, Any]) -> Dict[str, Any]:
    """Multi-modal confluence count (paper Alg.1 Step 3): how many INDEPENDENT bullish
    confirmations align — trend, momentum, mean-reversion, volume, regime, sentiment. Uses
    whatever fields are present (GA enrichment and/or legacy MTF/enrich), so it has teeth on
    current datasets and gets sharper as enrichment fills in. Returns count + breakdown +
    the sentiment St and regime so callers can gate without recomputing."""
    st = sentiment_to_st(row.get("ga_sentiment_st", row.get("sentiment_score")))
    reg = to_f(row.get("ga_regime_switch"))
    rsi_v = to_f(row.get("ga_rsi"), 50.0)
    checks = {
        "trend":     to_f(row.get("ga_ema_ratio")) > 0 or to_f(row.get("mtf_alignment")) > 0,
        "momentum":  to_f(row.get("ga_macd_hist")) > 0,
        "meanrev":   0.0 < rsi_v < 45.0 or 0.0 < to_f(row.get("ga_bollinger_pctb"), 0.5) < 0.2,
        "volume":    to_f(row.get("rel_volume"), 1.0) > 1.2 or to_f(row.get("catalyst_strength")) > 0.3,
        "regime":    reg > 0 or to_f(row.get("sector_regime")) > 0 or to_f(row.get("spy_regime")) > 0,
        "sentiment": st > 0.0,
    }
    return {"confluence": sum(1 for v in checks.values() if v), "checks": checks,
            "sentiment_st": round(st, 4), "regime_switch": reg}


def _ga_vals(e: dict) -> List[float]:
    """The six GA feature values from an event, centered/scaled for the RL observation
    (neutral 0.0 when absent). Order matches GA_FEATURES."""
    return [
        to_f(e.get("ga_ema_ratio")),
        to_f(e.get("ga_macd_hist")),
        (to_f(e.get("ga_rsi"), 50.0) - 50.0) / 50.0,            # 0..100 → -1..1 (oversold negative)
        (to_f(e.get("ga_bollinger_pctb"), 0.5) - 0.5) * 2.0,    # 0..1   → -1..1 (below-band negative)
        to_f(e.get("ga_regime_switch")),                        # {-1, 0, 1}
        sentiment_to_st(e.get("ga_sentiment_st", e.get("sentiment_score"))),
    ]


# ── Opening Range entry assessment  (Bysshe, "The Opening Range Handbook", 2009) ─────
# The handbook reduces "is this a good entry?" to THREE questions about the opening range
# (first 30 min H/L), which we encode as bounded numeric features the options-entry RL agent
# can learn from (it currently has zero opening-range awareness):
#   Q1  Is there a WELL-DEFINED support/resistance at the OR high/low?  → or_level_quality
#   Q2  Is there a distinctly BULLISH/BEARISH bias to the OR?           → or_position, or_breakout
#   Q3  Is the OR VOLUME unusually high (interest → explosive)?         → or_volume_confirm
# Statistical edge the book cites: the day's high/low forms in the first 30 min ~35% of the
# time vs ~8% random — so OR extremes are high-probability inflection levels. Pure; thin/absent
# data → neutral 0.0 (never fabricated).
OR_FEATURES = ["or_position", "or_breakout", "or_volume_confirm", "or_level_quality"]


def compute_or_assessment(orb_high: float, orb_low: float, price: float,
                          rel_volume: float = 1.0, level_touches_frac: float = 0.0,
                          atr_abs: float = 0.0) -> Dict[str, float]:
    """The OR Handbook's three-question assessment as numeric features. orb_high/orb_low are
    the first-30-min range; price is current. rel_volume is OR-period volume / baseline (Q3);
    level_touches_frac is the fraction of OR bars that tested an extreme (Q1); atr_abs gives a
    tightness fallback for Q1. Returns neutral zeros if the range is degenerate. Never raises."""
    out = {k: 0.0 for k in OR_FEATURES}
    H, L, P = to_f(orb_high), to_f(orb_low), to_f(price)
    width = H - L
    if H <= 0 or L <= 0 or width <= 0 or P <= 0:
        return out
    mid = 0.5 * (H + L)
    # Q2 bias: signed position within/beyond the OR. >1 ≈ at OR high (bullish), <−1 ≈ at OR low.
    out["or_position"] = round(max(-1.5, min(1.5, (P - mid) / (0.5 * width))), 4)
    out["or_breakout"] = 1.0 if P > H else (-1.0 if P < L else 0.0)   # breakout/breakdown state
    # Q3 volume confirmation: OR-period relative volume centered (1.0 = average → 0.0).
    out["or_volume_confirm"] = round(max(0.0, min(1.0, (to_f(rel_volume, 1.0) - 1.0) / 2.0)), 4)
    # Q1 well-defined level: touch-tested extreme dominates; a tight OR vs ATR is a fallback proxy.
    tq = max(0.0, min(1.0, to_f(level_touches_frac)))
    if tq <= 0.0 and atr_abs > 0:
        tq = max(0.0, min(1.0, 1.0 - width / (2.0 * atr_abs)))        # tight range → cleaner level
    out["or_level_quality"] = round(tq, 4)
    return out


def opening_range_features(highs: List[float], lows: List[float], closes: List[float],
                           volumes: List[float], price: Optional[float] = None,
                           baseline_volume: Optional[float] = None,
                           atr_abs: float = 0.0) -> Dict[str, float]:
    """Compute the OR assessment from the OR-WINDOW bars (the first-30-min slice of a 1m/5m
    frame). H/L are the window extremes; level-quality counts bars that tested an extreme
    (within 15% of the range); rel_volume = OR mean volume / baseline. Neutral on thin data."""
    try:
        h = [float(x) for x in highs]; l = [float(x) for x in lows]
        c = [float(x) for x in closes]; v = [float(x) for x in volumes]
    except Exception:
        return {k: 0.0 for k in OR_FEATURES}
    if len(h) < 2 or len(l) < 2:
        return {k: 0.0 for k in OR_FEATURES}
    H, L = max(h), min(l)
    width = H - L
    p = to_f(price) if price is not None else (c[-1] if c else 0.0)
    touches = 0
    if width > 0:
        band = 0.15 * width
        for hi, lo in zip(h, l):
            if hi >= H - band or lo <= L + band:
                touches += 1
    touches_frac = touches / len(h) if h else 0.0
    or_vol = sum(v) / len(v) if v else 0.0
    rel_vol = (or_vol / baseline_volume) if (baseline_volume and baseline_volume > 0) else 1.0
    return compute_or_assessment(H, L, p, rel_vol, touches_frac, atr_abs)


def _or_vals(e: dict) -> List[float]:
    """The four OR feature values from an event (neutral 0.0 when absent). If the event carries
    orb_high/orb_low + last_price but not the derived scores, derive them on the fly."""
    if any(e.get(k) is not None for k in OR_FEATURES):
        return [to_f(e.get(k)) for k in OR_FEATURES]
    H, L = e.get("orb_high"), e.get("orb_low")
    if H is not None and L is not None:
        a = compute_or_assessment(H, L, e.get("last_price", e.get("price")),
                                  to_f(e.get("rel_volume"), 1.0),
                                  to_f(e.get("or_level_touches_frac")), to_f(e.get("atr_abs")))
        return [a[k] for k in OR_FEATURES]
    return [0.0 for _ in OR_FEATURES]


# ── OOPS payoff-shape features  (Faias & Santa-Clara, "Optimal Option Portfolio Strategies") ─
# The OOPS layer (oops_option_optimizer.py) already maximizes CRRA power-utility over a
# simulated payoff distribution and collapses the result into a 0–100 `oops_score`. That scalar
# blends away the two quantities the paper's utility actually trades off: the certainty-equivalent
# return (the risk-adjusted core of the objective) and the probability of TOTAL LOSS (the tail
# the γ=10 utility penalizes hardest). Exposing them lets the RL learn option PLACEMENT — which
# contract / how much — on the same risk-aware terms the paper optimizes, not just a scalar.
OOPS_EXTRA_FEATURES = ["oops_cert_equiv_return", "oops_prob_total_loss"]


def _oops_extra_vals(e: dict) -> List[float]:
    """OOPS certainty-equivalent return (signed, ~[-0.5, 0.5]) and probability of total loss
    ([0, 1], a downside signal). Neutral 0.0 when the OOPS layer didn't run for this event."""
    return [
        to_f(e.get("oops_cert_equiv_return", e.get("certainty_equivalent_return"))),
        max(0.0, min(1.0, to_f(e.get("oops_prob_total_loss", e.get("probability_total_loss"))))),
    ]


# ── predictive state compression features  (Elliott et al. 2020, quantum modelling) ──
# The compressed CAUSAL-STATE embeddings from predictive_state_compression.py: a handful of
# latent coordinates that carry the predictive information about future returns while discarding
# historical noise (the ε-machine "minimal sufficient statistic of the past for the future",
# which mirrors RL belief states). High-priority RL inputs — but wired in ONLY when
# RL_PSC_FEATURES=1 AND a representation has been APPROVED out-of-sample (validation discipline:
# no unvalidated latent state enters the observation / live behavior). Default OFF, so existing
# model widths and datasets are unchanged unless explicitly enabled.
PSC_FEATURES = ["predictive_state_1", "predictive_state_2", "predictive_state_3",
                "latent_market_state", "future_return_embedding", "state_confidence",
                "predictive_information_score"]
PSC_ENABLED = os.getenv("RL_PSC_FEATURES", "0") == "1"


def _psc_vals(e: dict) -> List[float]:
    """The predictive-state embedding values from an event (neutral 0.0 when absent).
    latent_market_state is a discrete causal-state id (kept raw; tanh-normalized downstream)."""
    return [
        to_f(e.get("predictive_state_1")),
        to_f(e.get("predictive_state_2")),
        to_f(e.get("predictive_state_3")),
        to_f(e.get("latent_market_state")),
        to_f(e.get("future_return_embedding")),
        to_f(e.get("state_confidence"), 0.5),
        to_f(e.get("predictive_information_score")),
    ]


# ── Alpha Vantage realtime features (price + technical + options hidden alpha) ────────
# Normalized, schema-controlled realtime features produced by
# alpha_discovery.live_feature_transformer from the Alpha Vantage realtime entitlement
# (intraday, global quote, technical indicators, REALTIME_OPTIONS, put-call ratio,
# volume/OI ratio). These are the §7 "new hidden alpha" realtime signals. Wired into the
# observation ONLY when RL_REALTIME_AV_FEATURES=1 — default OFF so existing model widths
# and datasets are unchanged unless explicitly enabled. Raw API payloads NEVER enter the
# state; only these bounded, normalized features do.
AV_REALTIME_FEATURES = ["realtime_quote_momentum", "realtime_intraday_vwap_distance",
                        "realtime_intraday_volume_acceleration", "realtime_technical_alignment_score",
                        "realtime_option_iv_skew", "realtime_option_delta_exposure",
                        "realtime_option_gamma_exposure", "realtime_put_call_sentiment_score",
                        "realtime_volume_oi_unusual_activity_score", "realtime_options_liquidity_score"]
AV_REALTIME_ENABLED = os.getenv("RL_REALTIME_AV_FEATURES", "0") == "1"


def _av_realtime_vals(e: dict) -> List[float]:
    """Realtime Alpha Vantage feature values (neutral 0.0 when the bundle is absent).
    All are already bounded/centered by the builders; tanh-normalized downstream."""
    return [
        to_f(e.get("realtime_quote_momentum")),
        to_f(e.get("realtime_intraday_vwap_distance")) / 100.0,   # % -> fraction
        to_f(e.get("realtime_intraday_volume_acceleration")),
        to_f(e.get("realtime_technical_alignment_score")),
        to_f(e.get("realtime_option_iv_skew")),
        to_f(e.get("realtime_option_delta_exposure")),
        to_f(e.get("realtime_option_gamma_exposure")),
        to_f(e.get("realtime_put_call_sentiment_score")),
        to_f(e.get("realtime_volume_oi_unusual_activity_score")),
        to_f(e.get("realtime_options_liquidity_score")),
    ]


# ── Alpha Vantage NEWS_SENTIMENT features (windowed news + catalyst + compressed text) ─
# Approved compact subset of alpha_discovery.news_feature_builder /
# news_embedding_engine output. Only validated latent embeddings (never raw text) enter
# the state. Wired in ONLY when RL_NEWS_FEATURES=1 — default OFF. On any news failure the
# builders return neutral values, so this family is safe to leave enabled.
NEWS_FEATURES = ["sentiment_mean_60m", "sentiment_change_30m", "sentiment_acceleration",
                 "news_density_score", "catalyst_strength_score",
                 "hidden_news_continuation_score", "hidden_sentiment_momentum_score",
                 "compressed_news_embedding_1", "compressed_news_embedding_2",
                 "compressed_news_embedding_3"]
NEWS_ENABLED = os.getenv("RL_NEWS_FEATURES", "0") == "1"


def _news_vals(e: dict) -> List[float]:
    """Approved news feature values (neutral 0.0 when news is unavailable)."""
    return [
        to_f(e.get("sentiment_mean_60m")),
        to_f(e.get("sentiment_change_30m")),
        to_f(e.get("sentiment_acceleration")),
        to_f(e.get("news_density_score")),
        max(0.0, min(1.0, to_f(e.get("catalyst_strength_score")))),
        to_f(e.get("hidden_news_continuation_score")),
        to_f(e.get("hidden_sentiment_momentum_score")),
        to_f(e.get("compressed_news_embedding_1")),
        to_f(e.get("compressed_news_embedding_2")),
        to_f(e.get("compressed_news_embedding_3")),
    ]


# ── Alpha Vantage Advanced Analytics features (fixed/sliding-window statistics) ───────
# Approved compact subset of the alpha_discovery.analytics_feature_builder output produced
# from the Advanced Analytics endpoints (ANALYTICS_FIXED_WINDOW / ANALYTICS_SLIDING_WINDOW):
# return distribution, drawdown, autocorrelation (momentum vs mean-reversion), correlation /
# covariance to SPY/QQQ, rolling volatility/correlation slopes, and the §5 hidden-alpha
# regime scores. These give the RL state a statistical-edge / risk-regime read. Wired into
# the observation ONLY when RL_ANALYTICS_FEATURES=1 — default OFF, so existing model widths and
# datasets are unchanged unless explicitly enabled (mirrors PSC / AV realtime / news gating).
# Raw API payloads NEVER enter the state; only these bounded, normalized features do.
ANALYTICS_FEATURES = ["fixed_autocorrelation_lag1", "fixed_cumulative_return", "fixed_correlation_to_spy",
                      "rolling_volatility_slope", "rolling_correlation_slope", "rolling_risk_regime_score",
                      "hidden_risk_adjusted_momentum_score", "hidden_volatility_expansion_score",
                      "hidden_relationship_shift_score", "hidden_market_independence_score"]
ANALYTICS_ENABLED = os.getenv("RL_ANALYTICS_FEATURES", "0") == "1"


def _analytics_vals(e: dict) -> List[float]:
    """Approved Advanced Analytics feature values (neutral 0.0 when absent).

    Builder outputs are already bounded/centered; the two unbounded statistical reads
    (cumulative return, vol slope) are clamped here so the observation stays well-scaled."""
    def clamp(x, lo=-1.0, hi=1.0):
        return max(lo, min(hi, x))
    return [
        clamp(to_f(e.get("fixed_autocorrelation_lag1"))),
        clamp(to_f(e.get("fixed_cumulative_return")) * 5.0),     # ~±20% -> ±1
        clamp(to_f(e.get("fixed_correlation_to_spy"))),
        clamp(to_f(e.get("rolling_volatility_slope")) * 50.0),
        clamp(to_f(e.get("rolling_correlation_slope")) * 50.0),
        max(0.0, min(1.0, to_f(e.get("rolling_risk_regime_score")))),
        clamp(to_f(e.get("hidden_risk_adjusted_momentum_score"))),
        max(0.0, min(1.0, to_f(e.get("hidden_volatility_expansion_score")))),
        max(0.0, min(1.0, to_f(e.get("hidden_relationship_shift_score")))),
        max(0.0, min(1.0, to_f(e.get("hidden_market_independence_score")))),
    ]


# ── Alpha Vantage options-sentiment latent features (hidden options-flow alpha) ───────
# Compressed latent states discovered by alpha_discovery.latent_feature_discovery from the
# `options_sentiment` candidate-alpha family (PCR pressure, vol/OI activity, gamma/delta
# pressure, IV skew/term-structure). Only the validated, compressed LATENT states enter the
# observation here (the raw factors stay research-layer), mirroring the news/PSC pattern.
# Wired in ONLY when RL_OPTIONS_SENTIMENT_FEATURES=1 — default OFF, so existing model widths
# and datasets are unchanged unless explicitly enabled (and only after a representation is
# APPROVED out-of-sample via alpha_discovery.options_alpha_discovery). The live booster overlay
# (options_sentiment_av.py) is independent of this family.
OPTIONS_SENTIMENT_FEATURES = ["latent_options_factor_1", "latent_options_factor_2",
                              "latent_options_factor_3", "latent_options_sentiment_state",
                              "latent_gamma_state"]
OPTIONS_SENTIMENT_ENABLED = os.getenv("RL_OPTIONS_SENTIMENT_FEATURES", "0") == "1"


def _options_sentiment_vals(e: dict) -> List[float]:
    """Approved options-sentiment latent values (neutral 0.0 when absent).
    Latent coords are already centered by PCA; tanh-normalized downstream."""
    return [
        to_f(e.get("latent_options_factor_1")),
        to_f(e.get("latent_options_factor_2")),
        to_f(e.get("latent_options_factor_3")),
        to_f(e.get("latent_options_sentiment_state")),
        to_f(e.get("latent_gamma_state")),
    ]


# ── feature extraction (one fixed-width vector per agent) ──────────────────────────
# MTF trend features + catalyst strength (Roy 2024) are APPENDED to the base feature sets.
# Order matters: the dataset builder maps features by index against these lists, so the
# *_features() extractors below append the matching values in the SAME order.
ALPHA_ENTRY_FEATURES = ["setup_score", "black_scholes_score", "oops_score", "optionlab_score",
                        "probability_of_profit", "expected_R", "iv_rank", "market_state_confidence",
                        "premarket_gap_pct", "premarket_rank_score", "vol_regime", "time_of_day"] \
                       + MTF_FEATURES + ["catalyst_strength"] + ENRICH_EXTRA_FEATURES + GA_FEATURES \
                       + OR_FEATURES + OOPS_EXTRA_FEATURES \
                       + (PSC_FEATURES if PSC_ENABLED else []) \
                       + (AV_REALTIME_FEATURES if AV_REALTIME_ENABLED else []) \
                       + (NEWS_FEATURES if NEWS_ENABLED else []) \
                       + (ANALYTICS_FEATURES if ANALYTICS_ENABLED else []) \
                       + (OPTIONS_SENTIMENT_FEATURES if OPTIONS_SENTIMENT_ENABLED else [])   # realtime AV + news + analytics + options-sentiment latents, appended only when enabled
ALPHA_EXIT_FEATURES = ["unrealized_R", "premium_ratio", "stop_distance", "trail_high_ratio",
                       "black_scholes_score", "oops_score", "probability_band", "market_state_confidence",
                       "time_to_close", "overnight_confidence"] + MTF_FEATURES + ENRICH_EXTRA_FEATURES
BUFFETT_DCA_FEATURES = ["bal_score", "conviction_score", "intrinsic_value_discount", "composite_dca_score",
                        "optimized_target_weight", "current_weight", "bl_posterior_return",
                        "market_state_confidence", "rsi", "atr_expansion", "premarket_gap_pct", "sector_rot",
                        # AV insider trend signal (optional/backward-compatible; 0.0 when absent)
                        "insider_trend_score", "insider_confidence_modifier"] \
                       + MTF_FEATURES
ALLOC_FEATURES = ["alpha_expectancy", "alpha_perf", "buffett_quality", "market_state_confidence",
                  "drawdown", "volatility", "waterfall_ratio", "optimizer_perf"]


def _mtf_vals(e: dict) -> List[float]:
    """The four MTF feature values from an event (neutral 0.0 when absent)."""
    return [to_f(e.get(k)) for k in MTF_FEATURES]


def _enrich_vals(e: dict) -> List[float]:
    """The five bar-context enrichment values (neutral when absent). rel_volume is centered
    on its average (value − 1.0) so 'no data'/'average volume' both map to 0.0."""
    return [
        to_f(e.get("rel_volume"), 1.0) - 1.0,
        to_f(e.get("vwap_distance")),
        to_f(e.get("atr_pct")),
        to_f(e.get("spy_regime")),
        to_f(e.get("sector_regime")),
    ]


def alpha_entry_features(e: dict) -> List[float]:
    return [
        to_f(e.get("setup_score")) / 100.0,
        to_f(e.get("black_scholes_score")) / 100.0,
        to_f(e.get("oops_score")) / 100.0,
        to_f(e.get("optionlab_score")) / 100.0,
        to_f(e.get("probability_of_profit")),
        to_f(e.get("expected_R")),
        to_f(e.get("iv_rank")) / 100.0,
        to_f(e.get("market_state_confidence"), 0.5),
        to_f(e.get("premarket_gap_pct")) / 10.0,
        to_f(e.get("premarket_rank_score")) / 100.0,
        _vol_code(e.get("vol_regime")),
        _tod_frac(e.get("timestamp")),
    ] + _mtf_vals(e) + [
        max(0.0, min(1.0, to_f(e.get("catalyst_strength")))),   # §3.9 catalyst-stock focus
    ] + _enrich_vals(e) + _ga_vals(e) + _or_vals(e) + _oops_extra_vals(e) \
      + (_psc_vals(e) if PSC_ENABLED else []) \
      + (_av_realtime_vals(e) if AV_REALTIME_ENABLED else []) \
      + (_news_vals(e) if NEWS_ENABLED else []) \
      + (_analytics_vals(e) if ANALYTICS_ENABLED else []) \
      + (_options_sentiment_vals(e) if OPTIONS_SENTIMENT_ENABLED else [])


def alpha_exit_features(e: dict) -> List[float]:
    rmult = to_f(e.get("r_multiple", e.get("realized_R")))
    return [
        rmult,
        1.0 + 0.1 * rmult,                       # premium_ratio proxy
        max(0.0, 1.0 - 0.3 * rmult),             # stop_distance proxy
        max(0.0, rmult / 3.0),                   # trail_high_ratio proxy
        to_f(e.get("black_scholes_score")) / 100.0,
        to_f(e.get("oops_score")) / 100.0,
        _band_code(e.get("probability_band")),
        to_f(e.get("market_state_confidence"), 0.5),
        _tod_frac(e.get("timestamp")),
        to_f(e.get("probability_of_profit"), 0.5),
    ] + _mtf_vals(e) + _enrich_vals(e)


def buffett_dca_features(e: dict) -> List[float]:
    p2iv = to_f(e.get("price_to_intrinsic_value"), 1.0)
    return [
        to_f(e.get("bal_score")) / 100.0,
        to_f(e.get("conviction_score")) / 100.0,
        max(0.0, 1.0 - p2iv) if p2iv > 0 else 0.0,    # intrinsic value discount
        to_f(e.get("composite_dca_score")) / 100.0,
        to_f(e.get("optimized_target_weight")),
        to_f(e.get("current_weight")),
        to_f(e.get("bl_posterior_return")),
        to_f(e.get("market_state_confidence"), 0.5),
        to_f(e.get("RSI", e.get("rsi")), 50.0) / 100.0,
        to_f(e.get("ATR_expansion", e.get("atr_expansion")), 1.0),
        to_f(e.get("premarket_gap_pct")) / 10.0,
        _vol_code(e.get("sector_rotation")),
        # AV insider trend (optional/backward-compatible; 0.0 when key absent)
        to_f(e.get("av_insider_trend_score", e.get("insider_trend_score")), 0.0),
        to_f(e.get("av_insider_confidence_modifier", e.get("insider_confidence_modifier")), 0.0) / 10.0,
    ] + _mtf_vals(e)


# ── reward functions (Step 4) ──────────────────────────────────────────────────────
# ── entry reward shaping coefficients (env-tunable) ──────────────────────────────────
# Rebalanced 2026-05-31 to fix the entry agent's no-trade collapse. On the offline dataset
# (~65% losers / 33% large winners), the prior coefficients (dd_pen=0.5 over-penalising a
# loser entry at 1.5x, and skip loss-avoidance = +0.5|r| with no cap) made CONSTANT-SKIP the
# unconditional reward-maximum (mean skip +0.055 > enter -0.005) — inverting the evaluation
# objective (enter→realized_R / skip→0), so PPO correctly learned to skip everything and then
# scored 0 expectancy on the walk-forward eval (do_not_deploy). The rebalanced defaults flip
# the constant-action ordering back (enter +0.050 > skip -0.159) while keeping the
# enter-winners/skip-losers ORACLE the clear maximum (+0.88). All coefficients are env-tunable;
# ENTRY_REWARD_MODE=legacy restores the exact previous shaping for reproducibility / A-B.
def _entry_reward_cfg() -> Dict[str, float]:
    if os.getenv("ENTRY_REWARD_MODE", "aligned").lower() == "legacy":
        return {"slippage": 0.05, "dd_pen": 0.5, "win_bonus": 0.5,
                "skip_avoid": 0.5, "skip_cap": 1e9, "miss": 0.25}
    return {
        "slippage": float(os.getenv("ENTRY_REWARD_SLIPPAGE", "0.05")),
        "dd_pen": float(os.getenv("ENTRY_REWARD_DD_PEN", "0.2")),
        "win_bonus": float(os.getenv("ENTRY_REWARD_WIN_BONUS", "0.5")),
        "skip_avoid": float(os.getenv("ENTRY_REWARD_SKIP_AVOID", "0.2")),
        "skip_cap": float(os.getenv("ENTRY_REWARD_SKIP_CAP", "0.5")),
        "miss": float(os.getenv("ENTRY_REWARD_MISS", "0.3")),
    }


def alpha_entry_reward(action: str, realized_R: float,
                       slippage: Optional[float] = None, dd_pen: Optional[float] = None) -> float:
    """realized_R - slippage - drawdown_penalty, with avoid-loser / take-winner bonuses.

    Coefficients resolve from _entry_reward_cfg() (env-tunable) unless explicitly passed.
    The reward ordering is aligned with the eval objective so the agent is incentivised to
    ENTER winners rather than collapse to a no-trade policy (see _entry_reward_cfg notes)."""
    cfg = _entry_reward_cfg()
    slippage = cfg["slippage"] if slippage is None else slippage
    dd_pen = cfg["dd_pen"] if dd_pen is None else dd_pen
    r = float(realized_R)
    if action == "enter_now":
        base = r - slippage - (dd_pen * abs(min(0.0, r)))
        return base + (cfg["win_bonus"] if r >= 1.5 else 0.0)   # bonus: took a high-R winner
    if action == "reduce_size":
        return 0.5 * r - 0.5 * slippage - 0.5 * dd_pen * abs(min(0.0, r))
    if action == "skip":
        # +avoid loser (capped) / -missed winner
        return cfg["skip_avoid"] * min(abs(r), cfg["skip_cap"]) if r < 0 else -cfg["miss"] * max(0.0, r)
    if action == "wait":
        return -0.1 + (0.2 * r if r > 0 else 0.0)               # small opportunity cost
    return 0.0


def alpha_exit_reward(action: str, realized_R: float, peak_R: float,
                      overnight_loss: bool = False, hold_pen: float = 0.05) -> float:
    """realized_R - giveback_from_peak - holding_risk, with winner/overnight bonuses."""
    r = float(realized_R)
    giveback = max(0.0, float(peak_R) - r)
    if action in ("full_exit",):
        base = r - hold_pen
    elif action == "partial_exit":
        base = 0.6 * r + 0.4 * peak_R - giveback * 0.3 - hold_pen
    elif action == "tighten_trail":
        base = r - giveback * 0.5 - hold_pen
    elif action == "hold":
        base = r - giveback - hold_pen
    elif action == "no_overnight":
        base = r - hold_pen + (0.5 if overnight_loss else -0.1)
    elif action == "allow_overnight":
        base = r - hold_pen - (0.5 if overnight_loss else 0.0)
    else:
        base = r
    if r >= 2.5 and action in ("partial_exit", "tighten_trail", "hold"):
        base += 0.5                                          # captured a large winner
    return base


def buffett_dca_reward(action: str, fwd_return: float, valuation_improvement: float,
                       current_weight: float, optimized_target: float) -> float:
    """forward_return + valuation_improvement - drawdown - overconcentration."""
    scale = {"no_dca": 0.0, "defer": 0.0, "small_dca": 0.5,
             "normal_dca": 1.0, "aggressive_dca": 1.5}.get(action, 0.0)
    fr = float(fwd_return)
    overconc = 0.0
    if optimized_target > 0 and (current_weight + 0.02 * scale) > optimized_target:
        overconc = 0.5 * scale                                # penalise buying past target
    dd_pen = 0.5 * abs(min(0.0, fr)) * scale
    return scale * fr + 0.3 * valuation_improvement * scale - dd_pen - overconc


def capital_alloc_reward(weights: List[float], bucket_returns: List[float],
                         drawdown: float = 0.0, turnover: float = 0.0) -> float:
    """portfolio_log_return + Sharpe/Sortino components - drawdown - turnover."""
    w = _normalize(weights)
    port = sum(wi * ri for wi, ri in zip(w, bucket_returns))
    log_ret = math.log(max(1e-6, 1.0 + port))
    downside = abs(min(0.0, port))
    sharpe_c = port / 0.02 * 0.01
    sortino_c = (port / (downside + 1e-6)) * 0.005
    return log_ret + sharpe_c + sortino_c - 0.5 * abs(drawdown) - 0.1 * turnover


def _normalize(weights: List[float]) -> List[float]:
    vals = [max(0.0, float(w)) for w in weights]
    s = sum(vals)
    if s <= 0:
        n = len(weights) or 1
        return [1.0 / n] * len(weights)
    return [v / s for v in vals]


# ── walk-forward metrics + deployment gate ──────────────────────────────────────────
def compute_metrics(r_multiples: List[float]) -> Dict[str, float]:
    """Trade-level metrics from a list of per-decision R multiples."""
    rs = [float(r) for r in r_multiples if r is not None]
    n = len(rs)
    if n == 0:
        return {"trades": 0, "active_trades": 0, "activity_ratio": 0.0, "win_rate": 0.0,
                "expectancy": 0.0, "avg_R": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0,
                "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "apr": 0.0}
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    mean = sum(rs) / n
    var = sum((r - mean) ** 2 for r in rs) / n
    std = math.sqrt(var)
    downside = math.sqrt(sum(min(0.0, r) ** 2 for r in rs) / n)
    # equity curve + max drawdown
    eq = 0.0; peak = 0.0; max_dd = 0.0
    for r in rs:
        eq += r
        peak = max(peak, eq)
        max_dd = max(max_dd, peak - eq)
    cagr_proxy = sum(rs)
    active = sum(1 for r in rs if abs(r) > 1e-9)   # decisions that actually took a position
    return {
        "trades": n,
        "active_trades": active,
        "activity_ratio": round(active / n, 4),
        "win_rate": round(len(wins) / n, 4),
        "expectancy": round(mean, 4),
        "avg_R": round(mean, 4),
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else float(gross_win > 0) * 99.0,
        "max_drawdown": round(max_dd, 4),
        "sharpe": round(mean / std, 4) if std > 0 else 0.0,
        "sortino": round(mean / downside, 4) if downside > 0 else 0.0,
        "calmar": round(cagr_proxy / max_dd, 4) if max_dd > 0 else 0.0,
        # Accumulated Percent/Position Returns (Roy 2024 §3.8 APR analog): the strategy's
        # accumulated per-decision outcome. Units are R here (per-trade outcomes are
        # R-normalized), so this is the accumulated-R return, reported under the paper's name.
        "apr": round(cagr_proxy, 4),
    }


def deployment_recommendation(baseline: Dict[str, float], rl: Dict[str, float],
                              samples: int) -> Tuple[str, Dict[str, Any]]:
    """
    Decide one of: do_not_deploy | shadow_mode_only | paper_mode_only | eligible_after_review.
    Precedence implements the Step-9 gate rules (safety-first).
    """
    b_exp = baseline.get("expectancy", 0.0)
    r_exp = rl.get("expectancy", 0.0)
    b_dd = baseline.get("max_drawdown", 0.0)
    r_dd = rl.get("max_drawdown", 0.0)
    b_wr = baseline.get("win_rate", 0.0)
    r_wr = rl.get("win_rate", 0.0)
    # Tran, Pham-Hi & Bui (Algorithms 2023): Sharpe ratio is the superior selection objective —
    # a policy that lifts raw expectancy while degrading Sharpe is trading more volatility for
    # the same edge (their return-reward agent: higher mean, much higher SD). So Sharpe is a
    # first-class gate criterion: a material Sharpe regression blocks deployment regardless of
    # expectancy. SHARPE_REGRESSION_TOL is the allowed relative drop before we object.
    b_sh = baseline.get("sharpe", 0.0)
    r_sh = rl.get("sharpe", 0.0)
    dd_worse = r_dd > b_dd * 1.15 + 1e-9
    exp_improved = r_exp > b_exp + 1e-9
    dd_improved = r_dd <= b_dd + 1e-9
    # "Materially worse" Sharpe: drops below baseline by more than the tolerance band. Only
    # meaningful when the baseline actually had positive risk-adjusted return.
    sharpe_worse = (b_sh > 1e-9) and (r_sh < b_sh * (1.0 - SHARPE_REGRESSION_TOL) - 1e-9)
    sharpe_improved = r_sh > b_sh + 1e-9
    notes = {"baseline_expectancy": b_exp, "rl_expectancy": r_exp,
             "baseline_max_drawdown": b_dd, "rl_max_drawdown": r_dd,
             "baseline_win_rate": b_wr, "rl_win_rate": r_wr,
             "baseline_sharpe": b_sh, "rl_sharpe": r_sh, "samples": samples}
    if samples < MIN_DEPLOY_SAMPLES:
        rec = "shadow_mode_only"; notes["reason"] = f"sample_too_small(<{MIN_DEPLOY_SAMPLES})"
    elif dd_worse:
        rec = "do_not_deploy"; notes["reason"] = "drawdown_materially_worse"
    elif sharpe_worse:
        # Risk-adjusted return regressed materially (Tran et al.): do not trade more variance
        # for the same/greater edge. Block even if raw expectancy improved.
        rec = "do_not_deploy"; notes["reason"] = "sharpe_materially_worse"
    elif (r_wr > b_wr + 1e-9) and (r_exp < b_exp - 1e-9):
        rec = "shadow_mode_only"; notes["reason"] = "win_rate_up_but_expectancy_down"
    elif not exp_improved:
        rec = "do_not_deploy"; notes["reason"] = "does_not_beat_baseline_out_of_sample"
    elif exp_improved and dd_improved and sharpe_improved:
        # Best case: edge up, drawdown down, AND risk-adjusted return up.
        rec = "paper_mode_only"; notes["reason"] = "expectancy_drawdown_and_sharpe_improved"
    elif exp_improved and dd_improved:
        rec = "paper_mode_only"; notes["reason"] = "expectancy_and_drawdown_improved"
    else:
        rec = "eligible_after_review"; notes["reason"] = "expectancy_improved_drawdown_flat"
    notes["overfit_warning"] = bool(samples < MIN_DEPLOY_SAMPLES * 2)

    # Trade-activity overfit guard (Yasin & Gill 2024, §11.5): a policy that looks great
    # on almost no actual trades (their DQN: 100% win rate, ONE trade) is overfit, not
    # deployable. Only downgrades positive recommendations; no-op when the metric is absent
    # (so hand-built test dicts are unaffected).
    rl_active = rl.get("active_trades")
    rl_ratio = rl.get("activity_ratio")
    if rl_active is not None and rec in ("paper_mode_only", "eligible_after_review"):
        if rl_active < MIN_ACTIVE_TRADES or (rl_ratio is not None and rl_ratio < MIN_ACTIVITY_RATIO):
            notes["pre_activity_recommendation"] = rec
            notes["reason"] = f"insufficient_trade_activity_overfit_risk(active={rl_active})"
            notes["overfit_warning"] = True
            rec = "shadow_mode_only"
    notes["rl_active_trades"] = rl_active
    return rec, notes
