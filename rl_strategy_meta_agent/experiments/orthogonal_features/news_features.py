"""Leak-free news / catalyst / FinBERT features aligned to the (symbol, bar) panel.

Reuses the existing plumbing -- data_sources.alpaca_news_client (Alpaca news +
FinBERT scoring + JSONL cache) and av_catalyst.earnings (PIT report dates) -- and
adds the windowed alignment the gate needs. The alignment is STRICTLY leak-free:
a feature for bar t aggregates only news with publish_time < t (the forward label
fut_ret_1 starts at t+1), inside a rolling lookback window, with recency decay.

Research only: no broker/order/live path. Reads news via the historical news API.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

TRADER_ROOT = Path(__file__).resolve().parents[3]   # /Users/blkpvnthr/Desktop/trader
RL_ROOT = Path(__file__).resolve().parents[2]        # rl_strategy_meta_agent
for p in (str(TRADER_ROOT), str(RL_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_env():
    """Populate APCA_*/ALPHA_VANTAGE_* from the project .env; enable FinBERT."""
    env = TRADER_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    os.environ["FINBERT_ENABLED"] = "1"


DEFAULTS = {"window_h": 24.0, "tau_min": 720.0, "rel_min": 0.0, "earnings_horizons": (7, 14)}


# ---------------------------------------------------------------- fetch / cache
# The Alpaca news client caps at one 50-item page (next_page_token isn't followed),
# so a single backfill_symbol call only returns the latest ~50 articles. We instead
# paginate by TIME: fetch [cursor, end] (50, asc), advance the cursor past the last
# article seen, repeat -- recovering full history across the whole panel window.
def _fetch_full(client, symbol, start: datetime, end: datetime, page=50):
    rows, seen = [], set()
    cursor = start
    while cursor < end:
        chunk = client.window_news(symbol, cursor, end, limit=page)
        chunk = [r for r in chunk if r.get("published_at")]
        if not chunk:
            break
        fresh = [r for r in chunk if (r["published_at"], r.get("title")) not in seen]
        for r in fresh:
            seen.add((r["published_at"], r.get("title")))
        rows.extend(fresh)
        last = max(r["published_at"] for r in chunk)
        nxt = pd.Timestamp(last) + pd.Timedelta(seconds=1)
        nxt = nxt.tz_convert("UTC") if nxt.tz else nxt.tz_localize("UTC")
        if len(chunk) < page or nxt.to_pydatetime() <= cursor:
            break                                         # exhausted this region
        cursor = nxt.to_pydatetime()
    return rows


def backfill(symbols, start: datetime, end: datetime, force: bool = False):
    load_env()
    from data_sources.alpaca_news_client import get_default_alpaca_news_client
    c = get_default_alpaca_news_client()
    counts = {}
    for s in symbols:
        if not force and c.load_symbol_history(s):
            counts[s] = len(c.load_symbol_history(s)); continue
        rows = _fetch_full(c, s, start, end)
        rows.sort(key=lambda r: r.get("published_at") or "")
        if rows:
            c._persist_symbol(s, rows)
        counts[s] = len(rows)
    return counts


def load_news(symbols) -> dict:
    """sym -> DataFrame[time(UTC-naive), score, pos, neg, neu, relevance], sorted by time."""
    load_env()
    from data_sources.alpaca_news_client import get_default_alpaca_news_client
    c = get_default_alpaca_news_client()
    out = {}
    for s in symbols:
        rows = c.load_symbol_history(s)
        if not rows:
            out[s] = pd.DataFrame(columns=["time", "score", "pos", "neg", "neu", "relevance"]); continue
        df = pd.DataFrame(rows)
        t = pd.to_datetime(df["published_at"], utc=True, errors="coerce").dt.tz_localize(None)
        score = pd.to_numeric(df["overall_sentiment_score"], errors="coerce").fillna(0.0)
        lab = df["overall_sentiment_label"].fillna("Neutral").str.lower()
        out[s] = pd.DataFrame({
            "time": t, "score": score,
            "pos": (lab == "positive").astype(float),     # per-article class (probs not persisted)
            "neg": (lab == "negative").astype(float),
            "neu": (~lab.isin(["positive", "negative"])).astype(float),
            "relevance": pd.to_numeric(df["relevance_score"], errors="coerce").fillna(0.0),
        }).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def load_earnings(symbols) -> dict:
    """sym -> sorted np.array of earnings report dates (UTC-naive). PIT: dates are
    announced weeks ahead, so 'within Nd' flags are knowable at bar time."""
    load_env()
    try:
        from av_catalyst import earnings
    except Exception:
        return {s: np.array([], dtype="datetime64[ns]") for s in symbols}
    out = {}
    for s in symbols:
        try:
            ds = [pd.to_datetime(e.get("reportedDate"), errors="coerce") for e in earnings(s)]
            ds = sorted(d.to_datetime64() for d in ds if pd.notna(d))
        except Exception:
            ds = []
        out[s] = np.array(ds, dtype="datetime64[ns]")
    return out


# ---------------------------------------------------------------- alignment
def align(panel_index: pd.MultiIndex, news: dict, earnings: dict, cfg=None) -> pd.DataFrame:
    """Build the candidate feature columns, leak-free (publish_time < bar t)."""
    c = {**DEFAULTS, **(cfg or {})}
    win = np.timedelta64(int(c["window_h"] * 3600), "s")
    feats = ["news_sentiment_mean", "news_sentiment_max", "news_sentiment_min",
             "news_volume_count", "abnormal_news_volume", "event_recency_decay",
             "finbert_pos", "finbert_neu", "finbert_neg", "catalyst_score"]
    feats += [f"earnings_within_{h}d" for h in c["earnings_horizons"]]
    out = pd.DataFrame(0.0, index=panel_index, columns=feats)

    pdf = pd.Series(panel_index.get_level_values(1), index=range(len(panel_index)))
    syms = panel_index.get_level_values(0)
    for s in pd.unique(syms):
        rows = np.where(syms == s)[0]
        bars = pd.to_datetime(panel_index.get_level_values(1)[rows]).to_numpy("datetime64[ns]")
        nd = news.get(s)
        ed = earnings.get(s, np.array([], dtype="datetime64[ns]"))
        if nd is None or nd.empty:
            block = _earnings_only(bars, ed, c)
        else:
            nt = nd["time"].to_numpy("datetime64[ns]")
            block = _symbol_block(bars, nt, nd["score"].to_numpy(), nd["relevance"].to_numpy(),
                                  nd["pos"].to_numpy(), nd["neg"].to_numpy(), nd["neu"].to_numpy(),
                                  ed, win, c)
        out.iloc[rows] = block
    return out


NBASE = 10   # sentiment_mean/max/min, volume, abnormal, recency, pos, neu, neg, catalyst


def _symbol_block(bars, nt, score, rel, pos, neg, neu, ed, win, c):
    n = len(bars)
    M = np.zeros((n, NBASE + len(c["earnings_horizons"])))
    elapsed_days = np.maximum((bars - bars[0]) / np.timedelta64(1, "D"), 1.0)
    tau = c["tau_min"]
    for i, t in enumerate(bars):
        hi = np.searchsorted(nt, t, side="left")          # strict < t  (LEAK-FREE)
        lo = np.searchsorted(nt, t - win, side="left")
        cnt = hi - lo
        all_before = hi                                   # total news strictly before t
        if cnt > 0:
            w = rel[lo:hi] + 0.05
            sc = score[lo:hi]
            M[i, 0] = float(np.average(sc, weights=w))    # sentiment_mean (relevance-weighted)
            M[i, 1] = float(sc.max())                     # sentiment_max
            M[i, 2] = float(sc.min())                     # sentiment_min
            M[i, 3] = float(cnt)                           # news_volume_count
            M[i, 6] = float(pos[lo:hi].mean())            # finbert_pos share
            M[i, 7] = float(neu[lo:hi].mean())            # finbert_neu share
            M[i, 8] = float(neg[lo:hi].mean())            # finbert_neg share
        if all_before > 0:
            dt_min = (t - nt[hi - 1]) / np.timedelta64(1, "m")
            M[i, 5] = float(np.exp(-dt_min / tau))         # event_recency_decay
        rate = all_before / elapsed_days[i]               # PIT expanding daily rate
        expected = rate * (c["window_h"] / 24.0)
        M[i, 4] = float(cnt / (expected + 0.5))            # abnormal_news_volume
        ew = _earnings_within(t, ed, c["earnings_horizons"])
        for j, v in enumerate(ew):
            M[i, NBASE + j] = v
        M[i, 9] = float(np.log1p(cnt) * abs(M[i, 0]) + 0.5 * (ew[0] if ew else 0.0))  # catalyst_score
    return M


def _earnings_only(bars, ed, c):
    n = len(bars)
    M = np.zeros((n, NBASE + len(c["earnings_horizons"])))
    for i, t in enumerate(bars):
        ew = _earnings_within(t, ed, c["earnings_horizons"])
        for j, v in enumerate(ew):
            M[i, NBASE + j] = v
        M[i, 9] = 0.5 * (ew[0] if ew else 0.0)
    return M


def _earnings_within(t, ed, horizons):
    if ed.size == 0:
        return [0.0] * len(horizons)
    future = ed[ed >= t]
    if future.size == 0:
        return [0.0] * len(horizons)
    days = (future[0] - t) / np.timedelta64(1, "D")
    return [1.0 if days <= h else 0.0 for h in horizons]


# ---------------------------------------------------------------- self-check
def _demo():
    # one symbol, one news item at T; assert leak-free: bars <= T see nothing, bars > T do
    idx = pd.MultiIndex.from_product(
        [["X"], pd.date_range("2024-01-02 10:00", periods=10, freq="h")], names=["symbol", "timestamp"])
    T = pd.Timestamp("2024-01-02 14:00")
    news = {"X": pd.DataFrame({"time": [T], "score": [0.8], "relevance": [1.0],
                               "pos": [1.0], "neg": [0.0], "neu": [0.0]})}
    out = align(idx, news, {"X": np.array([], dtype="datetime64[ns]")}, {"window_h": 6.0})
    cnt = out["news_volume_count"].to_numpy()
    bars = idx.get_level_values(1)
    before = cnt[bars <= T]
    after = cnt[(bars > T) & (bars <= T + pd.Timedelta(hours=6))]
    assert before.max() == 0.0, f"LEAK: news at {T} visible at/before T: {before}"
    assert after.max() >= 1.0, f"news not visible after T within window: {after}"
    assert out["news_sentiment_mean"][bars > T].max() > 0
    print("news_features._demo OK  leak-free alignment verified", dict(before=before.max(), after=after.max()))


if __name__ == "__main__":
    _demo()
