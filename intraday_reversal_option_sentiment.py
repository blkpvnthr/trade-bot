#!/usr/bin/env python3
"""Option-sentiment buy-gate for the intraday reversal CALL overlay.

Before buying a 1-7 DTE call, optionally confirm with two AlphaVantage signals
(reused from av_catalyst.py):
  * HISTORICAL_PUT_CALL_RATIO  — full-chain put/call ratio. Low (<=0.6) ~ bullish,
    high (>=1.0) ~ bearish; CONTRARIAN at the extremes (very high can be bullish, very
    low near zero can be bearish).
  * HISTORICAL_VOLUME_OPEN_INTEREST_RATIO — aggregated into a directional CALL-SHARE
    (fraction of fresh vol/OI activity on the call side; >0.5 = bullish lean).

This module is a CONFIRMATION FILTER only — it never places orders, never sizes, and
never bypasses the engine gates or the v2 router. The decision (`option_sentiment_ok`)
is PURE and testable; the fetch (`fetch_option_sentiment`) is the only impure part and
is read-only AV data. Fail-OPEN: missing data does not block a trade.

Gated by INTRADAY_REVERSAL_OPTION_SENTIMENT (default off); thresholds live on EngineConfig.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("intraday_reversal_option_sentiment")


@dataclass
class OptionSentiment:
    put_call_ratio: Optional[float] = None      # full-chain PCR
    vol_oi_call_share: Optional[float] = None   # fresh vol/OI fraction on calls (0..1)


def option_sentiment_ok(s: OptionSentiment, cfg: Any) -> tuple[bool, str]:
    """Should we BUY a call given the option-positioning sentiment? Pure. Blocks only on
    clearly bearish reads; allows bullish / contrarian-bullish / neutral / unknown.

      PCR:
        <= pcr_contra_bear_max (≈0.20)            -> contrarian BEARISH -> block
        in [pcr_bear_min, pcr_contra_bull_min)    -> bearish           -> block   ([1.0, 1.5))
        <= pcr_bull_max (≈0.60)                    -> bullish           -> allow
        >= pcr_contra_bull_min (≈1.5)             -> contrarian bullish -> allow
        else (neutral 0.6..1.0) / None             -> allow (not bearish)
      vol/OI call-share (if min_vol_oi_call_share>0): below it -> block (fresh activity not
        call-skewed enough)."""
    pcr = s.put_call_ratio
    if pcr is not None:
        if pcr <= cfg.pcr_contra_bear_max:
            return False, f"pcr_extreme_low_contrarian_bearish:{pcr:.2f}"
        if cfg.pcr_bear_min <= pcr < cfg.pcr_contra_bull_min:
            return False, f"pcr_bearish:{pcr:.2f}"
    cs = s.vol_oi_call_share
    if cs is not None and cfg.min_vol_oi_call_share > 0 and cs < cfg.min_vol_oi_call_share:
        return False, f"vol_oi_call_share_low:{cs:.2f}"
    if pcr is None:
        label = "pcr_unknown"
    elif pcr <= cfg.pcr_bull_max:
        label = "pcr_bullish"
    elif pcr >= cfg.pcr_contra_bull_min:
        label = "pcr_contrarian_bullish"
    else:
        label = "pcr_neutral"
    detail = label + (f":{pcr:.2f}" if pcr is not None else "")
    if cs is not None:
        detail += f"|call_share={cs:.2f}"
    return True, detail


def fetch_option_sentiment(symbol: str, date: Optional[str] = None) -> OptionSentiment:
    """Read-only AV fetch (lazy import av_catalyst). Returns whatever is available;
    fields are None on failure so the gate fails open. ``date`` defaults to the previous
    trading session (PIT-safe: pass a date <= entry)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    d = date or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    pcr = cs = None
    try:
        import av_catalyst
        pcr = av_catalyst.put_call_ratio(symbol, d)
        cs = av_catalyst.volume_oi_call_share(symbol, d)
    except Exception as exc:
        logger.debug("option sentiment fetch failed for %s: %s", symbol, exc)
    return OptionSentiment(put_call_ratio=pcr, vol_oi_call_share=cs)
