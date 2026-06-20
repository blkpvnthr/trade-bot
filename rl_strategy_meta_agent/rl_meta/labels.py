"""Human-curated strategy-symbol labels (WINS / LOSSES / FLAT).

These come from the export FOLDER the user filed each CSV under -- a *prior*
("I believed this strategy works / fails / stalls on this ticker"). They are
metadata only: never used to compute win rate, PF, expectancy, drawdown, or RL
eligibility (those come from Net PnL inside the CSV). This module compares the
human prior against the actual PnL-derived outcome.
"""
from __future__ import annotations

BUCKETS = ("WINS", "LOSSES", "FLAT", "UNKNOWN")

# A per-symbol net PnL (normalized $) within +/- this band reads as "flat".
FLAT_THRESHOLD = 1000.0


def normalize_bucket(label) -> str:
    b = (label or "UNKNOWN").upper()
    return b if b in BUCKETS else "UNKNOWN"


def actual_bucket(net_pnl: float, flat_threshold: float = FLAT_THRESHOLD) -> str:
    """The outcome bucket implied by realized (normalized) net PnL."""
    if abs(net_pnl) <= flat_threshold:
        return "FLAT"
    return "WINS" if net_pnl > 0 else "LOSSES"


def label_mismatch(human_bucket, net_pnl: float, profit_factor: float,
                   flat_threshold: float = FLAT_THRESHOLD):
    """Return a reason string if the human label disagrees with the actual
    outcome, else None. UNKNOWN labels are never a mismatch."""
    hb = normalize_bucket(human_bucket)
    pf = profit_factor
    if hb == "WINS" and (pf < 1.0 or net_pnl < 0):
        return f"WINS but PF {pf:.2f}<1 / net {net_pnl:+,.0f}"
    if hb == "LOSSES" and (pf > 1.0 or net_pnl > 0):
        return f"LOSSES but PF {pf:.2f}>1 / net {net_pnl:+,.0f}"
    if hb == "FLAT" and abs(net_pnl) > flat_threshold:
        return f"FLAT but net {net_pnl:+,.0f} (|net|>{flat_threshold:,.0f})"
    return None


def label_correct(human_bucket, net_pnl: float, profit_factor: float,
                  flat_threshold: float = FLAT_THRESHOLD):
    """True/False if labeled and matches/mismatches; None if UNKNOWN."""
    if normalize_bucket(human_bucket) == "UNKNOWN":
        return None
    return label_mismatch(human_bucket, net_pnl, profit_factor, flat_threshold) is None


def bucket_accuracy(rows: list, flat_threshold: float = FLAT_THRESHOLD) -> dict:
    """rows: dicts with human_bucket, net_pnl, profit_factor. Returns per-bucket
    and overall accuracy over LABELED (non-UNKNOWN) rows."""
    from collections import Counter
    correct, total = Counter(), Counter()
    for r in rows:
        ok = label_correct(r["human_bucket"], r["net_pnl"], r["profit_factor"],
                           flat_threshold)
        if ok is None:
            continue
        hb = normalize_bucket(r["human_bucket"])
        total[hb] += 1
        total["ALL"] += 1
        if ok:
            correct[hb] += 1
            correct["ALL"] += 1
    return {b: (correct[b] / total[b] if total[b] else None, correct[b], total[b])
            for b in ("WINS", "LOSSES", "FLAT", "ALL") if total[b]}


def _demo():
    # A profitable symbol filed under LOSSES is a mismatch; metrics unaffected.
    assert label_mismatch("LOSSES", net_pnl=5000, profit_factor=1.4)
    assert label_mismatch("WINS", net_pnl=-300, profit_factor=0.8)
    assert label_mismatch("FLAT", net_pnl=9000, profit_factor=2.0)
    assert label_mismatch("WINS", net_pnl=4000, profit_factor=1.6) is None
    assert label_mismatch("UNKNOWN", net_pnl=-9999, profit_factor=0.1) is None
    assert actual_bucket(50) == "FLAT" and actual_bucket(5000) == "WINS"
    print("labels._demo OK")


if __name__ == "__main__":
    _demo()
