#!/usr/bin/env python3
"""
adopt_position.py — audited operator adoption of a legacy/manual position.
==========================================================================

Companion to position_ownership.py. The ownership guard (run.py chokepoint)
only lets the bot SELL symbols recorded in the bot-owned ledger
(system-data/bot_owned_ledger.json, format {"owned": {SYM: qty}}). A position
the user opened by hand is, by design, NOT in the ledger, so the bot's exit
paths (sma200_exit, option-safety, etc.) are blocked from touching it.

This tool lets an operator *deliberately* hand one specific legacy position to
the bot so its existing exit logic may sell it — and, unlike the silent
`position_ownership.seed()`, it writes an append-only AUDIT record
(timestamp / symbol / qty / reason) to logs/ownership_adoptions.jsonl.

It is intentionally self-contained (no project imports) so it runs even when the
project virtualenv is unavailable, and it writes the EXACT ledger format the
guard reads.

IMPORTANT — takes effect on the next bot start. position_ownership._load()
caches the ledger in-process and never re-reads the file, so a *running* bot
will not see an adoption until it is restarted.

Usage:
    python adopt_position.py show
    python adopt_position.py adopt DBC 33 "reason text"
    python adopt_position.py release DBC "reason text"     # undo an adoption
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LEDGER_PATH = Path(
    os.getenv("BOT_OWNED_LEDGER_PATH", str(PROJECT_ROOT / "system-data" / "bot_owned_ledger.json"))
)
ADOPTION_LOG_PATH = Path(
    os.getenv("BOT_ADOPTION_LOG_PATH", str(PROJECT_ROOT / "logs" / "ownership_adoptions.jsonl"))
)
_EPS = 1e-4


def _norm(sym: str) -> str:
    return str(sym or "").upper().strip()


def _load() -> dict:
    if not LEDGER_PATH.exists():
        return {}
    raw = json.loads(LEDGER_PATH.read_text() or "{}")
    owned = raw.get("owned", raw) if isinstance(raw, dict) else {}
    out = {}
    for k, v in owned.items():
        try:
            out[_norm(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def _save(data: dict) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"owned": {k: round(v, 6) for k, v in data.items() if v > _EPS}},
                              indent=2, sort_keys=True))
    os.replace(tmp, LEDGER_PATH)


def _audit(action: str, symbol: str, qty: float, reason: str, ledger_after: dict) -> dict:
    entry = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "symbol": _norm(symbol),
        "qty": round(float(qty), 6),
        "reason": reason,
        "operator": os.getenv("USER", "operator"),
        "ledger_after": {k: round(v, 6) for k, v in ledger_after.items()},
    }
    ADOPTION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ADOPTION_LOG_PATH.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return entry


def adopt(symbol: str, qty: float, reason: str) -> dict:
    sym = _norm(symbol)
    q = float(qty)
    if q <= 0:
        raise SystemExit("qty must be > 0 to adopt")
    if not reason.strip():
        raise SystemExit("a non-empty reason is required for an audited adoption")
    data = _load()
    data[sym] = data.get(sym, 0.0) + q
    _save(data)
    return _audit("adopt", sym, q, reason, data)


def release(symbol: str, reason: str) -> dict:
    sym = _norm(symbol)
    data = _load()
    removed = data.pop(sym, 0.0)
    _save(data)
    return _audit("release", sym, removed, reason, data)


def _show() -> None:
    data = _load()
    print(f"bot-owned ledger ({LEDGER_PATH}):")
    if not data:
        print("  (empty — bot owns nothing; it will not sell any existing position)")
    for s, q in sorted(data.items()):
        print(f"  {s:24s} {q:g}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("show", "list"):
        _show()
    elif args[0] == "adopt":
        if len(args) < 4:
            raise SystemExit('usage: adopt_position.py adopt SYMBOL QTY "reason"')
        e = adopt(args[1], float(args[2]), " ".join(args[3:]))
        print("ADOPTED:", json.dumps(e, indent=2))
        print("\nNOTE: restart the bot for a running process to honour this (ledger is cached in-process).")
    elif args[0] == "release":
        if len(args) < 3:
            raise SystemExit('usage: adopt_position.py release SYMBOL "reason"')
        e = release(args[1], " ".join(args[2:]))
        print("RELEASED:", json.dumps(e, indent=2))
    else:
        raise SystemExit(__doc__)
