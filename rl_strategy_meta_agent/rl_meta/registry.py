"""PineScript discovery, parsing and the strategy registry.

We do NOT compile PineScript. We extract practical metadata via regex so
strategies can be cataloged, hashed and versioned. The backtest layer
(signals.py) supplies the executable behavior until real TradingView
strategy exports are wired in (see README "next steps").
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

from .types import StrategyMeta, Status

# Project layout: <root>/pinescripts and <root>/data live next to rl_meta/.
ROOT = Path(__file__).resolve().parent.parent
PINE_DIR = ROOT / "pinescripts"
DATA_DIR = ROOT / "data"
REGISTRY_PATH = DATA_DIR / "strategy_registry.json"

# Indicators we recognise by their Pine builtin call name.
_INDICATOR_PATTERNS = {
    "sma": r"\bta\.sma\b", "ema": r"\bta\.ema\b", "rsi": r"\bta\.rsi\b",
    "macd": r"\bta\.macd\b", "atr": r"\bta\.atr\b", "vwap": r"\bta\.vwap\b|\bvwap\b",
    "bbands": r"\bta\.bb\b", "stoch": r"\bta\.stoch\b", "adx": r"\bta\.adx\b|\bta\.dmi\b",
    "supertrend": r"\bta\.supertrend\b", "crossover": r"\bta\.crossover\b",
    "crossunder": r"\bta\.crossunder\b",
}

_TF_HINT = re.compile(r"""timeframe\s*=\s*["']([^"']+)["']""")
_INPUT = re.compile(
    r"""(\w+)\s*=\s*input(?:\.\w+)?\s*\(\s*([^,\)]+)"""
)


def file_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _lines_matching(text: str, *needles: str) -> list:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        low = line.lower()
        if any(n in low for n in needles):
            out.append(line)
    return out


def parse_pine(text: str) -> dict:
    """Extract practical metadata from PineScript source text."""
    name = None
    m = re.search(r"""(?:strategy|indicator)\s*\(\s*["']([^"']+)["']""", text)
    if m:
        name = m.group(1)

    inputs = {}
    for var, default in _INPUT.findall(text):
        inputs[var] = default.strip()

    indicators = sorted(
        key for key, pat in _INDICATOR_PATTERNS.items()
        if re.search(pat, text)
    )

    # Entry / exit / stop / target lines. PineScript uses strategy.entry/exit/close.
    entry = _lines_matching(text, "strategy.entry", "longcondition", "long = ", "buy")
    exit_ = _lines_matching(text, "strategy.close", "shortcondition", "exitcondition", "sell")
    stop = _lines_matching(text, "stop", "stoploss", "sl ", "sl=", "stop=")
    take = _lines_matching(text, "limit", "takeprofit", "tp ", "tp=", "target", "profit")
    tf = _TF_HINT.findall(text)

    return {
        "strategy_name": name,
        "inputs": inputs,
        "indicators": indicators,
        "entry_logic": entry[:20],
        "exit_logic": exit_[:20],
        "stop_loss": stop[:20],
        "take_profit": take[:20],
        "timeframe_hints": sorted(set(tf)),
    }


def discover(pine_dir: Path = PINE_DIR) -> list:
    """Return sorted list of *.txt PineScript files."""
    if not pine_dir.exists():
        return []
    return sorted(pine_dir.glob("*.txt"))


def load_strategies(pine_dir: Path = PINE_DIR, today: str | None = None) -> list:
    """Discover and parse every PineScript into a StrategyMeta."""
    today = today or date.today().isoformat()
    metas = []
    for path in discover(pine_dir):
        text = path.read_text(encoding="utf-8", errors="replace")
        h = file_hash(text)
        parsed = parse_pine(text)
        name = parsed.pop("strategy_name") or path.stem
        # Stable ID from filename stem; version tracked by hash.
        sid = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
        metas.append(StrategyMeta(
            strategy_id=sid,
            strategy_name=name,
            version_hash=h,
            date_added=today,
            source_file=path.name,
            status=Status.PENDING.value,
            **parsed,
        ))
    return metas


def write_registry(metas: list, path: Path = REGISTRY_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "count": len(metas),
        "strategies": [m.to_dict() for m in metas],
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_registry(path: Path = REGISTRY_PATH) -> list:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return [StrategyMeta(**s) for s in payload.get("strategies", [])]


def _demo():
    sample = (
        'strategy("Demo X", overlay=true, timeframe="60")\n'
        'fast = input.int(10, "Fast")\n'
        'slow = input.int(30, "Slow")\n'
        'long = ta.crossover(ta.sma(close, fast), ta.sma(close, slow))\n'
        'if long\n    strategy.entry("L", strategy.long, stop=close*0.97, limit=close*1.06)\n'
        'strategy.close("L", when=ta.crossunder(ta.sma(close, fast), ta.sma(close, slow)))\n'
    )
    p = parse_pine(sample)
    assert p["strategy_name"] == "Demo X"
    assert p["inputs"]["fast"] == "10" and p["inputs"]["slow"] == "30"
    assert "sma" in p["indicators"] and "crossover" in p["indicators"]
    assert p["timeframe_hints"] == ["60"]
    assert p["entry_logic"], "entry lines should be detected"
    assert file_hash(sample) == file_hash(sample)
    print("registry._demo OK")


if __name__ == "__main__":
    _demo()
