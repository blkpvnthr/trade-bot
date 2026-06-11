#!/usr/bin/env python3
"""
bootstrap_finbert.py — one-time fetch / verify of the local FinBERT model.

FinBERT (ProsusAI/finbert, ~440MB) scores Alpaca news sentiment locally. The runtime scorer
(data_sources/finbert_scorer.py) loads it with local_files_only=True and NEVER downloads —
so the model must be fetched once per host with this script, which is the only place network
access happens.

Usage:
    python bootstrap_finbert.py --check      # report whether the model is installed
    python bootstrap_finbert.py --download   # fetch + save to models/nlp/finbert/ (needs network + transformers)

The bot runs fine WITHOUT the model (FINBERT_ENABLED=0 or model absent -> neutral sentiment).
Run --download, then set FINBERT_ENABLED=1, only when you want real sentiment.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
import os

MODEL_ID = os.getenv("FINBERT_MODEL_ID", "ProsusAI/finbert")
DEST = Path(os.getenv("FINBERT_MODEL_DIR", str(PROJECT_ROOT / "models" / "nlp" / "finbert")))


def check() -> int:
    cfg = DEST / "config.json"
    if not cfg.exists():
        print(f"✗ NOT installed — no model at {DEST}")
        print("  Run: python bootstrap_finbert.py --download   (then set FINBERT_ENABLED=1)")
        return 1
    files = sorted(p.name for p in DEST.glob("*") if p.is_file())
    weights = any(n.endswith((".bin", ".safetensors")) for n in files)
    size_mb = sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file()) / 1e6
    print(f"✓ Installed at {DEST}  ({size_mb:.0f} MB)")
    print(f"  files: {files}")
    print(f"  weights present: {'yes' if weights else 'NO (incomplete download)'}")
    return 0 if weights else 2


def download() -> int:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as e:
        print(f"✗ transformers not installed ({e}). Install it first: pip install transformers")
        return 1
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} → {DEST} (~440MB, one-time)…")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        tok.save_pretrained(str(DEST))
        mdl.save_pretrained(str(DEST))
    except Exception as e:
        print(f"✗ download failed: {e}")
        return 1
    print(f"  id2label: {mdl.config.id2label}")
    return check()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the installed model")
    ap.add_argument("--download", action="store_true", help="fetch + cache the model (needs network)")
    a = ap.parse_args()
    if a.download:
        return download()
    return check()


if __name__ == "__main__":
    sys.exit(main())
