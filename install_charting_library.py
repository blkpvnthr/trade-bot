#!/usr/bin/env python3
"""
install_charting_library.py — place / verify the self-hosted TradingView Advanced Charts library.

Advanced Charts is a licensed, self-hosted-ONLY product. You obtain the files from TradingView
(https://www.tradingview.com/charting-library/ → private `tradingview/charting_library` GitHub repo);
they cannot be fetched here. This script just installs a folder you already have, and verifies the
layout the dashboard expects:

    static/charting_library/charting_library.standalone.js

Usage:
    python3 install_charting_library.py --check                 # verify what's installed
    python3 install_charting_library.py --from /path/to/charting_library   # copy it into static/
    python3 install_charting_library.py --from /path/to/repo    # auto-finds charting_library/ inside

The chart page (tv_chart.html) loads it from /dash/static/charting_library/ via the existing
Flask route in dashboard.py. The /dash/udf data backend is independent and already live.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "static" / "charting_library"
ENTRY = DEST / "charting_library.standalone.js"


def find_lib(src: Path) -> Path | None:
    """Return the charting_library dir given either it directly or a repo containing it."""
    if (src / "charting_library.standalone.js").exists():
        return src
    cand = src / "charting_library"
    if (cand / "charting_library.standalone.js").exists():
        return cand
    # search a couple levels deep (e.g. cloned repo layouts)
    for p in src.rglob("charting_library.standalone.js"):
        return p.parent
    return None


def check() -> int:
    if not ENTRY.exists():
        print(f"✗ NOT installed — missing {ENTRY.relative_to(ROOT)}")
        print("  The chart page will show the setup card until this exists.")
        return 1
    files = sum(1 for _ in DEST.rglob("*") if _.is_file())
    size_mb = sum(f.stat().st_size for f in DEST.rglob("*") if f.is_file()) / 1e6
    bundles = DEST / "bundles"
    print(f"✓ Installed: {ENTRY.relative_to(ROOT)}")
    print(f"  files: {files}   total: {size_mb:.1f} MB   bundles/: {'present' if bundles.is_dir() else 'MISSING (chart will not render)'}")
    print(f"  served at: /dash/static/charting_library/charting_library.standalone.js")
    return 0 if bundles.is_dir() else 2


def install(src: Path) -> int:
    if not src.exists():
        print(f"✗ source path does not exist: {src}")
        return 1
    lib = find_lib(src)
    if lib is None:
        print(f"✗ no charting_library.standalone.js found under {src}")
        print("  Point --from at the 'charting_library' folder from the TradingView repo.")
        return 1
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"  removing existing {DEST.relative_to(ROOT)}")
        shutil.rmtree(DEST)
    print(f"  copying {lib}  →  {DEST.relative_to(ROOT)}")
    shutil.copytree(lib, DEST)
    return check()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="verify the installed library layout")
    ap.add_argument("--from", dest="src", metavar="PATH", help="copy charting_library/ from PATH into static/")
    a = ap.parse_args()
    if a.src:
        return install(Path(a.src).expanduser().resolve())
    if a.check or True:  # default action is --check
        return check()


if __name__ == "__main__":
    sys.exit(main())
