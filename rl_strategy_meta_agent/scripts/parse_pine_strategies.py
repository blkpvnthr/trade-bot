#!/usr/bin/env python3
"""Parse PineScripts, detect executable adapters, and write the registry.

Discover pinescripts/*.txt -> parse metadata -> detect a Python adapter for
each recognised pattern -> write data/strategy_registry.json with execution
status. Unrecognised scripts are marked parsed_but_not_executable and are
excluded from RL allocation.

This step does NOT need market data. Run scripts/fetch_alpaca_bars.py then
scripts/backtest_strategies.py to backtest the executable strategies.

    python scripts/parse_pine_strategies.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_meta.registry import load_strategies, write_registry, REGISTRY_PATH
from rl_meta.adapters import annotate_executability, SUPPORTED


def main():
    metas = load_strategies()
    if not metas:
        print("No PineScripts found in pinescripts/. Add *.txt files and re-run.")
        return

    for m in metas:
        annotate_executability(m)
    write_registry(metas)

    print(f"Parsed {len(metas)} PineScript(s). Supported adapters: "
          f"{', '.join(SUPPORTED)}\n")
    print(f"{'strategy':<26} {'adapter':<22} {'executable':<10} reason")
    print("-" * 86)
    for m in metas:
        reason = "" if m.executable else (m.unsupported_reason or "")
        print(f"{m.strategy_name[:26]:<26} {(m.adapter_type or '-'):<22} "
              f"{str(m.executable):<10} {reason}")

    n_exec = sum(1 for m in metas if m.executable)
    print(f"\n{n_exec}/{len(metas)} executable. registry -> {REGISTRY_PATH}")
    print("Next: scripts/fetch_alpaca_bars.py  then  scripts/backtest_strategies.py")


if __name__ == "__main__":
    main()
