"""Print the sleeve framework status. Run: python -m rl_trading.sleeves.status [--json]"""
from __future__ import annotations

import json
import sys

from .initial import build_default_registry
from .registry import render_markdown, status_report


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    reg = build_default_registry()
    if "--json" in argv:
        print(json.dumps(status_report(reg), indent=2, default=str))
    else:
        print(render_markdown(reg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
