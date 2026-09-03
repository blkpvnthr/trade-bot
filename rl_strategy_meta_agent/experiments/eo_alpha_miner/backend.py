"""EO backend: detect/run the compiled C++ miner, and parse its sexp formulas
into the existing alphas.py AST so every candidate flows through the same
evaluation + validation stack. Research only.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BINARY = HERE / "eo_miner"

# C++ sexp op name -> alphas.py AST op. safe_div is special-cased to mul(a, inv(b))
# so the Python re-eval is NaN-safe (guarded inverse), matching the C++ safe_div.
_OP = {"add": "add", "sub": "sub", "mul": "mul",
       "min": "gmin", "max": "gmax", "abs": "abs", "sign": "sign", "neg": "neg"}
_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def available() -> bool:
    """True if the compiled EO binary exists (CI without EO -> tests skip)."""
    return BINARY.exists() and os.access(BINARY, os.X_OK)


def parse_sexp(s: str, feature_names) -> tuple:
    """'add(f0,mul(f1,0.5))' -> AST tuple. Leaves: feature name -> ('in',name),
    number -> ('const',val)."""
    feats = set(feature_names)
    pos = 0

    def parse():
        nonlocal pos
        m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*|[-+]?[\d.][\w.+-]*)", s[pos:])
        tok = m.group(1); pos += m.end()
        if pos < len(s) and s[pos] == "(":            # function
            pos += 1
            args = [parse()]
            while s[pos] == ",":
                pos += 1; args.append(parse())
            assert s[pos] == ")"; pos += 1
            if tok == "safe_div":                      # a/b -> a * inv(b) (NaN-safe)
                return ("mul", args[0], ("inv", args[1]))
            return (_OP[tok], *args)
        if tok in feats:
            return ("in", tok)
        if _NUM.match(tok):
            return ("const", float(tok))
        raise ValueError(f"unknown leaf {tok!r}")

    node = parse()
    return node


def run(csv_path: str, out_path: str, cfg: dict, feature_names) -> list[tuple]:
    """Invoke the C++ EO miner; return parsed AST nodes (skips unparseable lines)."""
    if not available():
        raise RuntimeError("EO binary not built (run build.sh)")
    args = [str(BINARY), csv_path, out_path,
            str(cfg.get("population", 300)), str(cfg.get("generations", 40)),
            str(cfg.get("max_depth", 5)), str(cfg.get("seed", 42)),
            str(cfg.get("crossover_rate", 0.7)), str(cfg.get("mutation_rate", 0.3)),
            str(cfg.get("top_k", 25))]
    subprocess.run(args, check=True, capture_output=True, text=True)
    nodes = []
    for line in Path(out_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            nodes.append(parse_sexp(line, feature_names))
        except Exception:
            continue
    return nodes


def _demo():
    feats = ["f0", "f1", "f2"]
    n = parse_sexp("add(safe_div(f1,-1.848),f0)", feats)
    assert n == ("add", ("mul", ("in", "f1"), ("inv", ("const", -1.848))), ("in", "f0")), n
    assert parse_sexp("neg(abs(f2))", feats) == ("neg", ("abs", ("in", "f2")))
    print("backend._demo OK  available()=", available())


if __name__ == "__main__":
    _demo()
