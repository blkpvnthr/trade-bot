"""alpha_factory CLI.

  python -m alpha_factory.cli discover            [--demo | --panel closes.pkl]
  python -m alpha_factory.cli evaluate --strategy rev_1d   [--demo | --panel ...]
  python -m alpha_factory.cli evaluate-all        [--demo | --panel ...]
  python -m alpha_factory.cli registry
  python -m alpha_factory.cli accepted
  python -m alpha_factory.cli rejected
  python -m alpha_factory.cli export-allocator-input

Data: --demo uses a deterministic synthetic panel (planted alpha + noise). --panel PATH loads a
pickled closes DataFrame [dates x symbols, tz-aware UTC index]; an optional sibling *_vol.pkl gives
volume (else a flat volume is used). No broker, no orders - research only.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from . import miner
from .registry import AlphaRegistry
from .scientist import Criteria, cost_curve, run_batch
from .schema import ACCEPTED


def _parse_sweep(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


# Cost presets: (registry-acceptance cost, sweep, promotion cost). --cost-bps overrides the accept
# cost; --cost-sweep-bps overrides the sweep. liquid where cost is controllable, small-cap where not.
PRESETS = {
    "liquid_etf": {"accept": 5.0, "sweep": "2,5,10,15", "promote": 10.0},
    "small_cap":  {"accept": 20.0, "sweep": "15,20,25", "promote": 20.0},
}


def _resolve_costs(args):
    """-> (accept_cost, sweep_list, promote_cost). Explicit flags beat preset; preset beats default."""
    preset = PRESETS.get(getattr(args, "cost_preset", "") or "")
    accept = args.cost_bps if args.cost_bps is not None else (preset["accept"] if preset else 5.0)
    sweep = args.cost_sweep_bps or (preset["sweep"] if preset else "5,15,20,25")
    promote = preset["promote"] if preset else 10.0
    return accept, _parse_sweep(sweep), promote


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load(args):
    universe = getattr(args, "universe", "") or None
    source = args.source or ("alpaca_cache" if universe else "")
    if source:
        from .data import load_panel
        closes, vol, report = load_panel(source=source, universe=universe)
        scope = f"universe={universe}" if universe else "default"
        print(f"loaded {len(report['present'])}/{len(report['requested'])} symbols ({scope}), "
              f"{len(closes)} rows [{report['start']} .. {report['end']}]")
        if report["missing"]:
            print(f"  missing {len(report['missing'])}: {', '.join(report['missing'])}")
        return closes, vol, None                              # no planted alpha on real data
    if args.demo:
        from .synth import make_panel
        return make_panel()                                   # closes, vol, factor
    if not args.panel:
        raise SystemExit("need --source alpaca_cache, --demo, or --panel PATH")
    closes = pd.read_pickle(args.panel)
    vp = args.panel.replace(".pkl", "_vol.pkl")
    vol = pd.read_pickle(vp) if os.path.exists(vp) else pd.DataFrame(
        1.0, index=closes.index, columns=closes.columns)
    return closes, vol, None


def _candidates(args):
    closes, vol, factor = _load(args)
    cands = miner.discover(closes, vol, created_at=_now(), factor=factor)
    return closes, cands


def cmd_discover(args):
    _, cands = _candidates(args)
    print(f"{len(cands)} candidates:")
    for c in cands:
        print(f"  {c.name:18s} [{c.source}]  {c.formula}")


def cmd_evaluate(args):
    closes, cands = _candidates(args)
    cands = [c for c in cands if c.name == args.strategy]
    if not cands:
        raise SystemExit(f"no candidate named {args.strategy!r}")
    _run(closes, cands, args)


def cmd_evaluate_all(args):
    closes, cands = _candidates(args)
    _run(closes, cands, args)


def _run(closes, cands, args):
    accept_cost, sweep, promote_cost = _resolve_costs(args)
    crit = Criteria(cost_bps=accept_cost, promote_cost_bps=promote_cost)
    reg = AlphaRegistry()
    recs = run_batch(cands, closes, crit, reg, now=_now())
    by_name = {c.name: c for c in cands}
    fwd1 = closes.shift(-1) / closes - 1

    print(f"acceptance cost = {accept_cost:g}bps   promotion cost = {promote_cost:g}bps")
    for r in sorted(recs, key=lambda r: -r.score):
        ic = r.out_of_sample_metrics["oos_ic_mean"]
        ns = r.cost_model["net_sharpe"]
        flag = {"accepted": "ACCEPT", "rejected": "reject", "needs_more_data": "  ?  "}[r.decision]
        tag = " [FRAGILE_COST_MIRAGE]" if "FRAGILE_COST_MIRAGE" in r.tags else ""
        print(f"  [{flag}] {r.strategy_name:18s} score={r.score:.2f} oos_ic={ic:+.3f} "
              f"net_sharpe={ns:+.2f}{tag}  {r.rejection_reason}")
    n_acc = sum(r.decision == ACCEPTED for r in recs)
    print(f"\n{n_acc}/{len(recs)} accepted at {accept_cost:g}bps. registry: {reg.jsonl}")

    if getattr(args, "source", "") or getattr(args, "universe", ""):
        print("\nWARNING: 5bps is optimistic for small-cap/illiquid universes; "
              "use --cost-bps 15/20 for promotion decisions.")

    # fail-safe: ALWAYS show the cost-sensitivity table for every candidate
    hdr = "  ".join(f"{c:g}bp" for c in sweep)
    print(f"\ncost-sensitivity (net Sharpe), all candidates:\n  {'alpha':18s}{hdr}")
    for r in sorted(recs, key=lambda r: -r.score):
        curve = cost_curve(by_name[r.strategy_name].signals, fwd1, sweep)
        row = "  ".join(f"{curve[c]:+5.2f}" for c in sweep)
        mark = "  <- accepted" if r.decision == ACCEPTED else ""
        print(f"  {r.strategy_name:18s}{row}{mark}")

    # which survive Sharpe > 1 at the promotion cost?
    survivors = [r.strategy_name for r in recs
                 if cost_curve(by_name[r.strategy_name].signals, fwd1, [promote_cost])[promote_cost] >= 1.0]
    print(f"\nnet Sharpe >= 1.0 at {promote_cost:g}bps: "
          f"{', '.join(survivors) if survivors else 'NONE'}")

    # promotion verdict for accepted alphas
    accepted = [r for r in recs if r.decision == ACCEPTED]
    if accepted:
        print("\npromotion verdict (allocator-export gate):")
        for r in accepted:
            p = r.promotion
            ok = "PROMOTABLE" if p.get("promotable") else "blocked"
            fails = [k for k, v in p.get("checks", {}).items() if not v]
            print(f"  {r.strategy_name:18s} [{ok}] net_sharpe@{p['cost_bps']:g}bps={p['net_sharpe']:+.2f} "
                  f"turnover={p['turnover']:.2f} yrs+={p['year_pos_frac']:.0%} wf+={p['wf_frac']:.0%}"
                  + (f"  fails: {', '.join(fails)}" if fails else ""))
    print("\nPromotion rule: allocator export requires net Sharpe >= 1 at realistic cost, "
          "sane turnover, walk-forward + 60%-of-years positive.")


def cmd_registry(args):
    for r in AlphaRegistry().all():
        print(f"  {r.decision:15s} {r.strategy_name:18s} score={r.score:.2f}  {r.alpha_id}")


def cmd_accepted(args):
    for r in AlphaRegistry().accepted():
        print(f"  {r.strategy_name:18s} score={r.score:.2f} ic={r.out_of_sample_metrics['oos_ic_mean']:+.3f}")


def cmd_rejected(args):
    for r in AlphaRegistry().rejected():
        print(f"  {r.strategy_name:18s} {r.rejection_reason}")


def cmd_export(args):
    """Only PROMOTABLE alphas reach the allocator. Accepted-but-not-promotable is held back."""
    reg = AlphaRegistry()
    accepted = reg.accepted()
    promotable = [r for r in accepted if r.promotion.get("promotable")]
    out = {"accepted_alphas": [_allocator_row(r) for r in promotable]}
    path = os.path.join(reg.root, "allocator_input.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    held = [r.strategy_name for r in accepted if not r.promotion.get("promotable")]
    print(f"wrote {len(promotable)} promotable alphas -> {path}")
    if held:
        print(f"held back (accepted but not promotable): {', '.join(held)}")


def _allocator_row(r):
    """Bridge to the RL allocator. expected_return/risk/confidence are proxies (noted as such)."""
    cm = r.cost_model
    return {
        "alpha_id": r.alpha_id,
        "strategy_name": r.strategy_name,
        "score": r.score,
        "symbols": r.symbols,
        "timeframe": r.timeframe,
        "expected_return": r.out_of_sample_metrics["oos_ic_mean"],   # proxy: mean daily IC
        "risk": abs(cm.get("max_drawdown", 0.0)),                     # proxy: |max drawdown|
        "correlation_cluster": r.source,                             # proxy: source family
        "confidence": r.score,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="alpha_factory")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("discover", "evaluate", "evaluate-all", "registry", "accepted", "rejected",
                 "export-allocator-input"):
        sp = sub.add_parser(name)
        if name in ("discover", "evaluate", "evaluate-all"):
            sp.add_argument("--demo", action="store_true")
            sp.add_argument("--panel", default="")
            sp.add_argument("--source", default="", help="alpaca_cache (adjusted bar cache)")
            sp.add_argument("--universe", default="", help="liquid_etf_mega (implies alpaca_cache)")
        if name in ("evaluate", "evaluate-all"):
            sp.add_argument("--cost-bps", type=float, default=None,
                            help="round-trip cost for registry acceptance (overrides preset)")
            sp.add_argument("--cost-sweep-bps", default="",
                            help="comma list for the cost-sensitivity report (overrides preset)")
            sp.add_argument("--cost-preset", default="", choices=["", "liquid_etf", "small_cap"],
                            help="liquid_etf (accept 5 / sweep 2,5,10,15) or small_cap (20 / 15,20,25)")
        if name == "evaluate":
            sp.add_argument("--strategy", required=True)
    return p


def main():
    a = build_parser().parse_args()
    {"discover": cmd_discover, "evaluate": cmd_evaluate, "evaluate-all": cmd_evaluate_all,
     "registry": cmd_registry, "accepted": cmd_accepted, "rejected": cmd_rejected,
     "export-allocator-input": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
