#!/usr/bin/env python3
"""Idempotent minimal patch for dashboard.py: add _render_tjl_backtest() and show it on
the /backtest page (above the existing crossover tearsheet iframe). Reverts cleanly via
the dashboard.py backup. Run from the repo dir: python3 tjl_dash_patch.py"""
import sys

P = "dashboard.py"
src = open(P, encoding="utf-8").read()
if "_render_tjl_backtest" in src:
    print("already patched — no change")
    sys.exit(0)

RENDERER = r'''
def _render_tjl_backtest():
    """Render backtest_outputs/tjl_backtest.json (Trend Join Long backtest). Empty-safe."""
    import json as _json
    from html import escape as _esc
    f = ROOT / "backtest_outputs" / "tjl_backtest.json"
    if not f.exists():
        return '<p style="color:#8786b5">No TJL backtest yet. Run <code>python tjl_backtest_report.py</code>.</p>'
    try:
        d = _json.loads(f.read_text())
    except Exception as e:
        return '<p style="color:#ff6b6b">Could not read TJL backtest: ' + _esc(str(e)) + '</p>'
    agg = d.get("aggregate", {}); cfg = d.get("config", {})
    pf = lambda x: "∞" if x is None else x
    cards = "".join('<div class=card><div class=k>%s</div><div class=v>%s</div></div>' % (k, v) for k, v in [
        ("symbols", agg.get("symbols", "-")), ("trades", agg.get("trades", "-")),
        ("win rate", str(agg.get("win_rate", "-")) + "%"),
        ("total return", str(agg.get("total_ret_pct", "-")) + "%"),
        ("profit factor", pf(agg.get("profit_factor"))),
        ("no lookahead", "yes" if agg.get("no_lookahead") else "NO")])
    rows = "".join("<tr><td style='text-align:left'>%s</td><td>%s</td><td>%s</td><td>%s%%</td><td>%s%%</td><td>%s</td><td>%s</td></tr>" % (
        _esc(str(p.get("symbol"))), p.get("source"), p.get("trades"), p.get("win_rate"),
        p.get("total_ret_pct"), pf(p.get("profit_factor")), "yes" if p.get("lookahead_ok") else "NO")
        for p in d.get("per_symbol", []))
    table = "<table><thead><tr><th style='text-align:left'>symbol</th><th>src</th><th>trades</th><th>win%</th><th>total%</th><th>PF</th><th>no-LA</th></tr></thead><tbody>" + rows + "</tbody></table>"
    cav = "".join("<li>" + _esc(str(c)) + "</li>" for c in d.get("caveats", []))
    cov = "<p style='color:#8786b5;margin:6px 0 2px'>" + str(len(d.get("universe", []))) + \
        " shown / " + str(len(d.get("universe_requested", d.get("universe", [])))) + " names &middot; " + \
        str(agg.get("symbols_traded", "-")) + " traded &middot; no setups: " + \
        (_esc(", ".join(d.get("no_setups", []))) or "none") + \
        ((" &middot; dropped PF&lt;" + str(d.get("min_pf")) + ": " + _esc(", ".join(d.get("dropped_pf", []))))
         if d.get("dropped_pf") else "") + "</p>"
    header = "<h1>Strategy &middot; " + _esc(str(d.get("strategy", "TJL"))) + " backtest</h1>" + \
        "<p style='color:#8786b5;margin-top:-6px'>Long-only momentum-into-trend. trend=" + _esc(str(cfg.get("trend_tf"))) + \
        "/SMA" + str(cfg.get("trend_sma_len")) + ", momo=" + _esc(str(cfg.get("momo_tf"))) + " stoch " + _esc(str(cfg.get("stoch"))) + \
        ", EMA " + _esc(str(cfg.get("ema"))) + " on " + _esc(str(cfg.get("entry_tf"))) + ", stop " + str(cfg.get("stop_pct")) + \
        "% / TP " + str(cfg.get("tp_pct")) + "% &middot; mode " + _esc(str(cfg.get("exit_mode", "-"))) + \
        ((" (trail " + str(cfg.get("trail_gap_pct")) + "%)") if cfg.get("use_trail") else "") + ". " + \
        str(d.get("days")) + "d &middot; data=" + _esc(str(d.get("data_source"))) + \
        " &middot; " + _esc(str(d.get("generated_at", ""))) + "</p>"
    return header + "<div style='display:flex;flex-wrap:wrap;gap:10px;margin:10px 0'>" + cards + "</div>" + cov + table + \
        "<h2 style='margin-top:22px'>Caveats</h2><ul style='line-height:1.55;color:#c7c7e6'>" + cav + "</ul>" + \
        "<p style='margin-top:14px;color:#8786b5'><a href='" + PREFIX + "/backtest/raw'>legacy crossover tearsheet &#8599;</a></p>"


'''

anchor = '@app.route(PREFIX + "/backtest")'
old_ret = 'return page("Strategy", iframe, "backtest")'
new_ret = 'return page("Strategy", _render_tjl_backtest() + iframe, "backtest")'

if anchor not in src or old_ret not in src:
    print("ABORT: expected anchors not found (dashboard.py shape differs) — no change made")
    sys.exit(2)

src = src.replace(anchor, RENDERER + anchor, 1)   # define renderer just before the route
src = src.replace(old_ret, new_ret, 1)            # show TJL above the existing tearsheet
open(P, "w", encoding="utf-8").write(src)
print("patched OK")
