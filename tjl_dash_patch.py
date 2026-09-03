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
    # inline SVG equity curve (no chart lib): cumulative sum of per-trade returns over time
    ec = d.get("equity_curve", []); eqm = d.get("equity", {})
    eqchart = ""
    if len(ec) >= 2:
        vs = [p["v"] for p in ec] + [0.0]
        col = "#3fb950" if ec[-1]["v"] >= 0 else "#f85149"
        svg = _tjl_chart([(col, ec)], min(vs), max(vs), eqm.get("start_date", ""),
                         eqm.get("end_date", ""), "pct", 0.0)
        eqchart = ("<div style='border:1px solid #21262d;border-radius:10px;background:#0d1117;padding:10px 12px;margin:8px 0 16px'>"
                   "<div style='color:#c7c7e6;margin-bottom:4px'>Equity curve (%) &middot; final " +
                   str(eqm.get("final_pct")) + "% &middot; max DD &minus;" + str(eqm.get("max_drawdown_pct")) +
                   "% &middot; " + str(eqm.get("points", 0)) + " pts</div>" + svg +
                   "<div style='color:#8786b5;font-size:12px;margin-top:2px'>cumulative sum of per-trade "
                   "returns (equal-weight, additive &mdash; not compounded)</div></div>")
    header = "<h1>Strategy &middot; " + _esc(str(d.get("strategy", "TJL"))) + " backtest</h1>" + \
        "<p style='color:#8786b5;margin-top:-6px'>Long-only momentum-into-trend. trend=" + _esc(str(cfg.get("trend_tf"))) + \
        "/SMA" + str(cfg.get("trend_sma_len")) + ", momo=" + _esc(str(cfg.get("momo_tf"))) + " stoch " + _esc(str(cfg.get("stoch"))) + \
        ", EMA " + _esc(str(cfg.get("ema"))) + " on " + _esc(str(cfg.get("entry_tf"))) + ", stop " + str(cfg.get("stop_pct")) + \
        "% / TP " + str(cfg.get("tp_pct")) + "% &middot; mode " + _esc(str(cfg.get("exit_mode", "-"))) + \
        ((" (trail " + str(cfg.get("trail_gap_pct")) + "%)") if cfg.get("use_trail") else "") + ". " + \
        str(d.get("days")) + "d &middot; data=" + _esc(str(d.get("data_source"))) + \
        " &middot; " + _esc(str(d.get("generated_at", ""))) + "</p>"
    return header + "<div style='display:flex;flex-wrap:wrap;gap:10px;margin:10px 0'>" + cards + "</div>" + cov + eqchart + table + \
        "<h2 style='margin-top:22px'>Caveats</h2><ul style='line-height:1.55;color:#c7c7e6'>" + cav + "</ul>" + \
        "<p style='margin-top:14px;color:#8786b5'><a href='" + PREFIX + "/backtest/raw'>legacy crossover tearsheet &#8599;</a></p>"


'''

PORTFOLIO_RENDERER = r'''
def _render_tjl_portfolio():
    """Portfolio dollar-equity backtest (tjl_portfolio_swing/intraday.json): multi-series
    equity curve + round-trip-slippage table. Empty-safe (returns '' if no artifact)."""
    import json as _json
    from html import escape as _esc
    _bd = ROOT / "backtest_outputs"
    def _load(n):
        f = _bd / n
        try:
            return _json.loads(f.read_text()) if f.exists() else None
        except Exception:
            return None
    sw = _load("tjl_portfolio_swing.json"); itr = _load("tjl_portfolio_intraday.json")
    base = sw or itr
    if not base:
        return ""
    e0 = base.get("start_equity", 100000)
    series = []
    if sw and sw.get("equity_curve"):   series.append(("swing+trail(PDH)", "#3fb950", sw["equity_curve"]))
    if itr and itr.get("equity_curve"): series.append(("intraday", "#58a6ff", itr["equity_curve"]))
    if base.get("spy_curve"):           series.append(("SPY B&H", "#8b949e", base["spy_curve"]))
    allv = [p["v"] for _l, _c, cv in series for p in cv] + [e0]
    _w = base.get("window", {})
    svg = _tjl_chart([(col, cv) for _l, col, cv in series], min(allv), max(allv),
                     _w.get("start", ""), _w.get("end", ""), "dollar", e0)
    legend = " &nbsp;&nbsp; ".join("<span style='color:" + col + "'>&#9632; " + _esc(lbl) + "</span>"
                                   for lbl, col, _cv in series)
    def _row(dct, name):
        if not dct or not dct.get("table"):
            return ""
        t = dct["table"]
        cells = "".join("<td>" + str(r.get("ret")) + "%</td>" for r in t)
        return ("<tr><td style='text-align:left'>" + _esc(name) + "</td>" + cells +
                "<td>" + str(t[0].get("sharpe", "-")) + "</td><td>" + str(t[0].get("maxdd", "-")) + "%</td></tr>")
    spy = base.get("spy")
    spyrow = (("<tr><td style='text-align:left'>SPY B&amp;H</td><td>" + str(spy["ret"]) +
               "%</td><td>&mdash;</td><td>&mdash;</td><td>" + str(spy["sharpe"]) + "</td><td>" +
               str(spy["maxdd"]) + "%</td></tr>") if spy else "")
    table = ("<table><thead><tr><th style='text-align:left'>config</th><th>0bp</th><th>5bp</th><th>10bp</th>"
             "<th>Sharpe</th><th>maxDD</th></tr></thead><tbody>" + _row(sw, "swing+trail(PDH)") +
             _row(itr, "intraday") + spyrow + "</tbody></table>")
    win = base.get("window", {})
    hdr = ("<h1>Portfolio backtest &middot; what would my account have done?</h1>"
           "<p style='color:#8786b5;margin-top:-6px'>$" + format(int(e0), ",") + " start &middot; " +
           str(base.get("slots")) + " equal-weight slots &middot; compounding, marked-to-market daily &middot; " +
           _esc(str(win.get("start", ""))) + " &rarr; " + _esc(str(win.get("end", ""))) + " &middot; " +
           str(base.get("names")) + " names</p>")
    cav = ("<p style='color:#8786b5;font-size:13px;margin-top:6px'>Total return at 0 / 5 / 10 bp round-trip "
           "slippage. Equity marked-to-market daily (drawdown includes overnight swings). "
           "<b>Universe is survivorship-curated</b> (current watchlist) &middot; IEX 5m fills &middot; not a live-P&amp;L claim.</p>")
    return (hdr + "<div style='border:1px solid #21262d;border-radius:10px;background:#0d1117;padding:10px 12px;margin:8px 0'>"
            "<div style='color:#c7c7e6;margin-bottom:4px'>Equity ($) &middot; " + legend + "</div>" + svg + "</div>" +
            table + cav + "<hr style='border-color:#21262d;margin:24px 0'>")


'''

CHART_HELPER = r'''
def _tjl_chart(series, vmin, vmax, start_date, end_date, kind="dollar", baseline=None):
    """Inline SVG line chart with a y-axis (price / %) and x-axis (dates). Uniform scaling
    (no preserveAspectRatio='none') so axis labels stay crisp. series: [(color, [{x,v}])]."""
    import datetime as _dt
    L, R, TOP, BOT = 64.0, 950.0, 14.0, 262.0
    rng = (vmax - vmin) or 1.0
    def _X(x):
        return L + x * (R - L)
    def _Y(v):
        return TOP + (1 - (v - vmin) / rng) * (BOT - TOP)
    def _vf(v):
        if kind == "pct":
            return ("+" if v > 0 else "") + str(int(round(v))) + "%"
        if abs(v) >= 1000:
            return "$" + format(int(round(v / 1000.0)), ",") + "k"
        return "$" + str(int(round(v)))
    parts = []
    for i in range(5):                                       # y gridlines + value labels
        v = vmin + rng * i / 4.0
        y = _Y(v)
        parts.append("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='#21262d'/>" % (L, y, R, y))
        parts.append("<text x='%.1f' y='%.1f' fill='#8786b5' font-size='12' text-anchor='end' "
                     "dominant-baseline='middle'>%s</text>" % (L - 6, y, _vf(v)))
    try:                                                     # x date ticks + labels
        d0 = _dt.date.fromisoformat(str(start_date)[:10]); d1 = _dt.date.fromisoformat(str(end_date)[:10])
        span = (d1 - d0).days or 1
        for i in range(5):
            f = i / 4.0; x = _X(f)
            dd = d0 + _dt.timedelta(days=int(round(span * f)))
            anc = "start" if i == 0 else ("end" if i == 4 else "middle")
            parts.append("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='#161b22'/>" % (x, TOP, x, BOT))
            parts.append("<text x='%.1f' y='%.1f' fill='#8786b5' font-size='12' text-anchor='%s'>%s</text>"
                         % (x, BOT + 17, anc, dd.strftime("%b %d")))
    except Exception:
        pass
    if baseline is not None:
        by = _Y(baseline)
        parts.append("<line x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f' stroke='#30363d' "
                     "stroke-dasharray='4 4'/>" % (L, by, R, by))
    for col, pts in series:
        if len(pts) >= 2:
            dpts = " ".join("%.1f,%.1f" % (_X(p["x"]), _Y(p["v"])) for p in pts)
            parts.append("<polyline points='%s' fill='none' stroke='%s' stroke-width='1.8'/>" % (dpts, col))
    return "<svg viewBox='0 0 960 286' style='width:100%;height:auto;display:block'>" + "".join(parts) + "</svg>"


'''

anchor = '@app.route(PREFIX + "/backtest")'
old_ret = 'return page("Strategy", iframe, "backtest")'
new_ret = 'return page("Strategy", _render_tjl_portfolio() + _render_tjl_backtest() + iframe, "backtest")'

if anchor not in src or old_ret not in src:
    print("ABORT: expected anchors not found (dashboard.py shape differs) — no change made")
    sys.exit(2)

src = src.replace(anchor, CHART_HELPER + RENDERER + PORTFOLIO_RENDERER + anchor, 1)   # helper + renderers
src = src.replace(old_ret, new_ret, 1)            # show TJL above the existing tearsheet

# --- /dash/logs: add an "exec health" source sub-tab (renders exec_health_latest.txt) ---
_h_tail = '    src = request.args.get("src", "run")'
if _h_tail in src:
    src = src.replace(_h_tail, _h_tail +
        '\n    if src == "health":'
        '\n        _hf = ROOT / "logs-instance2" / "exec_health_latest.txt"'
        '\n        return Response(_hf.read_text() if _hf.exists() else '
        '"(no exec health report yet — runs weekday mornings 10:25 ET)", mimetype="text/plain")', 1)
_h_tab = '''<a href="#" id=src-webhook class=navlink onclick="SRC='webhook';mark();refresh();return false">webhook</a>'''
if _h_tab in src:
    src = src.replace(_h_tab, _h_tab +
        '''\n      <a href="#" id=src-health class=navlink onclick="SRC='health';mark();refresh();return false">exec health</a>''', 1)
_h_mark = "document.getElementById('src-webhook').className=(SRC=='webhook'?'navlink on':'navlink');}}"
if _h_mark in src:
    src = src.replace(_h_mark,
        "document.getElementById('src-webhook').className=(SRC=='webhook'?'navlink on':'navlink');"
        "\n      document.getElementById('src-health').className=(SRC=='health'?'navlink on':'navlink');}}", 1)

open(P, "w", encoding="utf-8").write(src)
print("patched OK")
