#!/usr/bin/env python3
"""Generate crossover_tearsheet.html — the /dash/backtest page content for instance2 (nonl0cal.2).
Renders the gates-off BAL backtest (4.44y + 1y), the survivorship drop-winners test, and the
1-year gate ablation, with self-contained inline-SVG equity curves (no CDN). Read-only inputs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "crossover_tearsheet.html"

GEN = "2026-06-12"

# ── headline results ─────────────────────────────────────────────────────────
HEADLINE = {
    "4.44-year (2022-01-03 → 2026-06-12)": [
        ("BAL gates-off ($1M, regime+caps)", 1773261, 77.3, 13.8, 19.3, 0.77, -25.2, True),
        ("Equal-weight basket B&H (no regime)", 1819133, 81.9, 14.4, 23.9, 0.69, -34.9, False),
        ("SPY buy & hold", 1646065, 64.6, 11.9, 17.6, 0.73, -24.5, False),
    ],
    "1-year (2025-06-12 → 2026-06-12)": [
        ("BAL gates-off ($1M, regime+caps)", 1686935, 68.7, 68.8, 22.3, 2.47, -13.2, True),
        ("Equal-weight basket B&H (no regime)", 1795554, 79.6, 79.6, 25.9, 2.40, -14.3, False),
        ("SPY buy & hold", 1242712, 24.3, 24.3, 12.3, 1.83, -8.9, False),
    ],
}

# ── survivorship drop-winners test (ret%, sharpe, dd%, beats_spy) ─────────────
DROP = {
    "4.44-year  (SPY: +64.6%, Sharpe 0.73, DD −24.5%)": [
        ("Baseline (all 44)", "—", 77.3, 0.77, -25.2, True),
        ("Drop top-2", "AXSM +575%, CELC +564%", 56.1, 0.62, -26.0, False),
        ("Drop top-6", "+NUVL, ORKA, PTGX, TT", 28.8, 0.40, -25.1, False),
    ],
    "1-year  (SPY: +24.3%, Sharpe 1.83, DD −8.9%)": [
        ("Baseline (all 44)", "—", 68.7, 2.47, -13.2, True),
        ("Drop top-2", "ERAS +818%, CELC +605%", 42.5, 2.08, -7.3, True),
        ("Drop top-6", "+ORKA, SYRE, DNTH, AXSM", 17.3, 1.08, -11.2, False),
    ],
}

# ── 1-year ablation (one factor at a time) — final$, ret%, sharpe, dd%, inv% ──
ABLATION = [
    ("baseline (gates-off: regime + all caps ON)", 1686935, 68.69, 2.47, -13.21, 81.4, "ref"),
    ("− regime gate (SPY>200DMA)", 1686935, 68.69, 2.47, -13.21, 81.4, "null"),
    ("− exposure cap (80%→100%)", 1801005, 80.10, 2.48, -15.01, 92.0, "binds"),
    ("− name cap (10%)", 1686935, 68.69, 2.47, -13.21, 81.4, "null"),
    ("− max-tranches (5→∞)", 1691424, 69.14, 2.47, -13.26, 82.6, "null"),
    ("− cooldown (3d→0)", 1679320, 67.93, 2.48, -12.88, 82.5, "null"),
    ("− cash buffer (8%→0)", 1686935, 68.69, 2.47, -13.21, 81.4, "null"),
    ("+ trend filter (name>200DMA) — proxy quality gate", 1502012, 50.20, 2.16, -11.63, 73.2, "hurts"),
    ("+ dip filter (name<50DMA) — proxy valuation gate", 1478735, 47.87, 2.12, -13.25, 68.7, "hurts"),
    ("ALL structural caps OFF", 1888031, 88.80, 2.57, -14.75, 99.2, "binds"),
]
SPY_1Y = (1242712, 24.3, 1.83, -8.9)


def svg_equity(csv_path, title):
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    series = [("bal_gates_off", "#3fb950", "BAL gates-off"),
              ("ew_basket", "#d29922", "EW basket"),
              ("spy", "#58a6ff", "SPY B&H")]
    W, H, PADL, PADB, PADT = 1000, 320, 70, 28, 14
    vals = pd.concat([df[c] for c, _, _ in series])
    lo, hi = vals.min(), vals.max()
    span = (hi - lo) or 1.0
    n = len(df)
    step = max(1, n // 320)
    idx = list(range(0, n, step))
    if idx[-1] != n - 1:
        idx.append(n - 1)

    def x(i): return PADL + (W - PADL - 10) * i / (n - 1)
    def y(v): return PADT + (H - PADT - PADB) * (1 - (v - lo) / span)

    paths = []
    for col, color, _ in series:
        pts = " ".join(f"{x(i):.1f},{y(df[col].iloc[i]):.1f}" for i in idx)
        paths.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/>')
    # gridlines + y labels (5)
    grid = []
    for k in range(5):
        v = lo + span * k / 4
        yy = y(v)
        grid.append(f'<line x1="{PADL}" y1="{yy:.1f}" x2="{W-10}" y2="{yy:.1f}" stroke="#21262d" stroke-width="1"/>')
        grid.append(f'<text x="{PADL-8}" y="{yy+4:.1f}" fill="#8b949e" font-size="11" text-anchor="end">${v/1000:.0f}k</text>')
    # x labels (start / mid / end)
    xt = []
    for i in (0, n // 2, n - 1):
        d = df["timestamp"].iloc[i].strftime("%b %y")
        xt.append(f'<text x="{x(i):.1f}" y="{H-8}" fill="#8b949e" font-size="11" text-anchor="middle">{d}</text>')
    # legend
    leg, lx = [], PADL + 6
    for col, color, name in series:
        leg.append(f'<rect x="{lx}" y="{PADT}" width="11" height="11" fill="{color}"/>')
        leg.append(f'<text x="{lx+16}" y="{PADT+10}" fill="#c9d1d9" font-size="11">{name}</text>')
        lx += 130
    return (f'<div class="chart-title">{title}</div>'
            f'<svg viewBox="0 0 {W} {H}" width="100%" preserveAspectRatio="xMidYMid meet">'
            f'{"".join(grid)}{"".join(paths)}{"".join(xt)}{"".join(leg)}</svg>')


def headline_table(title, rows):
    body = ""
    for name, final, ret, cagr, vol, sharpe, dd, hl in rows:
        cls = ' class="hl"' if hl else ""
        body += (f"<tr{cls}><td>{name}</td><td class='num'>${final:,.0f}</td>"
                 f"<td class='num'>+{ret:.1f}%</td><td class='num'>{cagr:.1f}%</td>"
                 f"<td class='num'>{vol:.1f}%</td><td class='num'>{sharpe:.2f}</td>"
                 f"<td class='num neg'>{dd:.1f}%</td></tr>")
    return (f"<h3>{title}</h3><table><thead><tr><th>strategy</th><th>final $</th>"
            f"<th>tot ret</th><th>CAGR</th><th>vol</th><th>Sharpe</th><th>max DD</th>"
            f"</tr></thead><tbody>{body}</tbody></table>")


def drop_table(title, rows):
    body = ""
    for name, dropped, ret, sharpe, dd, beats in rows:
        badge = '<span class="ok">✓ beats SPY</span>' if beats else '<span class="bad">✗ loses</span>'
        body += (f"<tr><td>{name}</td><td class='muted'>{dropped}</td>"
                 f"<td class='num'>+{ret:.1f}%</td><td class='num'>{sharpe:.2f}</td>"
                 f"<td class='num neg'>{dd:.1f}%</td><td>{badge}</td></tr>")
    return (f"<h3>{title}</h3><table><thead><tr><th>config</th><th>dropped</th>"
            f"<th>tot ret</th><th>Sharpe</th><th>max DD</th><th>vs SPY</th>"
            f"</tr></thead><tbody>{body}</tbody></table>")


def ablation_table():
    base_ret = ABLATION[0][2]
    body = ""
    tag_cls = {"ref": "muted", "null": "muted", "binds": "warn", "hurts": "bad"}
    tag_txt = {"ref": "reference", "null": "non-binding", "binds": "binds", "hurts": "hurts"}
    for name, final, ret, sharpe, dd, inv, tag in ABLATION:
        delta = ret - base_ret
        dtxt = "—" if tag == "ref" else (f"+{delta:.1f}" if delta >= 0 else f"{delta:.1f}")
        body += (f"<tr><td>{name}</td><td class='num'>${final:,.0f}</td>"
                 f"<td class='num'>+{ret:.1f}%</td><td class='num'>{dtxt}</td>"
                 f"<td class='num'>{sharpe:.2f}</td><td class='num neg'>{dd:.1f}%</td>"
                 f"<td class='num'>{inv:.0f}%</td>"
                 f"<td><span class='{tag_cls[tag]}'>{tag_txt[tag]}</span></td></tr>")
    f, r, s, d = SPY_1Y
    body += (f"<tr class='spy'><td>SPY buy &amp; hold</td><td class='num'>${f:,.0f}</td>"
             f"<td class='num'>+{r:.1f}%</td><td class='num'>—</td><td class='num'>{s:.2f}</td>"
             f"<td class='num neg'>{d:.1f}%</td><td class='num'>100%</td><td class='muted'>benchmark</td></tr>")
    return (f"<table><thead><tr><th>factor toggled (vs gates-off baseline)</th><th>final $</th>"
            f"<th>tot ret</th><th>Δ ret</th><th>Sharpe</th><th>max DD</th><th>avg inv</th><th>effect</th>"
            f"</tr></thead><tbody>{body}</tbody></table>")


html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BAL Gates-Off — Backtest & Ablation</title>
<style>
  body{{background:#0d1117;color:#c9d1d9;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;padding:22px}}
  h1{{font-size:20px;margin:0 0 2px}} h2{{font-size:16px;margin:30px 0 6px;color:#e6edf3;border-bottom:1px solid #21262d;padding-bottom:5px}}
  h3{{font-size:13px;margin:16px 0 6px;color:#8b949e;font-weight:600}}
  .sub{{color:#8b949e;font-size:12px;margin-bottom:4px}}
  table{{border-collapse:collapse;width:100%;margin:4px 0 10px;font-size:13px}}
  th,td{{padding:6px 9px;text-align:left;border-bottom:1px solid #1b1f24}}
  th{{color:#8b949e;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.3px}}
  td.num{{text-align:right;font-variant-numeric:tabular-nums}}
  tr.hl{{background:#16301d}} tr.hl td{{color:#e6edf3;font-weight:600}}
  tr.spy{{background:#11233b}}
  .neg{{color:#f85149}} .ok{{color:#3fb950;font-weight:600}} .bad{{color:#f85149;font-weight:600}}
  .warn{{color:#d29922;font-weight:600}} .muted{{color:#6e7681}}
  .chart-title{{color:#8b949e;font-size:12px;margin:10px 0 2px}}
  .card{{background:#0f141a;border:1px solid #21262d;border-radius:10px;padding:14px 16px;margin:10px 0}}
  .verdict{{background:#1c1106;border:1px solid #5a3a14}}
  .verdict b{{color:#e6edf3}}
  code{{background:#161b22;padding:1px 5px;border-radius:4px;font-size:12px}}
  .tag{{display:inline-block;background:#1f6feb22;border:1px solid #1f6feb55;color:#58a6ff;border-radius:6px;padding:1px 8px;font-size:11px;margin-left:6px}}
</style></head><body>
<h1>Buffett Accumulation Layer — Gates-Off <span class="tag">nonl0cal.2 · $1M paper</span></h1>
<div class="sub">Replay of the live instance-2 gates-off config (buffett_score + valuation bypass, min_score=0,
conviction_small=0, MAX_ORDER_DOLLARS=15000) on the 44-name exclusive universe. Total-return fills at daily
close, SIP. Controls still ON: SPY&gt;200-DMA regime gate · 5 tranches/name ≥3d · 10%/name · 80% exposure · 8% cash buffer.
Generated {GEN}.</div>

<h2>Headline — gates-off vs benchmarks</h2>
<div class="card">{headline_table(list(HEADLINE)[0], HEADLINE[list(HEADLINE)[0]])}
{headline_table(list(HEADLINE)[1], HEADLINE[list(HEADLINE)[1]])}</div>

<h2>Equity curves</h2>
<div class="card">{svg_equity(ROOT/'backtest_outputs'/'eq_4y.csv', '4.44-year — $1M start')}</div>
<div class="card">{svg_equity(ROOT/'backtest_outputs'/'eq_1y.csv', '1-year — $1M start')}</div>

<h2>Survivorship test — drop the top winners</h2>
<div class="sub">Rank each name by its own return over the window, remove the top-N, re-run. If the edge is real it survives.</div>
<div class="card">{drop_table(list(DROP)[0], DROP[list(DROP)[0]])}
{drop_table(list(DROP)[1], DROP[list(DROP)[1]])}</div>

<h2>Gate ablation — 1-year, one factor at a time</h2>
<div class="sub">Quality/valuation/conviction gates are fundamental (not PIT-backtestable); shown here are the
price-computable gates the bot actually applies, plus labelled price <i>proxies</i> for the selection gates.</div>
<div class="card">{ablation_table()}</div>

<h2>Read</h2>
<div class="card verdict">
<p><b>The edge is survivorship, not strategy.</b> Headline gates-off beats SPY (4.44y +77% vs +65%; 1y +69% vs +24%),
but dropping just <b>2 names</b> flips the 4.44y result <i>below</i> SPY risk-adjusted (Sharpe 0.77→0.62), and dropping
6 collapses it (0.40). The 44-name list was curated <i>today</i>, so a back-replay bakes in the winners — ~6 biotech
names (ERAS/AXSM/CELC/NUVL/ORKA/SYRE) carry ~70% of the gain.</p>
<p><b>In the 1-year up-only tape, most gates never bind.</b> Regime (SPY never dipped below its 200-DMA), name-cap,
max-tranches and cash-buffer are <span class="muted">non-binding</span> — only the 80% <span class="warn">exposure cap</span>
constrains, costing ~11 pts of return for almost no Sharpe gain. The two <span class="bad">selection proxies hurt</span>
(trend/dip filters cut return ~18–21 pts) — any selectivity that delays buying the curated winners is penalised in this window.</p>
<p><b>What's actually durable:</b> the regime gate + caps are <i>insurance</i> — free in this melt-up, but in the 4.44y
window (incl. 2022) the regime gate cut drawdown −35%→−25%. Gates-off isn't <i>reckless</i> (Sharpe ≥ SPY, DD controlled),
but it shows <b>no forward edge</b> — it just dilutes into the basket and rides whatever's hot. Consistent with the rest
of the system: only the insider copier has a validated edge.</p>
</div>
<div class="sub">Harness: <code>backtest_bal_gates_off.py &lt;start&gt; &lt;drop_top_n&gt;</code> · ablation via <code>ABL_*</code> env toggles.
Paper research only — risk/exit/kill controls intact.</div>
</body></html>"""

OUT.write_text(html)
print(f"wrote {OUT}  ({len(html):,} bytes)")
