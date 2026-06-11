#!/usr/bin/env python3
"""
trade_journal.py — HTML trade journal of ROUND TRIPS reconstructed from the Alpaca order history.

Alpaca doesn't expose round trips directly, so this pulls all filled orders and FIFO-matches opens
against closes (per symbol, signed-quantity → handles longs, shorts, scale-in/out, partial fills,
equities + options with the 100x multiplier). Emits a self-contained trade_journal.html: summary
cards, cumulative realized-P&L chart, a sortable round-trip table, and a per-symbol breakdown.

Creds come from the .env in the working directory (APCA_API_KEY_ID / APCA_API_SECRET_KEY). Paper vs
live is auto-detected from APCA_API_BASE_URL; override with --paper / --live.
  python trade_journal.py [--paper|--live] [--out trade_journal.html] [--days N]
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _load_env():
    f = Path(".env")
    if f.exists():
        for line in f.read_text().splitlines():
            s = line.strip()
            if "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _is_option(sym: str) -> bool:
    return len(sym) >= 15 and sym[-9] in ("C", "P") and sym[-8:].isdigit()


def _occ_parts(sym: str):
    """(underlying, 'CALL'/'PUT'/'', strike) for OCC; ('', '', None) for equity."""
    if not _is_option(sym):
        return sym, "", None
    import re
    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", sym)
    if not m:
        return sym, "OPT", None
    return m.group(1), ("CALL" if m.group(3) == "C" else "PUT"), int(m.group(4)) / 1000.0


def fetch_fills(client, days):
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    fills, until = [], None
    after = datetime.now(timezone.utc) - timedelta(days=days) if days else None
    for _ in range(40):                              # paginate (500/page)
        req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500,
                               direction="desc", until=until, after=after, nested=False)
        batch = client.get_orders(req)
        if not batch:
            break
        for o in batch:
            if o.filled_at and o.filled_qty and float(o.filled_qty) > 0 and o.filled_avg_price:
                fills.append({
                    "symbol": str(o.symbol), "side": str(o.side).split(".")[-1].lower(),
                    "qty": float(o.filled_qty), "price": float(o.filled_avg_price),
                    "time": o.filled_at, "id": str(o.id),
                })
        until = batch[-1].submitted_at
        if len(batch) < 500:
            break
    fills.sort(key=lambda f: f["time"])
    return fills


def round_trips(fills):
    """FIFO match opens vs closes per symbol using signed quantities."""
    books = defaultdict(deque)        # symbol -> deque of open lots {qty(signed), price, time}
    trips = []
    for f in fills:
        sym = f["symbol"]
        signed = f["qty"] if f["side"] == "buy" else -f["qty"]
        book = books[sym]
        pos = sum(l["qty"] for l in book)
        # opening (same direction or flat) → push lot
        if pos == 0 or (pos > 0) == (signed > 0):
            book.append({"qty": signed, "price": f["price"], "time": f["time"]})
            continue
        # closing → match FIFO against opposite lots
        remaining = signed
        mult = 100.0 if _is_option(sym) else 1.0
        while remaining != 0 and book and (book[0]["qty"] > 0) != (remaining > 0):
            lot = book[0]
            match = min(abs(lot["qty"]), abs(remaining))
            long_side = lot["qty"] > 0
            entry_px, exit_px = lot["price"], f["price"]
            pnl = (exit_px - entry_px) * match * mult * (1 if long_side else -1)
            basis = entry_px * match * mult
            trips.append({
                "symbol": sym, "long": long_side, "qty": match,
                "entry_time": lot["time"], "entry_px": entry_px,
                "exit_time": f["time"], "exit_px": exit_px,
                "pnl": pnl, "ret": (pnl / basis) if basis else 0.0,
                "hold_h": (f["time"] - lot["time"]).total_seconds() / 3600.0,
            })
            lot["qty"] -= match if long_side else -match
            remaining += match if not long_side else -match
            if abs(lot["qty"]) < 1e-9:
                book.popleft()
        if remaining != 0:            # flipped through zero → opens new lot other side
            book.append({"qty": remaining, "price": f["price"], "time": f["time"]})
    return trips


def chart_b64(trips):
    if not trips:
        return ""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ts = sorted(trips, key=lambda t: t["exit_time"])
    x = [t["exit_time"] for t in ts]
    cum, run = [], 0.0
    for t in ts:
        run += t["pnl"]; cum.append(run)
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(x, cum, lw=1.6, color="#1f77b4")
    ax.fill_between(x, cum, 0, where=[c >= 0 for c in cum], color="#2ca02c", alpha=0.15)
    ax.fill_between(x, cum, 0, where=[c < 0 for c in cum], color="#d62728", alpha=0.15)
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_title("Cumulative realized P&L ($)", fontsize=11)
    ax.grid(alpha=0.25)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=110, bbox_inches="tight"); plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _fmt_dt(d):
    return d.astimezone().strftime("%Y-%m-%d %H:%M")


def render_html(trips, account, mode):
    n = len(trips)
    wins = [t for t in trips if t["pnl"] > 0]
    losses = [t for t in trips if t["pnl"] < 0]
    tot = sum(t["pnl"] for t in trips)
    gp = sum(t["pnl"] for t in wins); gl = sum(t["pnl"] for t in losses)
    pf = (gp / abs(gl)) if gl else float("inf")
    win_rate = (len(wins) / n * 100) if n else 0
    avg_w = (gp / len(wins)) if wins else 0
    avg_l = (gl / len(losses)) if losses else 0
    avg_hold = (sum(t["hold_h"] for t in trips) / n) if n else 0
    best = max(trips, key=lambda t: t["pnl"]) if trips else None
    worst = min(trips, key=lambda t: t["pnl"]) if trips else None

    def card(label, val, cls=""):
        return f'<div class="card"><div class="lbl">{label}</div><div class="val {cls}">{val}</div></div>'

    pf_s = "∞" if pf == float("inf") else f"{pf:.2f}"
    cards = "".join([
        card("Round trips", f"{n}"),
        card("Realized P&L", f"${tot:,.0f}", "pos" if tot >= 0 else "neg"),
        card("Win rate", f"{win_rate:.0f}%"),
        card("Profit factor", pf_s, "pos" if pf >= 1 else "neg"),
        card("Avg win", f"${avg_w:,.0f}", "pos"),
        card("Avg loss", f"${avg_l:,.0f}", "neg"),
        card("Avg hold", f"{avg_hold:.1f}h"),
        card("Best", f"${best['pnl']:,.0f}" if best else "—", "pos"),
        card("Worst", f"${worst['pnl']:,.0f}" if worst else "—", "neg"),
    ])

    rows = ""
    for i, t in enumerate(sorted(trips, key=lambda x: x["exit_time"], reverse=True), 1):
        und, kind, strike = _occ_parts(t["symbol"])
        typ = (f"{kind} {strike:g}" if kind else "EQ") + ("" if t["long"] else " (short)")
        pcls = "pos" if t["pnl"] >= 0 else "neg"
        rows += (f"<tr><td>{i}</td><td class='sym'>{t['symbol']}</td><td>{typ}</td>"
                 f"<td class='num'>{t['qty']:g}</td>"
                 f"<td>{_fmt_dt(t['entry_time'])}</td><td class='num'>{t['entry_px']:.2f}</td>"
                 f"<td>{_fmt_dt(t['exit_time'])}</td><td class='num'>{t['exit_px']:.2f}</td>"
                 f"<td class='num'>{t['hold_h']:.1f}h</td>"
                 f"<td class='num {pcls}'>${t['pnl']:,.0f}</td>"
                 f"<td class='num {pcls}'>{t['ret']*100:+.1f}%</td></tr>")

    by_sym = defaultdict(lambda: [0, 0.0])
    for t in trips:
        by_sym[_occ_parts(t["symbol"])[0]][0] += 1
        by_sym[_occ_parts(t["symbol"])[0]][1] += t["pnl"]
    sym_rows = "".join(
        f"<tr><td class='sym'>{s}</td><td class='num'>{c}</td>"
        f"<td class='num {'pos' if p>=0 else 'neg'}'>${p:,.0f}</td></tr>"
        for s, (c, p) in sorted(by_sym.items(), key=lambda kv: kv[1][1])
    )

    img = chart_b64(trips)
    chart = f'<img src="data:image/png;base64,{img}" style="width:100%">' if img else "<p>No trades.</p>"
    gen = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    badge = "LIVE" if mode == "live" else "PAPER"
    bcls = "neg" if mode == "live" else "pos"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Trade Journal</title>
<style>
*{{box-sizing:border-box}} body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0f1116;color:#e6e6e6}}
.wrap{{max-width:1200px;margin:0 auto;padding:24px}}
h1{{font-size:22px;margin:0 0 4px}} .meta{{color:#9aa;font-size:13px;margin-bottom:18px}}
.badge{{padding:2px 8px;border-radius:4px;font-weight:700;font-size:12px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;margin-bottom:20px}}
.card{{background:#1a1d26;border:1px solid #262a36;border-radius:8px;padding:12px}}
.lbl{{color:#8a90a0;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
.val{{font-size:20px;font-weight:700;margin-top:4px}}
.pos{{color:#3fb950}} .neg{{color:#f85149}}
.chart{{background:#1a1d26;border:1px solid #262a36;border-radius:8px;padding:12px;margin-bottom:20px}}
table{{width:100%;border-collapse:collapse;font-size:13px;background:#1a1d26;border-radius:8px;overflow:hidden}}
th,td{{padding:7px 10px;text-align:left;border-bottom:1px solid #20242f}}
th{{background:#20242f;color:#aab;font-size:11px;text-transform:uppercase;cursor:pointer;user-select:none}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}} td.sym{{font-family:ui-monospace,monospace;font-size:12px}}
tr:hover td{{background:#20242f}} h2{{font-size:15px;margin:24px 0 8px}}
.two{{display:grid;grid-template-columns:1fr 320px;gap:20px;align-items:start}}
@media(max-width:800px){{.two{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap">
<h1>Trade Journal <span class="badge {bcls}">{badge}</span></h1>
<div class="meta">Account {account} · {n} round trips · generated {gen} · round trips FIFO-matched from Alpaca fills</div>
<div class="cards">{cards}</div>
<div class="chart">{chart}</div>
<div class="two">
<div><h2>Round trips</h2><table id="t"><thead><tr>
<th>#</th><th>Symbol</th><th>Type</th><th>Qty</th><th>Entry</th><th>In $</th><th>Exit</th><th>Out $</th><th>Hold</th><th>P&amp;L</th><th>Ret</th>
</tr></thead><tbody>{rows}</tbody></table></div>
<div><h2>By symbol</h2><table><thead><tr><th>Symbol</th><th>Trips</th><th>P&amp;L</th></tr></thead><tbody>{sym_rows}</tbody></table></div>
</div>
<script>
document.querySelectorAll('#t th').forEach((h,i)=>h.onclick=()=>{{
 const tb=document.querySelector('#t tbody'),rows=[...tb.rows];
 const num=v=>parseFloat(v.replace(/[$,%h]/g,''))||v;
 const asc=h.dataset.a=h.dataset.a?'':'1';
 rows.sort((a,b)=>{{let x=num(a.cells[i].innerText),y=num(b.cells[i].innerText);
  return (x>y?1:x<y?-1:0)*(asc?1:-1);}});
 rows.forEach(r=>tb.appendChild(r));}});
</script>
</div></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true")
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--out", default="trade_journal.html")
    ap.add_argument("--days", type=int, default=0, help="lookback days (0 = all available)")
    args = ap.parse_args()
    _load_env()
    key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        sys.exit("No Alpaca creds in .env (APCA_API_KEY_ID / APCA_API_SECRET_KEY)")
    url = (os.environ.get("APCA_API_BASE_URL") or "").lower()
    paper = True if args.paper else (False if args.live else ("paper" in url or not url))
    mode = "live" if not paper else "paper"
    from alpaca.trading.client import TradingClient
    client = TradingClient(key, sec, paper=paper)
    acct = client.get_account()
    print(f"Account {acct.account_number} ({mode}) — fetching fills...")
    fills = fetch_fills(client, args.days)
    trips = round_trips(fills)
    print(f"{len(fills)} fills → {len(trips)} round trips")
    html = render_html(trips, acct.account_number, mode)
    Path(args.out).write_text(html)
    realized = sum(t["pnl"] for t in trips)
    print(f"Realized P&L across round trips: ${realized:,.2f}")
    print(f"Journal → {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
