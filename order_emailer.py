#!/usr/bin/env python3
"""
order_emailer.py — emails contact@quantumquant.trade on every CONFIRMED (filled) Alpaca order,
with full order details + the bot's reasoning for the execution.

Runs on the droplet (has the live Alpaca creds + run.log). Polls Alpaca for newly-FILLED orders
since the last run (state persisted), correlates each fill's reasoning from the bot/webhook logs,
and sends a formatted email via SMTP. Idempotent — a given fill is emailed once.

Reasoning sources, in order: run.py's run.log ("ORDER <side> <SYM> ... reason=..."), the
trade-webhook journal (TradingView signals), else "external/manual".

Required env (in .env): APCA_API_KEY_ID, APCA_API_SECRET_KEY, and SMTP creds:
  SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASS, SMTP_FROM (defaults to SMTP_USER)
  ORDER_EMAIL_TO (defaults to contact@quantumquant.trade)
"""
from __future__ import annotations

import json
import os
import re
import smtplib
import subprocess
from datetime import datetime, timezone, timedelta
from email.message import EmailMessage
from pathlib import Path
from collections import defaultdict, deque

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "order_emailer_state.json"
RUN_LOG = ROOT / "logs" / "run.log"


def _load_env():
    f = ROOT / ".env"
    if f.exists():
        for line in f.read_text().splitlines():
            s = line.strip()
            if "=" in s and not s.startswith("#"):
                k, v = s.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _occ(sym: str) -> bool:
    return len(sym) >= 15 and sym[-9] in ("C", "P") and sym[-8:].isdigit()


def fetch_fills(client, days=120):
    """All filled orders over `days`, oldest-first (FIFO input)."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    fills, until = [], None
    after = datetime.now(timezone.utc) - timedelta(days=days)
    for _ in range(40):
        batch = client.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=500,
                                                   direction="desc", until=until, after=after))
        if not batch:
            break
        for o in batch:
            if o.filled_at and o.filled_qty and float(o.filled_qty) > 0 and o.filled_avg_price:
                fills.append({"symbol": str(o.symbol), "side": str(o.side).split(".")[-1].lower(),
                              "qty": float(o.filled_qty), "price": float(o.filled_avg_price),
                              "time": o.filled_at})
        until = batch[-1].submitted_at
        if len(batch) < 500:
            break
    fills.sort(key=lambda f: f["time"])
    return fills


def round_trips(fills):
    """FIFO-match opens vs closes per symbol; each trip carries entry_px, exit_px, pnl, ret, hold_h."""
    books = defaultdict(deque)
    trips = []
    for f in fills:
        sym = f["symbol"]
        signed = f["qty"] if f["side"] == "buy" else -f["qty"]
        book = books[sym]
        pos = sum(l["qty"] for l in book)
        if pos == 0 or (pos > 0) == (signed > 0):
            book.append({"qty": signed, "price": f["price"], "time": f["time"]})
            continue
        remaining = signed
        mult = 100.0 if _occ(sym) else 1.0
        while remaining != 0 and book and (book[0]["qty"] > 0) != (remaining > 0):
            lot = book[0]
            match = min(abs(lot["qty"]), abs(remaining))
            long_side = lot["qty"] > 0
            pnl = (f["price"] - lot["price"]) * match * mult * (1 if long_side else -1)
            basis = lot["price"] * match * mult
            trips.append({"symbol": sym, "qty": match, "entry_px": lot["price"], "exit_px": f["price"],
                          "exit_time": f["time"], "pnl": pnl, "ret": (pnl / basis) if basis else 0.0,
                          "hold_h": (f["time"] - lot["time"]).total_seconds() / 3600.0})
            lot["qty"] -= match if long_side else -match
            remaining += match if not long_side else -match
            if abs(lot["qty"]) < 1e-9:
                book.popleft()
        if remaining != 0:
            book.append({"qty": remaining, "price": f["price"], "time": f["time"]})
    return trips


def roundtrip_for_sell(trips, symbol, fill_time):
    """Aggregate the round-trip(s) this sell closed (matched by symbol + exit time). None if it opened
    a short / no match."""
    m = [t for t in trips if t["symbol"] == symbol and abs((t["exit_time"] - fill_time).total_seconds()) < 6]
    if not m:
        return None
    qty = sum(t["qty"] for t in m)
    pnl = sum(t["pnl"] for t in m)
    entry = (sum(t["entry_px"] * t["qty"] for t in m) / qty) if qty else 0.0
    basis = entry * qty * (100.0 if _occ(symbol) else 1.0)
    return {"entry_px": entry, "pnl": pnl, "ret": (pnl / basis) if basis else 0.0,
            "hold_h": max((t["hold_h"] for t in m), default=0.0)}


def reasoning_for(symbol: str, side: str) -> str:
    """Best-effort 'why this executed': the most recent bot ORDER reason for this symbol, else the
    webhook's recent signal, else external."""
    # 1. run.py log — "ORDER buy ADBE x1  reason=..." / "9EMA TAKE-PROFIT ..." / "SMA200 EXIT ..."
    try:
        if RUN_LOG.exists():
            tail = subprocess.run(["tail", "-n", "1500", str(RUN_LOG)], capture_output=True, text=True).stdout
            pat = re.compile(rf"ORDER {side} {re.escape(symbol)}\b.*?reason=(\S+)")
            hits = pat.findall(tail)
            if hits:
                return f"bot: {hits[-1]}"
            for kw in ("SMA200 EXIT", "9EMA TAKE-PROFIT", "INSIDER COPY", "INDEPENDENT OPTION EXIT"):
                m = [l for l in tail.splitlines() if kw in l and symbol in l]
                if m:
                    return f"bot: {m[-1].split(']')[-1].strip()}"
    except Exception:
        pass
    # 2. webhook journal — TradingView signals
    try:
        out = subprocess.run(["journalctl", "-u", "trade-webhook.service", "--no-pager", "-n", "400",
                              "--since", "30 min ago"], capture_output=True, text=True).stdout
        m = [l for l in out.splitlines() if symbol in l and ("signal" in l.lower() or "webhook" in l.lower())]
        if m:
            return f"tradingview-webhook: {m[-1].split('|')[-1].strip()[:160]}"
    except Exception:
        pass
    return "external / manual (no bot reasoning found)"


def send_email(subject: str, body: str):
    if os.environ.get("ORDER_EMAILER_DRYRUN"):
        print(f"--- DRYRUN (would email) ---\nSubject: {subject}\n{body}")
        return
    to = os.environ.get("ORDER_EMAIL_TO", "contact@quantumquant.trade")
    # Preferred: Resend HTTPS API (port 443 — not blocked by DigitalOcean's SMTP filter).
    api_key = os.environ.get("RESEND_API_KEY")
    if api_key:
        import urllib.request
        import urllib.error
        frm = os.environ.get("RESEND_FROM", "QuantumQuant <onboarding@resend.dev>")
        payload = json.dumps({"from": frm, "to": [to], "subject": subject, "text": body}).encode()
        req = urllib.request.Request("https://api.resend.com/emails", data=payload, method="POST",
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                r.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Resend API {e.code}: {e.read().decode()[:240]}")
        return
    # Fallback: SMTP (works locally; blocked on the DO droplet).
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    pwd = (os.environ.get("SMTP_PASS") or "").replace(" ", "")
    port = int(os.environ.get("SMTP_PORT", "587"))
    frm = os.environ.get("SMTP_FROM", user or to)
    if not (host and user and pwd):
        raise RuntimeError("No email transport configured (set RESEND_API_KEY, or SMTP_HOST/USER/PASS)")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = frm, to, subject
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)


def compose(o, reason: str, rt: dict = None) -> str:
    und = o.symbol if not _occ(o.symbol) else o.symbol[:-15].rstrip()
    cls = "OPTION" if _occ(o.symbol) else "equity"
    qtyline = f"qty {o.qty}" if o.qty else f"notional ${o.notional}"
    body = (
        f"ORDER CONFIRMED (filled) — QuantumQuant live\n"
        f"{'='*48}\n"
        f"Symbol      : {o.symbol}  ({cls}{(' on '+und) if cls=='OPTION' else ''})\n"
        f"Side        : {str(o.side).split('.')[-1].upper()}\n"
        f"Quantity    : {qtyline}\n"
        f"Fill price  : ${o.filled_avg_price}\n"
        f"Filled qty  : {o.filled_qty}\n"
        f"Notional    : ${(float(o.filled_avg_price or 0) * float(o.filled_qty or 0)):.2f}\n"
        f"Type        : {str(o.order_type).split('.')[-1]}\n"
        f"Status      : {str(o.status).split('.')[-1]}\n"
        f"Submitted   : {o.submitted_at}\n"
        f"Filled at   : {o.filled_at}\n"
        f"Order id    : {o.id}\n"
    )
    if str(o.side).split(".")[-1].lower() == "sell":
        if rt:
            sign = "PROFIT" if rt["pnl"] >= 0 else "LOSS"
            body += (
                f"{'-'*48}\n"
                f"Entry price : ${rt['entry_px']:.4f}\n"
                f"Exit price  : ${o.filled_avg_price}\n"
                f"ROUND-TRIP  : {sign} ${rt['pnl']:+.2f} ({rt['ret']*100:+.2f}%)  held {rt['hold_h']:.1f}h\n"
            )
        else:
            body += f"{'-'*48}\nRound-trip  : (no matching open lot — opening short / unmatched)\n"
    body += f"{'-'*48}\nREASONING   : {reason}\n"
    return body


def main():
    _load_env()
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    # Skip cleanly until a transport is configured (so the timer can be pre-staged without backlogging).
    if (not os.environ.get("ORDER_EMAILER_DRYRUN") and not os.environ.get("RESEND_API_KEY")
            and not (os.environ.get("SMTP_HOST") and os.environ.get("SMTP_PASS"))):
        print("No email transport configured — skipping (set RESEND_API_KEY in .env to enable).")
        return
    scope = os.environ.get("ORDER_EMAIL_SCOPE", "all").lower()   # all | equities (exclude OCC options)
    tc = TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=False)
    seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    since = datetime.now(timezone.utc) - timedelta(minutes=30)
    orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100,
                                            after=since, direction="desc"))
    new_fills = [o for o in orders if str(o.status).split(".")[-1].lower() == "filled"
                 and str(o.id) not in seen and o.filled_at
                 and (scope != "equities" or not _occ(o.symbol))]
    trips = round_trips(fetch_fills(tc, 120)) if any(str(o.side).split(".")[-1].lower() == "sell" for o in new_fills) else []
    sent = 0
    for o in sorted(new_fills, key=lambda x: x.filled_at):
        try:
            side = str(o.side).split(".")[-1].lower()
            reason = reasoning_for(o.symbol, side)
            rt = roundtrip_for_sell(trips, o.symbol, o.filled_at) if side == "sell" else None
            pnl_tag = f" (P&L ${rt['pnl']:+.0f})" if rt else ""
            subj = f"[QQ] {side.upper()} {o.symbol} filled @ ${o.filled_avg_price}{pnl_tag}"
            send_email(subj, compose(o, reason, rt))
            sent += 1
        except Exception as e:
            print(f"email failed for {o.id}: {e}")
            continue
        seen.add(str(o.id))
    STATE.write_text(json.dumps(sorted(seen)[-1000:]))
    print(f"{len(new_fills)} new fills, {sent} emailed.")


if __name__ == "__main__":
    main()
