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
    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    pwd = os.environ.get("SMTP_PASS")
    port = int(os.environ.get("SMTP_PORT", "587"))
    to = os.environ.get("ORDER_EMAIL_TO", "contact@quantumquant.trade")
    frm = os.environ.get("SMTP_FROM", user or to)
    if not (host and user and pwd):
        raise RuntimeError("SMTP not configured (need SMTP_HOST / SMTP_USER / SMTP_PASS in .env)")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = frm, to, subject
    msg.set_content(body)
    with smtplib.SMTP(host, port, timeout=30) as s:
        s.starttls()
        s.login(user, pwd)
        s.send_message(msg)


def compose(o, reason: str) -> str:
    und = o.symbol if not _occ(o.symbol) else o.symbol[:-15].rstrip()
    cls = "OPTION" if _occ(o.symbol) else "equity"
    qtyline = f"qty {o.qty}" if o.qty else f"notional ${o.notional}"
    return (
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
        f"{'-'*48}\n"
        f"REASONING   : {reason}\n"
    )


def main():
    _load_env()
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus
    tc = TradingClient(os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"], paper=False)
    seen = set(json.loads(STATE.read_text())) if STATE.exists() else set()
    since = datetime.now(timezone.utc) - timedelta(minutes=30)
    orders = tc.get_orders(GetOrdersRequest(status=QueryOrderStatus.CLOSED, limit=100,
                                            after=since, direction="desc"))
    new_fills = [o for o in orders if str(o.status).split(".")[-1].lower() == "filled"
                 and str(o.id) not in seen and o.filled_at]
    sent = 0
    for o in sorted(new_fills, key=lambda x: x.filled_at):
        try:
            reason = reasoning_for(o.symbol, str(o.side).split(".")[-1].lower())
            subj = (f"[QQ] {str(o.side).split('.')[-1].upper()} {o.symbol} filled @ ${o.filled_avg_price}")
            send_email(subj, compose(o, reason))
            sent += 1
        except Exception as e:
            print(f"email failed for {o.id}: {e}")
            continue
        seen.add(str(o.id))
    STATE.write_text(json.dumps(sorted(seen)[-1000:]))
    print(f"{len(new_fills)} new fills, {sent} emailed.")


if __name__ == "__main__":
    main()
