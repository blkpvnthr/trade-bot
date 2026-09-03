#!/usr/bin/env python3
"""Live end-to-end verification of the trade.py exit-path fix (PAPER only).

Reproduces the exact bug condition against the real Alpaca paper broker and proves
the fix: an OPEN sell order holds the position's shares (qty_available == 0, the same
state a protective bracket leg creates), so a naive market sell would hit Alpaca
40310000 "insufficient qty available". The fix (`_prepare_exit`) must cancel the
symbol's working orders, free the shares, and let the exit fill.

Steps:
  1. Require market open.
  2. Market BUY a tiny qty of a liquid cheap stock; wait for fill.
  3. Place an unfillable LIMIT SELL for the full qty -> shares become held
     (qty_available -> 0).  <-- reproduces the bug condition
  4. Run the executor's _prepare_exit (cancel-then-sell) and confirm it frees the
     shares and returns the full exit qty.
  5. submit_market_sell(exit_qty) via the real broker; confirm it fills with NO
     insufficient-qty error and the position goes flat.
Always cleans up (cancels F orders + flattens) in a finally block.
"""
import sys
import time

import trade

SYM = "F"          # liquid, low-priced -> tiny notional
QTY = 2


def poll_position_qty(broker, symbol, want, timeout=45):
    for _ in range(timeout):
        for p in broker.broker.get_all_open_positions():
            if p.symbol == symbol and float(p.qty) >= want:
                return float(p.qty)
        time.sleep(1.0)
    return 0.0


def main():
    broker = trade.AlpacaBroker(env_path=trade.Path(".env"), paper=True)
    clk = broker.get_clock_info()
    print(f"[clock] is_open={clk['is_open']} next_open={clk['next_open']}")
    if not clk["is_open"]:
        print("ABORT: market is closed — run this at/after 09:30 ET. No orders placed.")
        return 2

    ex = trade.ProductionOrderExecutor.__new__(trade.ProductionOrderExecutor)
    ex.broker = broker
    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce

    ok = False
    try:
        # 1) BUY and wait for fill
        buy = broker.submit_market_buy(SYM, QTY, f"qq_vbuy_{int(time.time())}", TimeInForce.DAY)
        print(f"[buy] submitted {SYM} x{QTY} id={getattr(buy,'id','')}")
        held_qty = poll_position_qty(ex, SYM, QTY)
        if held_qty < QTY:
            print(f"ABORT: buy did not fill in time (pos qty={held_qty}).")
            return 3
        print(f"[buy] filled — position qty={held_qty}")

        # 2) Place an unfillable LIMIT SELL for the full qty -> shares held
        last = broker.get_all_open_positions()
        px = next((float(p.current_price) for p in last if p.symbol == SYM), 0.0)
        hi = round(max(px, 1.0) * 1.5, 2)
        sell_hold = broker.trading.submit_order(LimitOrderRequest(
            symbol=SYM, qty=int(held_qty), side=OrderSide.SELL, time_in_force=TimeInForce.DAY,
            limit_price=hi, client_order_id=f"qq_vhold_{int(time.time())}"))
        print(f"[hold] placed unfillable limit SELL x{int(held_qty)} @ {hi} id={sell_hold.id}")
        time.sleep(2.0)

        avail = broker.get_available_qty(SYM)
        print(f"[check] position qty={held_qty}  qty_available={avail}  (bug condition needs avail < qty)")
        bug_condition = avail < held_qty
        print(f"[check] reproduced held-shares condition: {bug_condition}")

        # 3) Run the FIX: cancel-then-sell + size-to-available
        exit_qty, meta = ex._prepare_exit(SYM, int(held_qty), "verify_cycle", "verify_failed_breakout_cut")
        print(f"[fix] _prepare_exit -> exit_qty={exit_qty} canceled={meta['orders_canceled_count']} "
              f"avail {meta['available_qty_before']}->{meta['available_qty_after']}")
        assert exit_qty == int(held_qty), f"expected to free all {held_qty}, got exit_qty={exit_qty}"
        assert meta["orders_canceled_count"] >= 1, "expected the holding order to be canceled"

        # 4) Submit the actual exit — must NOT raise insufficient-qty
        try:
            sold = broker.submit_market_sell(SYM, exit_qty, f"qq_vexit_{int(time.time())}", TimeInForce.DAY)
            print(f"[exit] market SELL x{exit_qty} submitted id={getattr(sold,'id','')} status={getattr(sold,'status','')}")
        except Exception as e:
            print(f"FAIL: exit sell raised {type(e).__name__}: {str(e)[:200]}")
            return 4

        # 5) Confirm flat
        time.sleep(3.0)
        flat = not any(p.symbol == SYM for p in broker.get_all_open_positions())
        print(f"[exit] position flat after exit: {flat}")
        ok = bug_condition and flat
        print("RESULT:", "PASS — held shares freed, exit filled, no insufficient-qty error" if ok else "CHECK — see above")
        return 0 if ok else 5
    finally:
        # Cleanup: cancel any F orders + flatten any residual F position.
        try:
            c = broker.cancel_open_orders_for_symbol(SYM)
            if c:
                print(f"[cleanup] canceled residual {SYM} orders: {c}")
            time.sleep(1.0)
            res = broker.get_available_qty(SYM)
            if res > 0:
                broker.submit_market_sell(SYM, int(res), f"qq_vcleanup_{int(time.time())}", TimeInForce.DAY)
                print(f"[cleanup] flattened residual {SYM} x{int(res)}")
        except Exception as e:
            print(f"[cleanup] warning: {type(e).__name__}: {str(e)[:150]}")


if __name__ == "__main__":
    sys.exit(main())
