"""Research-only guard: prove there is no broker order / live-execution path."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Substrings that would indicate a trading / order-submission path.
FORBIDDEN = [
    "submit_order", "TradingClient", "/v2/orders", "place_order",
    "paper-api.alpaca.markets", "api.alpaca.markets/v2/orders",
    "alpaca.trading", "create_order", "OrderSide.BUY", "OrderSide.SELL",
]

# These are internal research dataclasses, not broker calls -- allow them.
ALLOWED_TOKENS = {"OrderRequest", "RiskDecision"}


def _py_files():
    for sub in ("rl_meta", "scripts"):
        yield from (ROOT / sub).rglob("*.py")


def test_no_broker_order_path():
    offenders = []
    for f in _py_files():
        text = f.read_text()
        for needle in FORBIDDEN:
            if needle in text:
                offenders.append(f"{f.name}: {needle}")
    assert not offenders, f"broker/order path found: {offenders}"


def test_data_layer_uses_data_host_only():
    text = (ROOT / "rl_meta" / "data.py").read_text()
    assert "data.alpaca.markets" in text
    # Never the trading host.
    assert "paper-api.alpaca.markets" not in text
    assert "api.alpaca.markets/v2/orders" not in text


def test_requirements_have_no_broker_sdk():
    # Inspect dependency lines only -- comments may legitimately *name* the SDKs
    # they exclude.
    deps = [ln.split("#", 1)[0].strip().lower()
            for ln in (ROOT / "requirements.txt").read_text().splitlines()]
    deps = [d for d in deps if d]
    assert not any("alpaca-py" in d or "alpaca-trade-api" in d for d in deps)


def test_internal_order_dataclass_is_not_a_broker_call():
    # risk.py defines OrderRequest as an in-memory research object; ensure it is
    # not imported from any alpaca/broker module.
    text = (ROOT / "rl_meta" / "risk.py").read_text()
    assert "import alpaca" not in text and "from alpaca" not in text
    assert "OrderRequest" in text  # sanity: the internal type still exists
