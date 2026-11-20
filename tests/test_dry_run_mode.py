"""Unit tests for the dry-run order gate added to run.py's AlpacaBroker.

Pure unit tests: no network call is made. TradingClient() construction in
AlpacaBroker.__init__ does not touch the network by itself; what's under
test here is that submit_market_order/submit_limit_order short-circuit
before ever calling self.trading.submit_order when dry_run=True.
"""

from __future__ import annotations

import run


def _dry_run_broker() -> run.AlpacaBroker:
    return run.AlpacaBroker(run.ENV_PATH, paper=True, dry_run=True)


def test_dry_run_broker_blocks_market_order():
    broker = _dry_run_broker()
    result = broker.submit_market_buy("AAPL", 1, "test-coid-1", run.TimeInForce.DAY)
    assert isinstance(result, run._DryRunOrder)
    assert result.status == "dry_run_not_submitted"
    assert broker.dry_run_order_attempts == 1


def test_dry_run_broker_blocks_market_sell():
    broker = _dry_run_broker()
    broker.submit_market_sell("AAPL", 1, "test-coid-2", run.TimeInForce.DAY)
    assert broker.dry_run_order_attempts == 1


def test_dry_run_broker_blocks_limit_order():
    broker = _dry_run_broker()
    result = broker.submit_limit_buy("AAPL", 1, 150.0, "test-coid-3", run.TimeInForce.DAY)
    assert isinstance(result, run._DryRunOrder)
    assert result.status == "dry_run_not_submitted"
    assert broker.dry_run_order_attempts == 1


def test_live_mode_broker_does_not_short_circuit():
    """dry_run=False (the default) must NOT take the dry-run branch -- this
    guards against the gate accidentally flipping on for real paper/live runs."""
    broker = run.AlpacaBroker(run.ENV_PATH, paper=True, dry_run=False)
    assert broker.dry_run is False
    assert broker.dry_run_order_attempts == 0


def test_executor_config_defaults_to_no_dry_run():
    """A bare ExecutorConfig() (the implicit default if nothing is passed)
    must not silently enable dry-run; --dry-run must be explicit."""
    config = run.ExecutorConfig()
    assert config.dry_run is False
    assert config.paper is True
    assert config.live is False
