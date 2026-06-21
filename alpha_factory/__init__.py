"""alpha_factory - autonomous alpha discovery/validation engine (research-only, no trading).

Pipeline:  Miner -> Scientist -> Librarian (registry) -> Portfolio export -> RL allocator.

The Scientist does NOT reinvent significance testing. It wraps the repo's existing
rl_trading.event_study gate (forward panels + block-bootstrap CI + rank-IC + placebo control),
adds FDR (Benjamini-Hochberg across the batch) and a net-of-cost long/short sim (1-day forward
returns, per the short-vol method-fix), and returns a preregistered accept/reject/needs-more-data
verdict. Nothing here imports a broker or submits an order.
"""
from .schema import AlphaRecord
from .registry import AlphaRegistry
from .scientist import Criteria, evaluate, run_batch
from . import miner

__all__ = ["AlphaRecord", "AlphaRegistry", "Criteria", "evaluate", "run_batch", "miner"]
