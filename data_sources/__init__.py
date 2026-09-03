"""data_sources — external market-data clients for QuantumQuant.

Currently exposes the unified Alpha Vantage realtime client and the
NEWS_SENTIMENT client. All clients are advisory-safe: any failure degrades to a
neutral / empty result and is logged, never raised, so the trading system keeps
running.

Security: API keys are read ONLY from the environment (.env via
config.env_loader). Never hardcode keys; never log keys.
"""

from __future__ import annotations

__all__ = [
    "AlphaVantageClient",
    "AlphaVantageConfig",
    "get_default_av_client",
    "AlphaVantageAnalyticsClient",
    "AnalyticsConfig",
    "get_default_analytics_client",
]

from data_sources.alpha_vantage_client import (  # noqa: E402
    AlphaVantageClient,
    AlphaVantageConfig,
    get_default_av_client,
)
from data_sources.alpha_vantage_analytics_client import (  # noqa: E402
    AlphaVantageAnalyticsClient,
    AnalyticsConfig,
    get_default_analytics_client,
)
