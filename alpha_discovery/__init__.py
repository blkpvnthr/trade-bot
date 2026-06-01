"""alpha_discovery — realtime feature engineering + hidden-alpha discovery.

Builders that transform normalized Alpha Vantage realtime data (price,
technical, options, news) into schema-controlled feature columns for the alpha
discovery layer and the RL options models. Every builder is advisory-safe:
missing inputs degrade to neutral feature values, never an exception.

Modules
-------
    price_feature_builder      realtime price / quote / intraday features
    technical_feature_builder  realtime technical-indicator features
    options_feature_builder    realtime options chain / flow / sentiment features
    news_feature_builder       windowed news + catalyst + interaction features
    news_embedding_engine      compressed news text embeddings (optional NLP)
    news_price_reaction_model  validate news features vs forward returns
    latent_feature_discovery   PCA/latent compression of realtime feature blocks
    feature_store              persist + load realtime feature snapshots
    rl_feature_exporter        export validated features into the RL schema
    live_feature_transformer   merge realtime features into the live event dict
"""

from __future__ import annotations

__all__ = [
    "build_news_features",
    "build_price_features",
    "build_technical_features",
    "build_options_features",
]
