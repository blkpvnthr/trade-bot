
## 13:31 | fix/13K-post-merge-integration

Replaced OpenAI with Claude (Anthropic Messages API) in `market_context_builder.py`; hardened JSON parsing with truncation detection, retries, explicit fallback to rules classifier on all failures (never fabricates); added source/timestamp tracking; lazy SDK import allows graceful degradation until ANTHROPIC_API_KEY is set.
## 13:44 | fix/13K-post-merge-integration

Hardened import (SDK-missing → rules), added SDK-missing test (10 tests, 612 passed), committed fb755bf, ran smoke test verifying fallback.