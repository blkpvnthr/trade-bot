# Kronos — Stage 0 Audit (PAUSED after feasibility, per user priority shift)

Repo: https://github.com/shiyu-coder/Kronos · License: **MIT** · AAAI 2026 · arXiv 2508.02739.
Status: **FEASIBLE — verified to load + run locally.** Paused at the user's request to prioritize the
legacy-Transformer robustness gate. Resume here when the signal work concludes.

## Findings
- **Architecture:** decoder-only foundation model for K-lines, two-stage: (1) `KronosTokenizer`
  (hierarchical BSQ quantization of OHLCV → discrete s1/s2 tokens), (2) `Kronos` autoregressive
  transformer over tokens. Pretrained on 45+ global exchanges.
- **Input schema:** columns `open, high, low, close, volume, amount` (`amount` auto-filled =
  volume×mean-price if missing); per-window z-score + clip(5) normalization; timestamps →
  `[minute, hour, weekday, day, month]` features. Daily bars work (intraday time-fields degrade gracefully).
- **Timeframes:** any K-line. Context length: Kronos-small/Tokenizer-base = 512; Kronos-mini/
  Tokenizer-2k = 2048.
- **Pretrained (HF `NeoQuasar`):** Kronos-mini (4.1M, ctx 2048), Kronos-small (24.7M, ctx 512),
  Kronos-base (larger). Loaded via `from_pretrained` (PyTorchModelHubMixin).
- **Inference API:** `KronosPredictor(model, tokenizer, device, max_context, clip).predict(df,
  x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count)` → forecast DataFrame
  (OHLCV+amount). `predict_batch` for parallel multi-series. `generate`/`auto_regressive_inference`
  lower-level.
- **Forecast output:** **probabilistic** — autoregressive sampling (`top_k`/`top_p`/`T`),
  `sample_count` paths. NOTE: `auto_regressive_inference` **averages the sample paths internally**
  (`np.mean(preds, axis=1)`), so distribution/uncertainty features require a custom path that grabs
  the per-sample tensor BEFORE averaging (reshape `(-1, sample_count, pred_len, 6)`). Doable.
- **Embeddings:** `KronosTokenizer.encode(x)` → discrete (s1, s2) tokens; transformer hidden states
  available via `model.decode_s1(...) -> (logits, context)`. No packaged embedding API, but extractable.
- **Compute:** auto cuda/mps/cpu. **VERIFIED on this Mac (MPS):** Kronos-mini + Tokenizer-2k load
  2.2s; 5-path 5-day forecast on adjusted AAPL 3.3s. Deps: torch≥2.0, einops, huggingface_hub,
  safetensors (all installed).
- **Feasibility gate: PASS** — weights download from HF and inference runs locally.

## Compute caveat for the planned experiment
Per-(symbol, date) frozen forecasting is heavy (~3s/symbol/window). A full daily export over 74
symbols × 2628 dates is ~100+ hrs. The one-shot must scope: focused universe + `predict_batch` +
modest `sample_count` + test-window dates only (or weekly). To estimate before committing.

## Next when resumed (Stage 1+)
Frozen forecast features (`kron_pred_ret_1d/5d`, `kron_up_prob`, `kron_pred_dispersion`) on the
research backfill, chronological/no-leakage → A/B vs the champion → information diagnostics → full
harness → keep/reject. Treat as guilty until it shows incremental, orthogonal information.
