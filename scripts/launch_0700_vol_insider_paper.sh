#!/usr/bin/env bash
set -euo pipefail

# 7:00 AM paper launch wrapper for the only validated live candidate:
# vol-gated Form-4 insider-buy SHARES. This does not bypass run.py or RiskManager.

cd "$(dirname "$0")/.."

export APCA_PAPER=true
export ENABLE_INSIDER_COPIER=1
export INSIDER_COPY_INSTRUMENT=share
export INSIDER_COPY_VOL_GATE=true
export INSIDER_COPY_VOL_MIN_RV20="${INSIDER_COPY_VOL_MIN_RV20:-0.0563}"
export INSIDER_COPY_MIN_ADV="${INSIDER_COPY_MIN_ADV:-1000000}"
export INSIDER_COPY_SHARE_HOLD_DAYS="${INSIDER_COPY_SHARE_HOLD_DAYS:-3}"
export INSIDER_COPY_SHARE_RISK_PCT="${INSIDER_COPY_SHARE_RISK_PCT:-0.01}"
export INSIDER_COPY_SHARE_STOP_PCT="${INSIDER_COPY_SHARE_STOP_PCT:-0.10}"
export INSIDER_COPY_MAX_OPEN="${INSIDER_COPY_MAX_OPEN:-3}"
export INSIDER_COPY_PF_KILL=true
export INSIDER_COPY_PF_THRESHOLD="${INSIDER_COPY_PF_THRESHOLD:-1.8}"

python run.py
