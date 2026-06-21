#!/usr/bin/env bash
# Build the EO-backed symbolic alpha miner. Requires EO 1.3.1 installed locally.
# Override the EO prefix with EO_HOME (default: ~/Downloads/EO-1.3.1/local).
set -euo pipefail

EO_HOME="${EO_HOME:-$HOME/Downloads/EO-1.3.1/local}"
here="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -d "$EO_HOME/include/eo" ]]; then
  echo "EO not found at $EO_HOME (set EO_HOME). Skipping C++ build; Python backend still works." >&2
  exit 0
fi

c++ -std=c++14 -O2 \
  -I"$EO_HOME/include/eo" \
  "$here/eo_miner.cpp" \
  "$EO_HOME/lib/libeo.a" "$EO_HOME/lib/libeoutils.a" \
  -o "$here/eo_miner"

echo "built $here/eo_miner"
