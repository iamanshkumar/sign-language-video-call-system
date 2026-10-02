#!/usr/bin/env bash
# Build the portable bundle (code, configs, frozen split, label mapping, run plan, parity reference).
# Usage (on the M2): scripts/cloud/make_bundle.sh <run_name>   -> dist/<run_name>_bundle.tar.gz (+ .sha256)
set -euo pipefail
RUN=$1
[[ -f data/landmarks/runs/$RUN/plan.json ]] || { echo "plan missing: run s2_07 first"; exit 1; }
mkdir -p dist
tar -czf "dist/${RUN}_bundle.tar.gz" --exclude='__pycache__' \
  pyproject.toml requirements.txt src scripts tests configs docs \
  data/splits data/sequences/label_mapping.json data/metadata/manifest.parquet \
  data/landmarks/benchmark_v2_mixed "data/landmarks/runs/$RUN/plan.json" "data/landmarks/runs/$RUN/shards" \
  "data/landmarks/runs/$RUN/parity.csv"
shasum -a 256 "dist/${RUN}_bundle.tar.gz" | tee "dist/${RUN}_bundle.tar.gz.sha256"
