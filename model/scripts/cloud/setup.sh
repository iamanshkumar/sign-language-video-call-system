#!/usr/bin/env bash
# One-time setup on a Linux x86_64 machine (glibc >= 2.28, Python 3.12). Run from the bundle root.
set -euo pipefail
[[ "$(uname -s)-$(uname -m)" == "Linux-x86_64" ]] || { echo "needs Linux x86_64 (mediapipe 0.10.21 has no Linux ARM wheel)"; exit 1; }
PY=${PYTHON:-python3.12}
$PY -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' 
$PY -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r scripts/cloud/requirements-cloud.txt
.venv/bin/pip install -q "torch==2.14.1" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -q -e .
# self-test: preprocessing, sequence and cloud tooling tests (no dataset needed)
PYTHONPATH=src .venv/bin/python -m pytest -q tests/test_landmarks.py tests/test_sequences.py tests/test_cloud.py
echo "setup OK"
