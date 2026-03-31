#!/usr/bin/env bash
set -euo pipefail

python -m src.benchmark --config ./configs/sr_baseline_x2_fp16.yaml
