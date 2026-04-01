#!/usr/bin/env bash
set -euo pipefail

python -m src.eval_bicubic --config ./configs/sr_baseline_x2.yaml
