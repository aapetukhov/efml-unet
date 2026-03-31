#!/usr/bin/env bash
set -euo pipefail

python -m src.train --config ./configs/sr_baseline_x2.yaml
