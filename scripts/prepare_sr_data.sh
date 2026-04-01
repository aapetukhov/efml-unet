#!/usr/bin/env bash
set -euo pipefail

echo "DIV2K dataset layout is expected to be pre-downloaded and unpacked as:"
echo "  ./data/div2k/hr/train/DIV2K_train_HR"
echo "  ./data/div2k/hr/val/DIV2K_valid_HR"
echo "  ./data/div2k/lr/x4/train/DIV2K_train_LR_bicubic/X4"
echo "  ./data/div2k/lr/x4/val/DIV2K_valid_LR_bicubic/X4"
echo "  ./data/div2k/lr/x8/train/DIV2K_train_LR_x8"
echo "  ./data/div2k/lr/x8/val/DIV2K_valid_LR_x8"
