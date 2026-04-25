#!/usr/bin/env bash
# Create train/val/test splits from raw DIV2K
# Expects data/raw/DIV2K_train_HR/ (0001-0800) and data/raw/DIV2K_valid_HR/ (0801-0900).
# Usage: bash scripts/prepare_data.sh
set -euo pipefail

TRAIN_RAW="$(realpath data/raw/DIV2K_train_HR)"
VALID_RAW="$(realpath data/raw/DIV2K_valid_HR)"

for dir in data/sr/train_hr data/sr/val_hr data/sr/test_hr; do
  mkdir -p "$dir"
  find "$dir" -type l -delete  # remove stale symlinks before re-creating
done

echo "==> Linking train (0001-0800)..."
for f in "$TRAIN_RAW"/0{001..800}.png; do
  [ -f "$f" ] && ln -sf "$f" data/sr/train_hr/
done

echo "==> Linking val (0801-0850)..."
for f in "$VALID_RAW"/08{01..50}.png; do
  [ -f "$f" ] && ln -sf "$f" data/sr/val_hr/
done

echo "==> Linking test (0851-0900)..."
for f in "$VALID_RAW"/08{51..99}.png "$VALID_RAW"/0900.png; do
  [ -f "$f" ] && ln -sf "$f" data/sr/test_hr/
done

echo "train: $(ls data/sr/train_hr | wc -l) images"
echo "val:   $(ls data/sr/val_hr   | wc -l) images"
echo "test:  $(ls data/sr/test_hr  | wc -l) images"
