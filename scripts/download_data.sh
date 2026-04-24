#!/usr/bin/env bash
# Download DIV2K dataset to data/raw/
# Usage: bash scripts/download_data.sh
set -euo pipefail

mkdir -p data/raw

echo "==> Downloading DIV2K_train_HR (~3.3 GB)..."
wget -c -P data/raw "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
unzip -q data/raw/DIV2K_train_HR.zip -d data/raw/
rm data/raw/DIV2K_train_HR.zip

echo "==> Downloading DIV2K_valid_HR (~440 MB)..."
wget -c -P data/raw "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
unzip -q data/raw/DIV2K_valid_HR.zip -d data/raw/
rm data/raw/DIV2K_valid_HR.zip

echo "==> Done. Run scripts/prepare_data.sh to create train/val/test splits."
