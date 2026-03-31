#!/usr/bin/env bash
set -euo pipefail

mkdir -p ./data/sr/train_hr ./data/sr/val_hr ./data/sr/test_hr

echo "Положите HR-изображения для обучения в ./data/sr/train_hr"
echo "Положите HR-изображения для валидации в ./data/sr/val_hr"
echo "Положите HR-изображения для теста в ./data/sr/test_hr"
echo "LR-входы генерируются на лету внутри датасет-пайплайна."
