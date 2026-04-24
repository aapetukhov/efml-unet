# EFML Super-Resolution

Single-image super-resolution with a U-Net baseline, plus inference acceleration experiments.

**Team 52** (Artem Chubov, Andrey Petukhov, Evgeny Veselkov)

## Project goal

Train an SR U-Net and benchmark five inference optimization methods:
1. FP32 baseline
2. FP16 (Artem)
3. `torch.compile` (Zhenya)
4. TVM Relax/TIR (Zhenya + Andrey)
5. Quantization INT8 (Artem + Zhenya)
6. Pruning / 2:4 sparsity (Andrey)

## Setup

```bash
pip install -r requirements.txt
bash scripts/download_data.sh   # ~3.7 GB, skippable if data/raw/ already populated
bash scripts/prepare_data.sh    # creates train/val/test symlinks in data/sr/
```

## Training

```bash
# x4 (default)
python train.py

# other scales
python train.py --config-name baseline_x3
python train.py --config-name baseline_x8

# override any param
python train.py trainer.epochs=10 trainer.device=cpu
```

Default config: [src/configs/baseline.yaml](src/configs/baseline.yaml) — scale x4, AMP on, 50 epochs, DIV2K 800 train / 50 val images.

## Data

**Dataset:** [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — 2K resolution HR images.

| Split | Source | Count |
|---|---|---|
| train | DIV2K_train_HR (0001–0800) | 800 |
| val   | DIV2K_valid_HR (0801–0850) | 50  |
| test  | DIV2K_valid_HR (0851–0900) | 50  |

LR images are generated on-the-fly: HR crop → bicubic downsample → bicubic upsample to HR size.
This matches the model's same-size I/O (standard U-Net, no internal upsampling).

## Repository structure

```
src/
  configs/          Hydra configs (one top-level .yaml per scale)
  datasets/         Dataset classes + collate
  model/            SRUNet architecture
  trainer/          Trainer + Inferencer
  loss/ metrics/ transforms/ logger/ utils/ writer/
scripts/
  download_data.sh  wget DIV2K train + valid HR
  prepare_data.sh   create train/val/test symlinks
train.py            entry point (Hydra)
inference.py        eval / inference entry point
```
