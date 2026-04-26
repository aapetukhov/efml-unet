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
  acceleration/     pruning, sparsity, group conv, benchmark utils
scripts/
  download_data.sh          wget DIV2K train + valid HR
  prepare_data.sh           create train/val/test symlinks
  run_prune.py              global magnitude pruning
  run_prune_structured.py   structural expand-channel pruning
  run_sparse.py             2:4 semi-structured sparsity
  run_group_conv.py         grouped 1×1 convolutions
train.py            entry point (Hydra)
inference.py        eval / inference entry point
```

---

## Model architecture — SRUNetHeavy

Checkpoint: `checkpoints/heavy_srunet_29M.pt`, 29M params, scale ×4.

```
stem (Conv3×3 → GroupNorm → GELU)
  │
  inc (MBResBlock bc→bc)
  │
  down1–4 (MaxPool2×2 → MBResBlock)    channels: bc → bc*2 → bc*4 → bc*8 → bc*8
  │
  bottleneck (MBResBlock + SEBlock)
  │
  up1–4 (bilinear×2 → cat(skip) → MBResBlock)
  │
  head (Conv1×1)
  │
  + global residual (input added to output)
```

**MBResBlock** (inverted residual): `GroupNorm+GELU+Conv1×1 (expand)` → `GroupNorm+GELU+DWConv3×3` → `Conv1×1+GroupNorm (project)` + skip. Expand ratio = 4, so mid_ch = in_ch × 4. GroupNorm used throughout — no BatchNorm, so channel removal doesn't break running statistics.

**SEBlock**: global avg pool → Linear(C → C//4) → ReLU → Linear(C//4 → C) → Sigmoid → channel-wise scale.

**Channels** (bc = base_channels = 96): inc=96, down1=192, down2=384, down3=768, down4=768, bottleneck=768, then mirror in decoder.

**Initialization**: PyTorch defaults — Conv2d and Linear use Kaiming uniform (He init), GroupNorm weight=1 bias=0. No custom `_init_weights`.

---

## Andrey: Acceleration experiments

Model: `checkpoints/heavy_srunet_29M.pt` (29M params, SRUNetHeavy, scale ×4).

**Latency measurement** (`src/acceleration/benchmark.py:measure_latency`): synthetic input `torch.randn`, 20 warmup runs outside the timer, then 200 timed runs via CUDA Events (`start.record() / end.record() / synchronize()`), result = `elapsed_time / 200` ms. CUDA Events measure GPU time directly, without CPU overhead. Speedup = `baseline_latency / result_latency` — same function called before and after model modification on identical input size.

### Methods and results

**1. Global magnitude pruning** (`run_prune.py`, results: `results/andrey-results/prune_*.json`)

Zeroes the N% smallest-magnitude weights globally across all 1×1 Conv2d layers using `torch.nn.utils.prune.global_unstructured`. Tensors stay dense — no real speedup (≈1.0×) at any ratio. SSIM degrades from 0.7689 to 0.67 at 100% ratio.

**2. Structural expand-channel pruning** (`run_prune_structured.py`, results: `struct_*.json`)

Physically removes the lowest-L1-norm output channels from the expand 1×1 conv in each MBResBlock. Only `mid_ch` shrinks — `in_ch`, `out_ch`, and skip connections are untouched, so U-Net topology stays valid. New `mid_ch` is rounded to a multiple of `num_groups` (GroupNorm constraint). After pruning, the block is re-assembled with new smaller Conv2d tensors.

| prune | speedup bs=1 | speedup bs=32 | SSIM (after ft) | SSIM drop |
|-------|-------------|--------------|-----------------|-----------|
| 25%   | 1.23×       | 1.23×        | 0.7598          | −0.009    |
| 50%   | 1.57×       | 1.68×        | 0.7581 (no ft val) | —     |
| 75%   | 2.27×       | 2.53×        | 0.7598          | −0.009    |

Best result: **75% + 10 epochs finetune** — 2.27× speedup, SSIM drop only 0.009, model shrinks from 29.3M to 8.4M params.

**3. Grouped 1×1 convolutions** (`run_group_conv.py`, results: `group*.json`)

Replaces expand/project 1×1 Conv2d with grouped conv (G=2 or G=4). Init: block-diagonal slice of original weight matrix, so finetune starts close to the original function. FLOPs in targeted layers ÷ G, but actual speedup is limited because CUDA matrix multiply overhead doesn't scale linearly with G.

| config      | speedup | SSIM drop |
|-------------|---------|-----------|
| G=2 expand  | 1.03×   | −0.016    |
| G=2 both    | 1.07×   | −0.021    |
| G=4 both    | 1.13×   | −0.035    |

Worse tradeoff than structural pruning.

**4. 2:4 semi-structured sparsity** (`run_sparse.py`, results: `sparse_244*.json`)

In every group of 4 consecutive weights exactly 2 are zeroed (smallest magnitude). `to_sparse_semi_structured` + `Pointwise2d` wrapper routes through `F.linear` to trigger cuSPARSELt.

Hardware constraints: **Ampere+ only** (sm_80+), **FP16/BF16 only**, **F.linear only** (not F.conv2d).  
RTX 8000 = Turing (sm_75) — falls back to dense masking, no speedup.

A100 results (FP16):

| bs   | baseline | sparse  | speedup |
|------|----------|---------|---------|
| 1    | 26.0 ms  | 31.1 ms | 0.84×   |
| 32   | 155 ms   | 411 ms  | 0.38×   |

No speedup at any tested batch size. Hypothesis: matrices too small for cuSPARSELt overhead to amortise — expected gain at bs ≥ 256.

### Running experiments

```bash
# structural pruning 75%, 10-epoch finetune
python scripts/run_prune_structured.py prune_ratio=0.75 finetune.epochs=10 results_name=struct_75_ft checkpoint_out=checkpoints/struct_75_ft.pt

# structural, latency at bs=32
python scripts/run_prune_structured.py prune_ratio=0.75 finetune.enabled=false benchmark.input_size=[32,3,256,256] results_name=struct_75_bs32

# 2:4 sparse on Ampere (requires fp16=true, convert_to_sparse=true)
python scripts/run_sparse.py fp16=true convert_to_sparse=true results_name=sparse_244_a100_fp16

# group conv G=4 both, 10 epochs
python scripts/run_group_conv.py groups=4 target=both finetune.epochs=10 results_name=group4_both
```
