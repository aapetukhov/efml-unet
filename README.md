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
  run_group_conv.py         grouped 1x1 convolutions
train.py            entry point (Hydra)
inference.py        eval / inference entry point
```

---

## Model architecture — SRUNetHeavy

Checkpoint: `checkpoints/heavy_srunet_29M.pt`, 29M params, scale x4.

```
stem (Conv3x3 → GroupNorm → GELU)
  │
  inc (MBResBlock bc→bc)
  │
  down1–4 (MaxPool2x2 → MBResBlock)    channels: bc → bc*2 → bc*4 → bc*8 → bc*8
  │
  bottleneck (MBResBlock + SEBlock)
  │
  up1–4 (bilinearx2 → cat(skip) → MBResBlock)
  │
  head (Conv1x1)
  │
  + global residual (input added to output)
```

**MBResBlock** (inverted residual): `GroupNorm+GELU+Conv1x1 (expand)` → `GroupNorm+GELU+DWConv3x3` → `Conv1x1+GroupNorm (project)` + skip. Expand ratio = 4, so mid_ch = in_ch x 4. GroupNorm used throughout — no BatchNorm, so channel removal doesn't break running statistics.

**SEBlock**: global avg pool → Linear(C → C//4) → ReLU → Linear(C//4 → C) → Sigmoid → channel-wise scale.

**Channels** (bc = base_channels = 96): inc=96, down1=192, down2=384, down3=768, down4=768, bottleneck=768, then mirror in decoder.

**Initialization**: PyTorch defaults — Conv2d and Linear use Kaiming uniform (He init), GroupNorm weight=1 bias=0. No custom `_init_weights`.

---

## Andrey: Acceleration experiments

Model: `checkpoints/heavy_srunet_29M.pt` (29M params, SRUNetHeavy, scale x4).

**Latency measurement** (`src/acceleration/benchmark.py:measure_latency`): synthetic input `torch.randn`, 20 warmup runs outside the timer, then 200 timed runs via CUDA Events (`start.record() / end.record() / synchronize()`), result = `elapsed_time / 200` ms. CUDA Events measure GPU time directly, without CPU overhead. Speedup = `baseline_latency / result_latency` — same function called before and after model modification on identical input size.

### Methods and results

**1. Global magnitude pruning** (`run_prune.py`, results: `results/andrey-results/prune_*.json`)

Zeroes the N% smallest-magnitude weights globally across all 1x1 Conv2d layers using `torch.nn.utils.prune.global_unstructured`. Tensors stay dense — no real speedup (≈1.0x) at any ratio. SSIM degrades from 0.7689 to 0.67 at 100% ratio.

**2. Structural expand-channel pruning** (`run_prune_structured.py`, results: `struct_*.json`)

Physically removes the lowest-L1-norm output channels from the expand 1x1 conv in each MBResBlock. Only `mid_ch` shrinks — `in_ch`, `out_ch`, and skip connections are untouched, so U-Net topology stays valid. New `mid_ch` is rounded to a multiple of `num_groups` (GroupNorm constraint). After pruning, the block is re-assembled with new smaller Conv2d tensors.

| prune | speedup bs=1 | speedup bs=32 | SSIM (after ft) | SSIM drop |
|-------|-------------|--------------|-----------------|-----------|
| 25%   | 1.23x       | 1.23x        | 0.7598          | −0.009    |
| 50%   | 1.57x       | 1.68x        | 0.7581 (no ft val) | —     |
| 75%   | 2.27x       | 2.53x        | 0.7598          | −0.009    |

Best result: **75% + 10 epochs finetune** — 2.27x speedup, SSIM drop only 0.009, model shrinks from 29.3M to 8.4M params.

**3. Grouped 1x1 convolutions** (`run_group_conv.py`, results: `group*.json`)

Replaces expand/project 1x1 Conv2d with grouped conv (G=2 or G=4). Init: block-diagonal slice of original weight matrix, so finetune starts close to the original function. FLOPs in targeted layers ÷ G, but actual speedup is limited because CUDA matrix multiply overhead doesn't scale linearly with G.

| config      | speedup | SSIM drop |
|-------------|---------|-----------|
| G=2 expand  | 1.03x   | −0.016    |
| G=2 both    | 1.07x   | −0.021    |
| G=4 both    | 1.13x   | −0.035    |

Worse tradeoff than structural pruning.

**4. 2:4 semi-structured sparsity** (`run_sparse.py`, results: `sparse_244*.json`)

In every group of 4 consecutive weights exactly 2 are zeroed (smallest magnitude). `to_sparse_semi_structured` + `Pointwise2d` wrapper routes through `F.linear` to trigger cuSPARSELt.

Hardware constraints: **Ampere+ only** (sm_80+), **FP16/BF16 only**, **F.linear only** (not F.conv2d).  
RTX 8000 = Turing (sm_75) — falls back to dense masking, no speedup.

A100 results (FP16):

| bs   | baseline | sparse  | speedup |
|------|----------|---------|---------|
| 1    | 26.0 ms  | 31.1 ms | 0.84x   |
| 32   | 155 ms   | 411 ms  | 0.38x   |

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

---

## Zhenya: Acceleration experiments

Model: `checkpoints/unet_sr_x4_baseline.pt` (8.23M params, UNetSR baseline, scale x4).

**Latency measurement** (`run_all_benchmarks.py`): synthetic `torch.randn` input, BS=1, 256×256 LR → 1024×1024 HR, 10 warmup runs, then 50 timed runs via CUDA Events. Hardware: NVIDIA GeForce RTX 3060 (12 GB, 360 GB/s). Speedup = `FP32 eager latency / result latency`.

### Methods and results

**1. FP16** (results: `results/zhenya-results/comparison_table.md`)

`.half()` cast on model and input — no structural changes. Compute throughput doubles (12.74 → 25.48 TFLOPS), DRAM traffic halved.

| Precision | Latency | Throughput | Speedup |
|---|---:|---:|---:|
| FP32 | 125 ms | 8.0 img/s | 1.00x |
| FP16 | 81.5 ms | 12.3 img/s | 1.53x |

**2. `torch.compile`** (results: `results/zhenya-results/comparison_table.md`)

Inductor backend compiles to Triton kernels. Both `default` and `reduce-overhead` modes are slower than eager — Triton-generated kernels cannot match cuDNN for standard 3×3 convolutions.

| Mode | Latency | Throughput | Speedup |
|---|---:|---:|---:|
| default | 237 ms | 4.2 img/s | 0.53x |
| reduce-overhead | 233 ms | 4.3 img/s | 0.54x |

**3. ONNX Runtime** (results: `results/zhenya-results/ort_benchmark_x4.json`)

Model exported to ONNX, run via two ORT execution providers. CUDA EP applies standard op fusion, no autotuning — essentially the same speed as eager. TensorRT EP applies layer fusion, cuDNN kernel autotuning, and in FP16 routes through tensor cores.

| EP | Precision | Latency | Throughput | Speedup |
|---|---|---:|---:|---:|
| CUDA EP | FP32 | 130 ms | 7.7 img/s | 0.97x |
| TensorRT EP | FP32 | 97 ms | 10.3 img/s | 1.40x |
| **TensorRT EP** | **FP16** | **34 ms** | **29.7 img/s** | **4.02x** |

Best result: **TensorRT EP FP16** — 4.02x speedup, identical PSNR/SSIM (23.497 / 0.6918).

**4. TVM** (results: `results/zhenya-results/comparison_table.md`)

Two schedule strategies tested. DLight (rule-based, no autotuning) falls back to generic schedules on 3×3 conv and runs ~4x slower than eager. MetaSchedule (evolutionary search) improves with more trials but plateaus below cuDNN.

| Schedule | Trials | Tuning time | Latency | Speedup |
|---|---:|---:|---:|---:|
| DLight (no tuning) | — | — | 511 ms | 0.24x |
| MetaSchedule replay-func | 512 | 37.5 min | 205 ms | 0.61x |
| MetaSchedule replay-func | 1024 | 72 min | 152 ms | 0.83x |

Each doubling of trials gives ~25% latency reduction; 1024 trials reaches 0.83x of PyTorch eager — close but still below cuDNN.

### Roofline analysis

UNetSR baseline: 806 GFLOPs, ~8892 MB DRAM traffic (FP32, upper bound). Arithmetic intensity = 90.7 FLOPs/byte; RTX 3060 FP32 ridge point = 35.4 FLOPs/byte → model is **compute-bound** in FP32 and FP16 (no tensor cores), **memory-bound** relative to FP16 tensor core ceiling. Eager FP32 achieves ~6450 GFLOPs/s (~51% GPU utilisation). Full derivation: `results/zhenya-results/roofline_calculations.md`.

### Running experiments

```bash
python run_all_benchmarks.py --checkpoint checkpoints/unet_sr_x4_baseline.pt --scale 4 --fp16
# add --no-tvm to skip TVM (slow to tune)
```
