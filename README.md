# EFML Super-Resolution — SRUNetHeavy

**Team 52** (Artem Chubov, Andrey Petukhov, Evgeny Veselkov)

## Setup

```bash
pip install -r requirements.txt
bash scripts/download_data.sh   # ~3.7 GB, skippable if data/raw/ already populated
bash scripts/prepare_data.sh    # creates train/val/test symlinks in data/sr/
```

## Training

```bash
python train.py                              # x4, default config
python train.py --config-name baseline_x4_heavy   # SRUNetHeavy
python train.py trainer.epochs=10 trainer.device=cpu
```

## Data

**Dataset:** [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) — 2K resolution HR images.

| Split | Source | Count |
|---|---|---|
| train | DIV2K_train_HR (0001–0800) | 800 |
| val   | DIV2K_valid_HR (0801–0850) | 50  |
| test  | DIV2K_valid_HR (0851–0900) | 50  |

LR images are generated on-the-fly: HR crop → bicubic downsample → bicubic upsample to HR size.

## Repository structure

```
src/
  configs/          Hydra configs
  datasets/         Dataset classes + collate
  model/            SRUNet / SRUNetHeavy architecture
  trainer/          Trainer + Inferencer
  loss/ metrics/ transforms/ logger/ utils/ writer/
  acceleration/     pruning, sparsity, group conv, quantization, benchmark utils
scripts/
  download_data.sh            wget DIV2K train + valid HR
  prepare_data.sh             create train/val/test symlinks
  run_prune.py                global magnitude pruning
  run_prune_structured.py     structural expand-channel pruning
  run_sparse.py               2:4 semi-structured sparsity
  run_group_conv.py           grouped 1×1 convolutions
  run_quantize.py             LSQ PTQ/QAT quantization
  benchmark_quant_batches.py  batch size sweep for quantized model
  animations/                 visualizations
train.py            entry point (Hydra)
inference.py        eval / inference entry point
```

---

## Model: SRUNetHeavy

Checkpoint: `checkpoints/heavy_srunet_29M.pt` · **29M params** · scale ×4 · input/output **256×256**

```
stem (Conv3x3 → GroupNorm → GELU)
  │
  inc (MBResBlock bc→bc)
  │
  down1–4 (MaxPool2x2 → MBResBlock)    bc → 2bc → 4bc → 8bc → 8bc
  │
  bottleneck (MBResBlock + SEBlock)
  │
  up1–4 (bilinear×2 → cat(skip) → MBResBlock)
  │
  head (Conv1x1)
  + global residual
```

**MBResBlock** (inverted residual): `GroupNorm+GELU+Conv1x1 (expand×4)` → `GroupNorm+GELU+DWConv3x3` → `Conv1x1+GroupNorm (project)` + skip. GroupNorm throughout — no BatchNorm, so channel removal doesn't break running statistics.

**SEBlock**: global avg pool → Linear(C → C//4) → ReLU → Linear(C//4 → C) → Sigmoid → channel-wise scale.

**Channels** (bc = 96): inc=96, down1=192, down2=384, down3=768, down4=768, bottleneck=768, mirror in decoder.

---

## Andrey: Acceleration experiments

Model: `checkpoints/heavy_srunet_29M.pt` (29M params, SRUNetHeavy, scale ×4).

**Latency measurement** (`src/acceleration/benchmark.py:measure_latency`): synthetic `torch.randn`, 20 warmup runs, then 200 timed runs via CUDA Events. Speedup = `baseline_latency / result_latency` on identical input size.

### Methods and results

**1. Global magnitude pruning** (`run_prune.py`, results: `results/andrey-results/prune_*.json`)

Zeroes the N% smallest-magnitude weights globally across all 1×1 Conv2d layers via `torch.nn.utils.prune.global_unstructured`. Tensors stay dense — no real speedup (≈1.0×) at any ratio. SSIM degrades from 0.7689 to 0.67 at 100%.

**2. Structural expand-channel pruning** (`run_prune_structured.py`, results: `results/andrey-results/struct_*.json`)

Physically removes the lowest-L1-norm output channels from the expand 1×1 conv in each MBResBlock. Only `mid_ch` shrinks — `in_ch`, `out_ch`, and skip connections are untouched, U-Net topology stays valid. `mid_ch` rounded to a multiple of `num_groups` (GroupNorm constraint).

| Prune | Speedup BS=1 | Speedup BS=32 | SSIM (after ft) | SSIM drop |
|-------|-------------|--------------|-----------------|-----------|
| 25%   | 1.23×       | 1.23×        | 0.7598          | −0.009    |
| 50%   | 1.57×       | 1.68×        | 0.7581          | —         |
| 75%   | 2.27×       | 2.53×        | 0.7598          | −0.009    |

Best result: **75% + 10 epochs finetune** — 2.27× speedup, SSIM drop only 0.009, model shrinks from 29.3M to 8.4M params.

**3. Grouped 1×1 convolutions** (`run_group_conv.py`, results: `results/andrey-results/group*.json`)

Replaces expand/project 1×1 Conv2d with grouped conv (G=2 or G=4). Init: block-diagonal slice of original weight matrix. FLOPs ÷ G, but actual speedup limited by CUDA matmul overhead.

Visualization: [grouped sparsification animation](scripts/animations/grouped_sparsification.webm) · [Google Drive](https://drive.google.com/file/d/11Wazp6hj3aZ3Mg-O0sREZ30C03yOJpsc/view?usp=share_link)

| Config     | Speedup | SSIM drop |
|------------|---------|-----------|
| G=2 expand | 1.03×   | −0.016    |
| G=2 both   | 1.07×   | −0.021    |
| G=4 both   | 1.13×   | −0.035    |

Worse tradeoff than structural pruning.

**4. 2:4 semi-structured sparsity** (`run_sparse.py`, results: `results/andrey-results/sparse_244*.json`)

In every group of 4 consecutive weights exactly 2 are zeroed (smallest magnitude). `to_sparse_semi_structured` + `Pointwise2d` wrapper routes through `F.linear` to trigger cuSPARSELt. **Ampere+ only** (sm_80+), FP16/BF16 only. RTX 8000 = Turing (sm_75) → falls back to dense masking, no speedup.

A100 results (FP16):

| BS | Baseline | Sparse  | Speedup |
|----|----------|---------|---------|
| 1  | 26.0 ms  | 31.1 ms | 0.84×   |
| 32 | 155 ms   | 411 ms  | 0.38×   |

No speedup at any tested batch size — matrices too small for cuSPARSELt overhead to amortise.

### Running experiments

```bash
# structural pruning 75%, 10-epoch finetune
python scripts/run_prune_structured.py prune_ratio=0.75 finetune.epochs=10 results_name=struct_75_ft

# latency at bs=32
python scripts/run_prune_structured.py prune_ratio=0.75 finetune.enabled=false benchmark.input_size=[32,3,256,256] results_name=struct_75_bs32

# 2:4 sparse (Ampere, fp16)
python scripts/run_sparse.py fp16=true convert_to_sparse=true results_name=sparse_244_a100_fp16

# group conv G=4
python scripts/run_group_conv.py groups=4 target=both finetune.epochs=10 results_name=group4_both
```

---

## Zhenya: Acceleration experiments

**Latency measurement** (`run_all_benchmarks.py`): synthetic `torch.randn` input, BS=1, 10 warmup + 50 timed runs via CUDA Events. Hardware: NVIDIA GeForce RTX 3060 (12 GB, 360 GB/s). Speedup = `FP32 eager latency / result latency`. Full tables: `results/zhenya-results/comparison_table.md`.

### SRUNetHeavy 29M — LR 256² → LR 256²

`torch.compile` and ORT give minimal gain — Triton and ORT lack fused kernels for the GroupNorm+GELU+DWConv sequences that TRT encodes via custom plugins.

| Method | Precision | Latency | Throughput | Speedup |
|---|---|---:|---:|---:|
| Eager PyTorch | FP32 | 48.1 ms | 20.8 img/s | 1.00× |
| Eager PyTorch | FP16 | 31.7 ms | 31.6 img/s | 1.52× |
| `torch.compile` (max-autotune) | FP32 | 46.6 ms | 21.5 img/s | 1.03× |
| ORT CUDA EP | FP32 | 46.1 ms | 21.6 img/s | 1.04× |
| **ORT TensorRT EP** | **FP32** | **29.8 ms** | **33.6 img/s** | **1.61×** |
| **ORT TensorRT EP** | **FP16** | **13.3 ms** | **75.2 img/s** | **3.62×** |

TRT FP32 ×1.61: fusion of GroupNorm+GELU+Conv1×1 sequences reduces kernel-launch overhead, more impactful for shallow 1×1 layers than deep 3×3. TRT FP16 ×3.62: 1×1 pointwise convolutions are GEMMs and use tensor cores, partially compensating for DWConv's TC-incompatibility.

### Roofline analysis (SRUNetHeavy)

138 GFLOPs, ~657 MB DRAM traffic. AI = 210 FLOPs/byte → **compute-bound** in FP32, **compute-bound** in FP16 TC (AI=420 > ridge=283). Despite high AI, eager FP32 achieves only 22% of peak — DWConv processes each channel independently, preventing warp batching and blocking tensor core use. Full derivation: `results/zhenya-results/roofline_calculations.md`.

### Running experiments

```bash
python run_all_benchmarks.py --checkpoint checkpoints/heavy_srunet_29M.pt --scale 4 --fp16 --no-tvm
```

---

## Artem: Quantization (LSQ PTQ)

### What was done

Quantization pipeline for **SRUNetHeavy** via **LSQ** in **Post-Training Quantization (PTQ)** mode.

**Implementation (`src/acceleration/`):**
- `lsq.py` — LSQ core: `LsqConv2d`, `LsqLinear`, `LsqQuantizer` with per-channel step size for weights; `finalize_lsq()` exports INT8 state dict with FP16 scales
- `quantize.py` — PTQ/QAT workflow: `apply_lsq_ptq()` for calibration on a calibration subset, `prepare_lsq_qat()` for QAT fine-tuning

**Scripts (`scripts/`):**
- `run_quantize.py` — run PTQ/QAT and save quantized checkpoint
- `benchmark_quant_batches.py` — batch size sweep [1, 4, 16, 64], comparing FP32 / FP16 / LSQ+FP32 / LSQ+FP16, e2e throughput and Chrome profiler traces

### Method

LSQ uses **fake quantization**: weights stay in float, but the forward pass simulates quantization via `round(w / s) * s`, where `s` is a learnable per-channel step size. Gradients through `round` use STE (Straight-Through Estimator).

In PTQ mode, step sizes `s` are initialized from weight statistics and frozen after calibration — no fine-tuning required.

### Results on A100-SXM4-80GB (256×256, 100 runs)

| Config | SSIM | Size | BS=1 | BS=4 | BS=16 | BS=64 | Speedup vs FP16 (BS=64) |
|---|---|---|---:|---:|---:|---:|---:|
| FP32 | 0.7689 | 111.7 MB | 30.2 ms | 51.0 ms | 128.9 ms | 485.8 ms | 0.63× |
| **FP16** | **0.7689** | **55.8 MB** | **26.1 ms** | **41.0 ms** | **89.3 ms** | **306.9 ms** | **1.00×** |
| LSQ + FP32 | 0.7681 | 28.1 MB | 41.3 ms | 61.5 ms | 139.5 ms | 496.0 ms | 0.62× |
| LSQ + FP16 | 0.7680 | 28.1 MB | 37.7 ms | 51.9 ms | 99.9 ms | 316.8 ms | 0.97× |

### Key findings

1. **4× model compression without fine-tuning.** INT8 state dict: 28.1 MB vs 111.7 MB (FP32). Reduces memory footprint and load time.

2. **Quality preserved.** LSQ+FP16 SSIM = 0.7680 vs 0.7689 (FP32) — PTQ without fine-tuning gives acceptable quality.

3. **LSQ+FP16 matches FP16 throughput.** At BS=64 speedup is 0.97× vs FP16, while using 2× less memory (28.1 MB vs 55.8 MB).

### Running experiments

```bash
# PTQ calibration + save checkpoint
python scripts/run_quantize.py mode=ptq checkpoint_out=checkpoints/lsq_ptq.pt

# batch size sweep benchmark
python scripts/benchmark_quant_batches.py --checkpoint checkpoints/lsq_ptq.pt
```
