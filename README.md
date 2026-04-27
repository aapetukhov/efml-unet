# EFML UNet — Super-Resolution Benchmarks

## Models

### Baseline — UNetSR (src/modeling.py)

Checkpoint: `checkpoints/unet_sr_x4_baseline.pt`, scale ×4.

```
input_proj (Conv3x3)
  │
  enc1 (Conv3x3 + LeakyReLU + 2×ResBlock) → MaxPool
  enc2 (Conv3x3 + LeakyReLU + 2×ResBlock) → MaxPool
  bottleneck (Conv3x3 + LeakyReLU + 4×ResBlock)
  │
  dec2 (Upsample + Conv3x3 + cat(skip) + 2×ResBlock)
  dec1 (Upsample + Conv3x3 + cat(skip) + 2×ResBlock)
  │
  lr_refinement (4×ResBlock)
  upsampling_head (Conv3x3 → PixelShuffle(2) × log2(scale) steps)
  reconstruction (2×ResBlock + Conv3x3)
  │
  + bicubic residual
```

ResBlock: pre-activation style — Conv3x3 → LeakyReLU(0.2) → Conv3x3 + skip. `conv2` initialized to zero so each block starts as identity.

### Heavy — SRUNetHeavy (src/heavy_modeling.py)

Checkpoint: `checkpoints/heavy_srunet_29M.pt`, 29 M params, scale ×4.

```
stem (Conv3x3 → GroupNorm → GELU)
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

**MBResBlock** (inverted residual): GroupNorm+GELU+Conv1×1 (expand) → GroupNorm+GELU+DWConv3×3 → Conv1×1+GroupNorm (project) + skip. Expand ratio = 4, so `mid_ch = in_ch × 4`. GroupNorm throughout — no BatchNorm.

**SEBlock**: global avg pool → Linear(C → C//4) → ReLU → Linear(C//4 → C) → Sigmoid → channel-wise scale.

**Channels** (bc = base_channels = 96): inc=96, down1=192, down2=384, down3=768, down4=768, bottleneck=768, then mirror in decoder.

---

## Inference Benchmarks (src/acceleration/benchmark.py, run_all_benchmarks.py)

**Hardware:** NVIDIA GeForce RTX 3060 (12 GB, 360 GB/s)  
**Input:** 256×256 LR → 1024×1024 HR (scale ×4), BS=1  
**Latency:** CUDA Events — 10 warmup + 50 timed iterations

| Method | Precision | Latency | Throughput | vs FP32 |
|---|---|---:|---:|---:|
| Eager PyTorch | FP32 | 125 ms | 8.0 img/s | ×1.00 |
| Eager PyTorch | FP16 | 81.5 ms | 12.3 img/s | ×1.53 |
| `torch.compile(default)` | FP32 | 237 ms | 4.2 img/s | ×0.53 |
| `torch.compile(reduce-overhead)` | FP32 | 233 ms | 4.3 img/s | ×0.54 |
| ORT CUDA EP | FP32 | 130 ms | 7.7 img/s | ×0.97 |
| ORT TensorRT EP | FP32 | 97 ms | 10.3 img/s | ×1.40 |
| **ORT TensorRT EP** | **FP16** | **34 ms** | **29.7 img/s** | **×4.02** |
| TVM DLight (no tuning) | FP32 | 511 ms | 2.0 img/s | ×0.24 |
| TVM MetaSchedule (512 trials, replay-func) | FP32 | 205 ms | 4.9 img/s | ×0.61 |
| TVM MetaSchedule (1024 trials, replay-func) | FP32 | 152 ms | 6.6 img/s | ×0.83 |
| TVM MetaSchedule (2047 trials, evolutionary) | FP32 | TBD | — | — |

To reproduce: `python run_all_benchmarks.py --checkpoint checkpoints/unet_sr_x4_baseline.pt --scale 4 --fp16 --no-tvm`

---

## Andrey — Acceleration Experiments (SRUNetHeavy)

Model: `checkpoints/heavy_srunet_29M.pt` (29 M params, scale ×4).

**Latency measurement:** `src/acceleration/benchmark.py:measure_latency` — synthetic `torch.randn` input, 20 warmup runs, then 200 timed runs via CUDA Events. Speedup = baseline_latency / result_latency.

### 1. Global magnitude pruning (`run_prune.py`, results: `results/andrey-results/prune_*.json`)

Zeroes the N% smallest-magnitude weights globally across all 1×1 Conv2d layers using `torch.nn.utils.prune.global_unstructured`. Tensors stay dense — no real speedup (≈1.0×) at any ratio. SSIM degrades from 0.7689 to 0.67 at 100% ratio.

### 2. Structural expand-channel pruning (`run_prune_structured.py`, results: `struct_*.json`)

Physically removes the lowest-L1-norm output channels from the expand 1×1 conv in each MBResBlock. Only `mid_ch` shrinks — `in_ch`, `out_ch`, and skip connections are untouched, so U-Net topology stays valid. New `mid_ch` rounded to a multiple of `num_groups` (GroupNorm constraint).

| Prune ratio | Speedup BS=1 | Speedup BS=32 | SSIM (after ft) | SSIM drop |
|---:|---:|---:|---:|---:|
| 25% | 1.23× | 1.23× | 0.7598 | −0.009 |
| 50% | 1.57× | 1.68× | 0.7581 | — |
| 75% | 2.27× | 2.53× | 0.7598 | −0.009 |

**Best:** 75% + 10 epochs finetune — 2.27× speedup, SSIM drop only 0.009, model 29.3 M → 8.4 M params.

### 3. Grouped 1×1 convolutions (`run_group_conv.py`, results: `group*.json`)

Replaces expand/project 1×1 Conv2d with grouped conv (G=2 or G=4). Init: block-diagonal slice of original weight matrix. FLOPs in targeted layers ÷ G, but actual speedup limited by CUDA matmul overhead.

| Config | Speedup | SSIM drop |
|---|---:|---:|
| G=2 expand | 1.03× | −0.016 |
| G=2 both | 1.07× | −0.021 |
| G=4 both | 1.13× | −0.035 |

Worse tradeoff than structural pruning.

### 4. 2:4 semi-structured sparsity (`run_sparse.py`, results: `sparse_244*.json`)

In every group of 4 consecutive weights exactly 2 are zeroed (smallest magnitude). `to_sparse_semi_structured` + Pointwise2d wrapper routes through `F.linear` to trigger cuSPARSELt.

**Hardware constraints:** Ampere+ only (sm_80+), FP16/BF16 only, `F.linear` only (not `F.conv2d`).  
RTX 8000 = Turing (sm_75) → falls back to dense masking, no speedup.

**A100 results (FP16):**

| BS | Baseline | Sparse | Speedup |
|---:|---:|---:|---:|
| 1 | 26.0 ms | 31.1 ms | 0.84× |
| 32 | 155 ms | 411 ms | 0.38× |

No speedup at any tested batch size. Matrices too small for cuSPARSELt overhead to amortise — expected gain at BS ≥ 256.
