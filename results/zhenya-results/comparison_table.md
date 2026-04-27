# Benchmark Results

**Hardware:** NVIDIA GeForce RTX 3060 (12 GB, 360 GB/s)  
**Input:** 256×256 LR crop, batch size = 1  
**Latency:** CUDA Events, 10 warmup + 50 measured iterations, synthetic noise input  
**TensorRT:** `libnvinfer.so.10` from `.venv/lib/python3.12/site-packages/tensorrt_libs/`  
(set `LD_LIBRARY_PATH=$VIRTUAL_ENV/lib/python3.12/site-packages/tensorrt_libs` before running)

---

## UNetSR Baseline — scale ×4  (8.2 M params, 806 GFLOPs, LR 256² → HR 1024²)

| Method | Precision | Latency mean | Latency p95 | Throughput | vs FP32 |
|---|---|---:|---:|---:|---:|
| Eager PyTorch | FP32 | 119.7 ms | 119.8 ms | 8.4 img/s | ×1.00 |
| Eager PyTorch | FP16 | 76.1 ms | 76.1 ms | 13.1 img/s | ×1.57 |
| `torch.compile(default)` | FP32 | 109.7 ms | 110.0 ms | 9.1 img/s | ×1.09 |
| `torch.compile(reduce-overhead)` | FP32 | 109.4 ms | 109.5 ms | 9.1 img/s | ×1.09 |
| `torch.compile(max-autotune)` | FP32 | 109.2 ms | 109.3 ms | 9.2 img/s | ×1.10 |
| ORT CUDA EP | FP32 | 120.3 ms | 120.5 ms | 8.3 img/s | ×0.99 |
| **ORT TensorRT EP** | **FP32** | **90.6 ms** | **91.1 ms** | **11.0 img/s** | **×1.32** |
| **ORT TensorRT EP** | **FP16** | **30.8 ms** | **31.1 ms** | **32.4 img/s** | **×3.87** |
| TVM (DLight, no tuning) | FP32 | 511 ms | 516 ms | 2.0 img/s | ×0.23 |
| TVM MetaSchedule (512 trials, 37.5 min, replay-func) | FP32 | 205 ms | 208 ms | 4.9 img/s | ×0.58 |
| TVM MetaSchedule (1024 trials, 72 min, replay-func) | FP32 | 152 ms | 153 ms | 6.6 img/s | ×0.79 |
| TVM MetaSchedule (2047 trials, 74.6 min, evolutionary) | FP32 | 136 ms | 138 ms | 7.3 img/s | ×0.88 |

> **Note on `torch.compile`:** Earlier runs reported ×0.5 (237 ms). That was measured during JIT compilation — Triton kernels compile on first call, taking ~5–10 s. With 10 warmup iterations the compile completes before measurement. True steady-state performance is ×1.09 vs eager.

---

## SRUNetHeavy — scale ×4  (29.2 M params, 138 GFLOPs, LR 256² → LR 256²)

| Method | Precision | Latency mean | Latency p95 | Throughput | vs FP32 |
|---|---|---:|---:|---:|---:|
| Eager PyTorch | FP32 | 48.1 ms | 49.0 ms | 20.8 img/s | ×1.00 |
| Eager PyTorch | FP16 | 31.7 ms | 31.8 ms | 31.6 img/s | ×1.52 |
| `torch.compile(default)` | FP32 | 47.0 ms | 47.9 ms | 21.3 img/s | ×1.02 |
| `torch.compile(reduce-overhead)` | FP32 | 47.1 ms | 47.6 ms | 21.2 img/s | ×1.02 |
| `torch.compile(max-autotune)` | FP32 | 46.6 ms | 46.7 ms | 21.5 img/s | ×1.03 |
| ORT CUDA EP | FP32 | 46.1 ms | 48.2 ms | 21.6 img/s | ×1.04 |
| **ORT TensorRT EP** | **FP32** | **29.8 ms** | **30.1 ms** | **33.6 img/s** | **×1.61** |
| **ORT TensorRT EP** | **FP16** | **13.3 ms** | **13.5 ms** | **75.2 img/s** | **×3.62** |

---

## Batch size sweep — Eager PyTorch FP32 (Baseline)

| BS | Latency mean | Latency p95 | Throughput | PSNR | SSIM |
|---:|---:|---:|---:|---:|---:|
| 1 | 119.7 ms | 119.8 ms | 8.4 img/s | — | — |
| 2 | 251.8 ms | 259.4 ms | 7.8 img/s | — | — |
| 4 | 495.2 ms | 508.2 ms | 7.9 img/s | — | — |
| 8+ | OOM | — | — | — | — |

## Batch size sweep — Eager PyTorch FP16 (Baseline)

| BS | Latency mean | Latency p95 | Throughput |
|---:|---:|---:|---:|
| 1 | 76.1 ms | 76.1 ms | 13.1 img/s |
| 2 | 156.9 ms | 165.3 ms | 12.6 img/s |
| 4 | 301.6 ms | 311.5 ms | 13.0 img/s |
| 8 | 456.3 ms | 570.7 ms | 13.9 img/s |
| 16+ | OOM | — | — |

## Batch size sweep — TensorRT FP16 (Baseline)

| BS | Latency mean | Latency p95 | Throughput | vs BS=1 |
|---:|---:|---:|---:|---:|
| 1 | 30.8 ms | 31.2 ms | 32.4 img/s | ×1.00 |
| 2 | 62.9 ms | 65.2 ms | 31.8 img/s | ×0.98 |
| 4 | 121.4 ms | 125.7 ms | 32.9 img/s | ×1.01 |
| 8 | 224.1 ms | 246.1 ms | 35.7 img/s | ×1.10 |
| 16+ | OOM | — | — | — |

---

## Roofline Analysis (BS=1)

### UNetSR Baseline

| Quantity | Value |
|---|---:|
| FLOPs | 806 GFLOPs |
| DRAM traffic FP32 (upper bound) | 8,892 MB |
| DRAM traffic FP16 (upper bound) | 4,446 MB |
| Arithmetic Intensity FP32 | 90.7 FLOPs/byte |
| Arithmetic Intensity FP16 | 181 FLOPs/byte |
| FP32 ridge point | 35.4 FLOPs/byte |
| FP16 TC ridge point | 283 FLOPs/byte |
| Achieved (eager FP32) | 6,733 GFLOPs/s (52.9% of FP32 peak) |
| Achieved (TRT FP32) | 8,896 GFLOPs/s (69.8% of FP32 peak) |
| Achieved (TRT FP16) | 26,175 GFLOPs/s (25.7% of FP16 TC peak) |

### SRUNetHeavy 29M

| Quantity | Value |
|---|---:|
| FLOPs | 138 GFLOPs |
| DRAM traffic FP32 (estimate) | 657 MB |
| DRAM traffic FP16 (estimate) | 328 MB |
| Arithmetic Intensity FP32 | 210 FLOPs/byte |
| Arithmetic Intensity FP16 | 420 FLOPs/byte |
| FP32 ridge point | 35.4 FLOPs/byte |
| FP16 TC ridge point | 283 FLOPs/byte |
| Achieved (eager FP32) | 2,862 GFLOPs/s (22.5% of FP32 peak) |
| Achieved (TRT FP32) | 4,620 GFLOPs/s (36.3% of FP32 peak) |
| Achieved (TRT FP16) | 10,353 GFLOPs/s (10.2% of FP16 TC peak) |

See `results/roofline_combined.png`.

---

## Выводы

### UNetSR Baseline
- Модель **compute-bound** в FP32 (AI=91 > ridge=35): обе ветви roofline — compute. Отсюда `torch.compile` (Triton) даёт +9% — cuDNN уже близко к потолку, Triton может предложить лучший tile size для конкретных ядер.
- `torch.compile` корректно даёт **×1.09** (не хуже eager, как казалось ранее) — прошлые 237 ms измерялись во время JIT-компиляции Triton-ядер.
- ORT CUDA EP не даёт ускорения (×0.99) — граф-оптимизации ORT эквивалентны eager; нет fusion, которого не делает cuDNN.
- TensorRT FP32 **×1.32** за счёт layer fusion (conv+bias+relu → единое ядро) и autotuning cuDNN algorithm per-layer.
- TensorRT FP16 **×3.87** — tensor cores + fusion + вдвое меньше DRAM трафика. AI=181 < TC ridge=283 → модель **memory-bound** относительно TC ceiling, т.е. потенциал ещё не исчерпан.
- TVM MetaSchedule: каждое удвоение trials даёт ~25% ускорение. Evolutionary strategy + 2047 trials = **136 ms (×0.88)** — TVM вплотную к cuDNN без явной оптимизации ядер. 2 workload с OOB в run4 исправлены заменой на расписания из run3.

### SRUNetHeavy 29M
- Модель в 3× быстрее baseline в FP32 (48 vs 120 ms) при в 4× больших параметрах — за счёт MBResBlock (DWConv3×3 + 1×1): ~7× меньше FLOPs (138 vs 806 GFLOPs), всё вычисление на LR-разрешении 256×256.
- AI=210 (FP32) далеко от ridge=35 → compute-bound, но достигается лишь 22.5% пика — DWConv плохо векторизуется: каждый input-channel независим, warp occupancy низкий.
- TRT FP32 **×1.61** лучше, чем у baseline (×1.32) — fusion conv1×1+GN+GELU даёт бо́льший выигрыш, чем у baseline со стандартными BN-free ResBlocks.
- TRT FP16 **×3.62** — AI=420 > TC ridge=283 → модель **compute-bound** даже для TC; throughput 10,353 GFLOPs/s = 10% от TC peak. DWConv не использует TC (TC нужны матричные умножения ≥16×16), поэтому FP16 выигрывает только от уменьшения bandwidth, а не от TC.
- `torch.compile` и ORT дают минимальный выигрыш (×1.02–1.04) — Triton и ORT не находят fusion для GroupNorm+GELU+DWConv, который TRT может закодировать в плагинах.
