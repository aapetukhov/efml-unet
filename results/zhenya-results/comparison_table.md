# Benchmark Results

**Hardware:** NVIDIA GeForce RTX 3060 (12 GB, 360 GB/s)  
**Dataset:** DIV2K validation (99 images after filtering, crop 256×256 LR → 1024×1024 HR)  
**Scale:** ×4  
**Metric:** PSNR / SSIM на центральном кропе 256×256 LR

---

## Batch size = 1

| Method | Precision | Latency mean | Latency p95 | Throughput | PSNR | SSIM | vs FP32 baseline |
|---|---|---:|---:|---:|---:|---:|---:|
| Eager PyTorch | FP32 | 125.0 ms | 130.0 ms | 8.0 img/s | 23.497 | 0.6918 | ×1.00 |
| Eager PyTorch | FP16 | 81.5 ms | 86.6 ms | 12.3 img/s | 23.497 | 0.6918 | ×1.53 |
| `torch.compile(default)` | FP32 | 237 ms | 235 ms | 4.2 img/s | 23.497 | 0.6918 | ×0.53 |
| `torch.compile(reduce-overhead)` | FP32 | 233 ms | 235 ms | 4.3 img/s | 23.497 | 0.6918 | ×0.54 |
| ORT CUDA EP | FP32 | 130 ms | 143 ms | 7.7 img/s | 23.497 | 0.6918 | ×0.97 |
| ORT TensorRT EP | FP32 | 97 ms | 100 ms | 10.3 img/s | 23.497 | 0.6918 | ×1.40 |
| **ORT TensorRT EP** | **FP16** | **34 ms** | **35 ms** | **29.7 img/s** | **23.497** | **0.6918** | **×4.02** |
| TVM (DLight, no tuning) | FP32 | 511 ms | 516 ms | 2.0 img/s | 23.497 | 0.6918 | ×0.24 |
| TVM MetaSchedule (512 trials, 37.5 min) | FP32 | 205 ms | 208 ms | 4.9 img/s | 23.497 | 0.6918 | ×0.61 |
| TVM MetaSchedule (1024 trials, 72 min) | FP32 | 152 ms | 153 ms | 6.6 img/s | 23.497 | 0.6918 | ×0.83 |

---

## Batch size sweep — Eager PyTorch FP32

| BS | Latency mean | Latency p95 | Throughput | PSNR | SSIM |
|---:|---:|---:|---:|---:|---:|
| 1 | 125.0 ms | 130.0 ms | 8.0 img/s | 23.497 | 0.6918 |
| 2 | 251.8 ms | 259.4 ms | 7.8 img/s | 23.248 | 0.6823 |
| 4 | 495.2 ms | 508.2 ms | 7.9 img/s | 23.397 | 0.6856 |
| 8+ | OOM | — | — | — | — |

## Batch size sweep — Eager PyTorch FP16

| BS | Latency mean | Latency p95 | Throughput | PSNR | SSIM |
|---:|---:|---:|---:|---:|---:|
| 1 | 81.5 ms | 86.6 ms | 12.3 img/s | 23.497 | 0.6918 |
| 2 | 156.9 ms | 165.3 ms | 12.6 img/s | 23.248 | 0.6823 |
| 4 | 301.6 ms | 311.5 ms | 13.0 img/s | 23.397 | 0.6856 |
| 8 | 456.3 ms | 570.7 ms | 13.9 img/s | — | — |
| 16+ | OOM | — | — | — | — |

## Batch size sweep — TensorRT FP16

| BS | Latency mean | Latency p95 | Throughput | vs BS=1 |
|---:|---:|---:|---:|---:|
| 1 | 32.9 ms | 36.3 ms | 30.4 img/s | ×1.00 |
| 2 | 62.9 ms | 65.2 ms | 31.8 img/s | ×1.05 |
| 4 | 121.4 ms | 125.7 ms | 32.9 img/s | ×1.08 |
| 8 | 224.1 ms | 246.1 ms | 35.7 img/s | ×1.17 |
| 16+ | OOM | — | — | — |

---

## Roofline (BS=1, FP32)

| Quantity | Value |
|---|---:|
| FLOPs | 806 GFLOPs |
| DRAM traffic (upper bound) | 8,892 MB |
| Arithmetic Intensity | 90.7 FLOPs/byte |
| FP32 ridge point | 35.4 FLOPs/byte |
| FP16 (no TC) ridge point | 70.8 FLOPs/byte |
| FP16 (tensor cores) ridge point | 283 FLOPs/byte |
| Achieved (eager FP32) | 6,452 GFLOPs/s |
| FP32 peak | 12,740 GFLOPs/s |
| GPU utilisation | ~51% |

Модель **compute-bound** в FP32 и FP16 без tensor cores, **memory-bound** относительно FP16 tensor core ceiling.

---

## Выводы

- `torch.compile` (Triton/Inductor) **медленнее** eager — Triton-ядра проигрывают cuDNN на стандартных 3×3 conv
- ORT CUDA EP даёт те же ~8 img/s — нет fusion
- TensorRT FP32 даёт **×1.4** за счёт layer fusion и autotuning cuDNN
- TensorRT FP16 даёт **×4** — tensor cores + fusion + вдвое меньше DRAM трафика
- Батчинг при любом методе даёт **≤+20%** throughput — GPU уже насыщен на BS=1 при 1024×1024 HR
- TVM DLight (без tuning) **в 4× медленнее** eager — DLight Fallback не может конкурировать с cuDNN для 3×3 conv
- TVM MetaSchedule: каждое удвоение trials даёт ~25% ускорение (512→205 ms, 1024→152 ms); 1024 trials ≈ **×0.83** от PyTorch — близко, но ещё не обгоняет cuDNN
