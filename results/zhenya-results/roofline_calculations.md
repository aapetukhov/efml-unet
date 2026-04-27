# Roofline Calculations — UNetSR x4 on RTX 3060

---

## 1. Hardware: NVIDIA GeForce RTX 3060

| Spec | Value |
|---|---|
| CUDA cores | 3584 |
| Boost clock | 1777 MHz |
| FP32 peak | 12.74 TFLOPS |
| FP16 peak (no tensor cores) | 25.48 TFLOPS |
| FP16 peak (tensor cores) | 101.90 TFLOPS |
| Memory bandwidth | 360 GB/s |
| VRAM | 12 GB GDDR6 (192-bit x 15 Gbps) |

**Compute peak formulas:**

```
FP32 peak  = CUDA_cores x 2 ops/cycle x boost_clock
           = 3584 x 2 x 1.777 GHz = 12.74 TFLOPS

FP16 peak (no TC) = FP32 peak x 2          = 25.48 TFLOPS
FP16 peak (TC)    = FP32 peak x 8          = 101.9 TFLOPS  (Ampere tensor core throughput)

Memory bandwidth  = bus_width x data_rate
                  = 192 bit x 15 Gbps / 8 = 360 GB/s
```

---

## 2. Model FLOPs

For each Conv2d layer with input `(N, C_in, H, W)` and output `(N, C_out, H, W)`:

```
MACs  = C_in x K_h x K_w x C_out x H x W    (per sample)
FLOPs = 2 x MACs                              (1 MAC = 1 multiply + 1 add)
```

Total for UNetSR x4 (256x256 LR input, profiled with `thop`):

```
Total MACs  = 403.1 GMACs
Total FLOPs = 2 x 403.1 = 806.2 GFLOPs
Parameters  = 8.23 M  →  32.9 MB (FP32)
```

---

## 3. DRAM Traffic

Measured by forward hooks — each leaf module contributes:

```
bytes_per_module = Σ(input tensors) + output tensor + parameters
                   (all in bytes, read + write counted separately)

DRAM_traffic = Σ bytes_per_module   over all leaf modules
```

This is an **upper bound**: assumes no L1/L2 cache reuse between consecutive ops.

For FP16, every element is 2 bytes instead of 4:

```
DRAM_FP32 = 8892 MB   (4 bytes/elem)
DRAM_FP16 = DRAM_FP32 / 2 = 4446 MB   (2 bytes/elem)
```

---

## 4. Arithmetic Intensity (AI)

```
AI = FLOPs / DRAM_traffic    [FLOPs/byte]

AI_FP32 = 806.2 x 10^9 / (8892 x 10^6) =  90.7 FLOPs/byte
AI_FP16 = 806.2 x 10^9 / (4446 x 10^6) = 181.3 FLOPs/byte
```

---

## 5. Ridge Points

The ridge point separates the memory-bound and compute-bound regions:

```
ridge = compute_peak / memory_bandwidth    [FLOPs/byte]

if AI > ridge  →  compute-bound   (bottleneck: compute units)
if AI < ridge  →  memory-bound    (bottleneck: DRAM bandwidth)
```

| Precision | Compute peak | Ridge | AI FP32 (90.7) | AI FP16 (181.3) |
|---|---|---|---|---|
| FP32 | 12.74 TFLOPS | 12.74e12 / 360e9 = **35.4** FLOPs/B | compute-bound | compute-bound |
| FP16 (no TC) | 25.48 TFLOPS | 25.48e12 / 360e9 = **70.8** FLOPs/B | compute-bound | compute-bound |
| FP16 (TC) | 101.90 TFLOPS | 101.9e12 / 360e9 = **283.1** FLOPs/B | memory-bound | memory-bound |

---

## 6. Roofline Ceiling

The attainable performance at a given AI is bounded by both the compute peak and memory bandwidth:

```
ceiling(AI) = min(compute_peak,  mem_bw x AI)    [FLOP/s]
```

At our AI values:

```
FP32,     AI=90.7:  min(12740,  360 x 90.7)  = min(12740, 32652) = 12740 GFLOPs/s
FP16 noTC, AI=181:  min(25480,  360 x 181.3) = min(25480, 65268) = 25480 GFLOPs/s
FP16 TC,   AI=181:  min(101900, 360 x 181.3) = min(101900,65268) = 65268 GFLOPs/s
```

---

## 7. Achieved Performance & Utilisation (BS=1)

```
achieved_perf = FLOPs / latency                    [FLOP/s]
utilisation   = achieved_perf / ceiling(AI)  x 100  [%]
```

| Method | AI | Latency | Achieved | Ceiling | Utilisation |
|---|---:|---:|---:|---:|---:|
| Eager FP32 | 90.7 | 125.0 ms | 806.2e9 / 0.125 = **6450 GFLOPs/s** | 12740 | 50.6% |
| Eager FP16 | 181.3 | 81.5 ms | 806.2e9 / 0.0815 = **9892 GFLOPs/s** | 25480 | 38.8% |
| torch.compile (default) | 90.7 | 237.0 ms | 806.2e9 / 0.237 = **3402 GFLOPs/s** | 12740 | 26.7% |
| torch.compile (reduce-overhead) | 90.7 | 233.0 ms | 806.2e9 / 0.233 = **3460 GFLOPs/s** | 12740 | 27.2% |
| ORT CUDA EP | 90.7 | 130.0 ms | 806.2e9 / 0.130 = **6202 GFLOPs/s** | 12740 | 48.7% |
| TensorRT FP32 | 90.7 | 97.0 ms | 806.2e9 / 0.097 = **8311 GFLOPs/s** | 12740 | 65.2% |
| TensorRT FP16 | 181.3 | 34.0 ms | 806.2e9 / 0.034 = **23712 GFLOPs/s** | 65268 | 36.3% |

---

## 8. Notes on Accuracy

- **FLOPs** — `thop` counts MACs for Conv2d and Linear only. Bilinear upsample, PixelShuffle, LeakyReLU, MaxPool are not counted → actual FLOPs ~1–3% higher (underestimate).
- **DRAM traffic** — hook-based sum over all leaf modules, no cache reuse assumed → upper bound. Real traffic is lower due to L2 (3 MB on RTX 3060) caching activations between adjacent ops. AI is therefore a **lower bound** on true arithmetic intensity.
- **Achieved performance** — computed from wall-clock latency with `cuda.synchronize()` before/after forward pass, 50 steps after 10 warmup. Includes kernel launch overhead.
- **TensorRT FP16 ceiling** uses FP16 TC peak (101.9 TFLOPS) since TensorRT explicitly routes to tensor cores. Eager FP16 uses FP16 no-TC peak since `.half()` cast without AMP does not guarantee tensor core usage on conv layers.
