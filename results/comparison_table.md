# Results Table Template

## Midterm baseline table

| Experiment | Dataset | Scale | Precision | PSNR | SSIM | Mean latency, ms | P95 latency, ms | Throughput, img/s | Notes |
|---|---|---:|---|---:|---:|---:|---:|---:|---|
| PyTorch baseline | Validation split | x2 | FP32 | TBD | TBD | TBD | TBD | TBD | First reproducible SR baseline |
| PyTorch baseline | Validation split | x2 | FP16 | TBD | TBD | TBD | TBD | TBD | Reference for future comparisons |

## Final project table

| Pipeline | Scale | Precision | PSNR | SSIM | Mean latency, ms | P95 latency, ms | Throughput, img/s | Delta PSNR vs FP16 ref | Delta SSIM vs FP16 ref | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| PyTorch baseline | x2 | FP16 | TBD | TBD | TBD | TBD | TBD | 0.000 | 0.000 | Reference |
| `torch.compile` | x2 | FP16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| TVM | x2 | FP16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| Quantization | x2 | INT8 / mixed | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| Pruning | x2 | FP16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
| 2:4 sparsity | x2 | FP16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | |
