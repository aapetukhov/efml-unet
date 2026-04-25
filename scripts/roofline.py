"""
Roofline analysis for UNetSR x4 on NVIDIA GeForce RTX 3060.

Hardware specs (RTX 3060, GA106, 12 GB):
  FP32 peak  :  12.74 TFLOPS
  FP16 peak  :  25.48 TFLOPS  (non-tensor-core shader path, 2× FP32)
  FP16 TC    : 101.90 TFLOPS  (Ampere tensor cores)
  Mem BW     :    360 GB/s
"""
from __future__ import annotations

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from thop import profile

from src.modeling import build_model

# ── Hardware specs ───────────────────────────────────────────────────────────
GPU_NAME = "NVIDIA GeForce RTX 3060"
HW = {
    "FP32":         12.74e12,
    "FP16 (no TC)": 25.48e12,
    "FP16 (TC)":   101.90e12,
}
MEM_BW = 360e9  # bytes/s

# ── Model FLOPs & memory traffic ─────────────────────────────────────────────
CROP_LR = 256
SCALE   = 4
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model(3, 3, 64, SCALE).eval().to(DEVICE)
x_fp32 = torch.randn(1, 3, CROP_LR, CROP_LR, device=DEVICE)

macs, params = profile(model, inputs=(x_fp32,), verbose=False)
flops = 2 * macs

# DRAM traffic via hooks (FP32 upper bound)
_bytes_fp32 = 0
def _make_hook(bpe):
    def hook(m, inp, out):
        global _bytes_fp32
        for t in inp:
            if isinstance(t, torch.Tensor): _bytes_fp32 += t.numel() * bpe
        if isinstance(out, torch.Tensor):   _bytes_fp32 += out.numel() * bpe
        for p in m.parameters(recurse=False): _bytes_fp32 += p.numel() * bpe
    return hook

handles = [m.register_forward_hook(_make_hook(4))
           for m in model.modules() if not list(m.children())]
with torch.inference_mode(): model(x_fp32)
for h in handles: h.remove()

bytes_fp32 = _bytes_fp32
bytes_fp16 = bytes_fp32 // 2   # FP16: 2 bytes/elem instead of 4

ai_fp32 = flops / bytes_fp32
ai_fp16 = flops / bytes_fp16

# ── Experiments (method, label, ai, latency_ms, precision_label, marker) ────
# latencies from benchmark runs (BS=1)
experiments = [
    # label                       ai        lat_ms   color    marker
    ("Eager FP32",               ai_fp32,   125.0,  "#1f77b4", "o"),
    ("Eager FP16",               ai_fp16,    81.5,  "#ff7f0e", "o"),
    ("torch.compile (default)",  ai_fp32,   237.0,  "#9467bd", "^"),
    ("torch.compile (r-o)",      ai_fp32,   233.0,  "#8c564b", "^"),
    ("ORT CUDA EP",              ai_fp32,   130.0,  "#17becf", "s"),
    ("TensorRT FP32",            ai_fp32,    97.0,  "#2ca02c", "D"),
    ("TensorRT FP16",            ai_fp16,    34.0,  "#d62728", "D"),
]

# ── Calculations printout ─────────────────────────────────────────────────────
SEP = "=" * 62
print(SEP)
print(f"Model: UNetSR x4  |  GPU: {GPU_NAME}")
print(SEP)
print(f"  Parameters        : {params/1e6:.2f} M  ({params*4/1e6:.1f} MB FP32)")
print(f"  FLOPs             : {flops/1e9:.1f} GFLOPs  ({macs/1e9:.1f} GMACs × 2)")
print(f"  DRAM traffic FP32 : {bytes_fp32/1e6:.0f} MB  (upper bound, no cache reuse)")
print(f"  DRAM traffic FP16 : {bytes_fp16/1e6:.0f} MB")
print(f"  AI (FP32)         : {ai_fp32:.1f} FLOPs/byte")
print(f"  AI (FP16)         : {ai_fp16:.1f} FLOPs/byte")
print(SEP)
print("Ridge points (compute peak / memory bandwidth):")
for name, peak in HW.items():
    ridge = peak / MEM_BW
    ai32_region = "compute-bound" if ai_fp32 > ridge else "memory-bound"
    ai16_region = "compute-bound" if ai_fp16 > ridge else "memory-bound"
    print(f"  {name:18s}: {ridge:6.1f} FLOPs/B  "
          f"(FP32→{ai32_region}, FP16→{ai16_region})")
print(SEP)
print(f"{'Method':<28} {'lat_ms':>7} {'GFLOPs/s':>10} {'roof':>10} {'util%':>6}")
print("-" * 62)
for label, ai, lat_ms, *_ in experiments:
    lat_s = lat_ms / 1000
    achieved = flops / lat_s / 1e9
    precision = "FP16" if "FP16" in label or "16" in label else "FP32"
    peak = HW["FP16 (TC)"] if "TensorRT FP16" in label else \
           HW["FP16 (no TC)"] if "FP16" in label else HW["FP32"]
    roof_at_ai = min(peak, MEM_BW * ai) / 1e9
    util = 100 * achieved / roof_at_ai
    print(f"  {label:<26} {lat_ms:>6.1f}ms {achieved:>9.0f}  {roof_at_ai:>9.0f}  {util:>5.1f}%")
print(SEP)

# ── Write calculations file ───────────────────────────────────────────────────
calc_path = Path(__file__).parent.parent / "results" / "roofline_calculations.md"
lines = [
    "# Roofline Calculations — UNetSR x4 on RTX 3060\n",
    f"## Hardware: {GPU_NAME}\n",
    "| Spec | Value |",
    "|---|---|",
    "| FP32 peak | 12.74 TFLOPS |",
    "| FP16 peak (no tensor cores) | 25.48 TFLOPS |",
    "| FP16 peak (tensor cores) | 101.90 TFLOPS |",
    "| Memory bandwidth | 360 GB/s |",
    "| VRAM | 12 GB GDDR6 |",
    "",
    "## Model: UNetSR ×4 (base_channels=64)\n",
    "| Quantity | Calculation | Value |",
    "|---|---|---|",
    f"| Parameters | — | {params/1e6:.2f} M |",
    f"| Weight size (FP32) | {params/1e6:.2f}M × 4 bytes | {params*4/1e6:.1f} MB |",
    f"| MACs | thop profile, 256×256 LR input | {macs/1e9:.1f} GMACs |",
    f"| FLOPs | MACs × 2 | {flops/1e9:.1f} GFLOPs |",
    f"| DRAM traffic FP32 | hook: Σ(inputs + outputs + weights) per leaf | {bytes_fp32/1e6:.0f} MB |",
    f"| DRAM traffic FP16 | FP32 traffic / 2 | {bytes_fp16/1e6:.0f} MB |",
    f"| Arithmetic intensity FP32 | {flops/1e9:.1f} GFLOPs / {bytes_fp32/1e9:.2f} GB | {ai_fp32:.1f} FLOPs/byte |",
    f"| Arithmetic intensity FP16 | {flops/1e9:.1f} GFLOPs / {bytes_fp16/1e9:.2f} GB | {ai_fp16:.1f} FLOPs/byte |",
    "",
    "## Ridge points\n",
    "Ridge = compute_peak / memory_bandwidth\n",
    "| Precision | Compute peak | Ridge | UNetSR FP32 region | UNetSR FP16 region |",
    "|---|---|---|---|---|",
]
for name, peak in HW.items():
    ridge = peak / MEM_BW
    r32 = "compute-bound" if ai_fp32 > ridge else "memory-bound"
    r16 = "compute-bound" if ai_fp16 > ridge else "memory-bound"
    lines.append(f"| {name} | {peak/1e12:.2f} TFLOPS | {ridge:.1f} FLOPs/B | {r32} | {r16} |")

lines += [
    "",
    "## Achieved performance (BS=1)\n",
    "Performance = FLOPs / latency  \n"
    "Roofline ceiling at AI = min(compute_peak, mem_bw × AI)\n",
    "| Method | AI (FLOPs/B) | Latency | Achieved (GFLOPs/s) | Ceiling (GFLOPs/s) | Utilisation |",
    "|---|---|---|---|---|---|",
]
for label, ai, lat_ms, *_ in experiments:
    lat_s = lat_ms / 1000
    achieved = flops / lat_s / 1e9
    peak = HW["FP16 (TC)"] if "TensorRT FP16" in label else \
           HW["FP16 (no TC)"] if "FP16" in label else HW["FP32"]
    roof_at_ai = min(peak, MEM_BW * ai) / 1e9
    util = 100 * achieved / roof_at_ai
    lines.append(
        f"| {label} | {ai:.1f} | {lat_ms:.1f} ms | {achieved:.0f} | {roof_at_ai:.0f} | {util:.1f}% |"
    )

calc_path.write_text("\n".join(lines) + "\n")
print(f"\nCalculations saved → {calc_path}")

# ── Roofline plot ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6.5))
ai_range = np.logspace(-1, 4, 1000)

roof_colors = {"FP32": "#1f77b4", "FP16 (no TC)": "#ff7f0e", "FP16 (TC)": "#2ca02c"}
for name, peak in HW.items():
    roof = np.minimum(peak / 1e9, MEM_BW * ai_range / 1e9)
    ax.loglog(ai_range, roof, color=roof_colors[name], linewidth=2.2,
              label=f"{name}  ({peak/1e12:.1f} TFLOPS)", zorder=2)
    ridge = peak / MEM_BW
    ax.axvline(ridge, color=roof_colors[name], linestyle=":", linewidth=1.0, alpha=0.5, zorder=1)
    ax.text(ridge * 1.05, peak / 1e9 * 0.62, f"ridge\n{ridge:.0f}",
            color=roof_colors[name], fontsize=7.5, va="top", ha="left")

# AI vertical lines
for ai_val, label_str in [(ai_fp32, f"AI FP32 = {ai_fp32:.0f}"),
                           (ai_fp16, f"AI FP16 = {ai_fp16:.0f}")]:
    ax.axvline(ai_val, color="dimgray", linestyle="--", linewidth=1.2, alpha=0.6, zorder=1)
    ax.text(ai_val * 1.05, 6e2, label_str, color="dimgray", fontsize=8, va="bottom", rotation=90)

# Experiment dots — plot all, collect legend handles
exp_handles = []
for label, ai, lat_ms, color, marker in experiments:
    achieved = flops / (lat_ms / 1000) / 1e9
    h = ax.scatter([ai], [achieved], color=color, marker=marker, s=110, zorder=6,
                   edgecolors="white", linewidths=0.8, alpha=0.7,
                   label=f"{label}  ({achieved:.0f} GF/s,  {lat_ms:.0f} ms)")
    exp_handles.append(h)

# Two-column legend: rooflines top-left, experiments right side
roof_handles, roof_labels = ax.get_legend_handles_labels()
# First 3 are rooflines, rest are experiments
n_roof = len(HW)
leg1 = ax.legend(roof_handles[:n_roof], roof_labels[:n_roof],
                 loc="upper left", fontsize=8.5, title="Hardware ceilings",
                 title_fontsize=8.5, framealpha=0.9)
ax.add_artist(leg1)
ax.legend(roof_handles[n_roof:], roof_labels[n_roof:],
          loc="lower right", fontsize=8, title="Experiments (BS=1)",
          title_fontsize=8.5, framealpha=0.9)

ax.set_xlabel("Arithmetic Intensity [FLOPs / byte]", fontsize=11)
ax.set_ylabel("Performance [GFLOPs/s]", fontsize=11)
ax.set_title(f"Roofline — UNetSR ×4  |  {GPU_NAME}", fontsize=13)
ax.grid(True, which="both", alpha=0.2)
ax.set_xlim(5, 1e4)
ax.set_ylim(5e2, 2e5)
ax.xaxis.set_major_formatter(ticker.LogFormatterMathtext())
ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

out = Path(__file__).parent.parent / "results" / "roofline_rtx3060.png"
fig.tight_layout()
fig.savefig(out, dpi=150)
print(f"Plot saved    → {out}")
