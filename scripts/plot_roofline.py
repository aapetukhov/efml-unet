"""
Roofline analysis for UNetSR baseline (x4) and SRUNetHeavy (29M).
Saves results/roofline_baseline.png and results/roofline_heavy.png,
plus a combined results/roofline_combined.png.

RTX 3060 specs:
  FP32 peak:       12,740 GFLOPs/s
  FP16 peak (TC):  101,920 GFLOPs/s
  DRAM bandwidth:  360 GB/s  (= 360,000 MB/s = 360 GBytes/s)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── GPU constants ──────────────────────────────────────────────────────────────
BW_GB   = 360.0          # GB/s DRAM bandwidth
PEAK_FP32_TFLOPS = 12.74 # TFLOPs/s
PEAK_FP16_TFLOPS = 101.92  # TFLOPs/s (tensor cores)

# ── Model constants ────────────────────────────────────────────────────────────
# Baseline UNetSR x4:  LR 256×256 → HR 1024×1024
#   FLOPs: 806.21 GFLOPs (from thop)
#   DRAM traffic FP32: 8,892 MB (upper bound, from layer-wise analysis)
#   DRAM traffic FP16: half the activations+params → ~4,446 MB
BASELINE_FLOPS_G = 806.21  # GFLOPs
BASELINE_DRAM_FP32_GB = 8.892  # GB
BASELINE_DRAM_FP16_GB = 4.446

# Heavy SRUNetHeavy 29M:  LR 256×256 → LR 256×256  (no PixelShuffle SR)
#   FLOPs: 137.70 GFLOPs (from thop)
#   DRAM traffic estimate FP32:
#     params: 29.2M × 4B = 116.8 MB
#     activations (encoder+decoder+expanded mid tensors): ~540 MB
#     total: ~0.657 GB
HEAVY_FLOPS_G = 137.70
HEAVY_DRAM_FP32_GB = 0.657
HEAVY_DRAM_FP16_GB = 0.328

# ── Benchmark results (from benchmark_baseline_x4_full.json and benchmark_heavy_29M_full.json)
BASELINE_RESULTS = [
    # (label, latency_ms, precision)
    ("Eager FP32",           119.7, "fp32"),
    ("Eager FP16",            76.1, "fp16"),
    ("torch.compile",        109.2, "fp32"),
    ("ORT CUDA EP",          120.3, "fp32"),
    ("TensorRT FP32",         90.6, "fp32"),
    ("TensorRT FP16",         30.8, "fp16"),
    ("TVM MetaSchedule",     136.3, "fp32"),
]

HEAVY_RESULTS = [
    ("Eager FP32",            48.1, "fp32"),
    ("Eager FP16",            31.7, "fp16"),
    ("torch.compile",         46.6, "fp32"),
    ("ORT CUDA EP",           46.1, "fp32"),
    ("TensorRT FP32",         29.8, "fp32"),
    ("TensorRT FP16",         13.3, "fp16"),
]


def roofline_perf(ai: float, bw: float, peak: float) -> float:
    """Roofline model: min(bandwidth × AI, peak_compute) in GFLOPs/s."""
    return min(bw * ai, peak)


def plot_single(
    ax: plt.Axes,
    model_flops_g: float,
    dram_fp32_gb: float,
    dram_fp16_gb: float,
    results: list,
    title: str,
    annotate_points: bool = True,
) -> None:
    # AI for each precision
    ai_fp32 = model_flops_g / dram_fp32_gb           # FLOPs/byte (GFLOPs/GB = FLOPs/byte)
    ai_fp16 = model_flops_g / dram_fp16_gb

    peak_fp32 = PEAK_FP32_TFLOPS * 1000  # GFLOPs/s
    peak_fp16 = PEAK_FP16_TFLOPS * 1000

    # AI range for roofline curve
    ai_range = np.logspace(-1, 3.5, 500)

    # Draw rooflines
    roof_fp32 = np.minimum(BW_GB * ai_range, peak_fp32)
    roof_fp16 = np.minimum(BW_GB * ai_range, peak_fp16)

    ax.loglog(ai_range, roof_fp32, "b-",  linewidth=2,  label="FP32 roofline", zorder=1)
    ax.loglog(ai_range, roof_fp16, "r--", linewidth=2,  label="FP16 TC roofline", zorder=1)

    # Ridge points
    ridge_fp32 = peak_fp32 / BW_GB
    ridge_fp16 = peak_fp16 / BW_GB
    ax.axvline(ridge_fp32, color="b", linestyle=":", alpha=0.4)
    ax.axvline(ridge_fp16, color="r", linestyle=":", alpha=0.4)
    ax.text(ridge_fp32 * 1.05, peak_fp32 * 0.55,
            f"FP32 ridge\n{ridge_fp32:.1f} FLOPs/B", color="b", fontsize=7, va="top")
    ax.text(ridge_fp16 * 1.05, peak_fp16 * 0.55,
            f"FP16 TC ridge\n{ridge_fp16:.0f} FLOPs/B", color="r", fontsize=7, va="top")

    # Peak lines
    ax.axhline(peak_fp32, color="b", linestyle="-.", alpha=0.25, linewidth=1)
    ax.axhline(peak_fp16, color="r", linestyle="-.", alpha=0.25, linewidth=1)

    # Mark the model's AI on X axis
    for ai, color, prec in [(ai_fp32, "blue", "FP32"), (ai_fp16, "red", "FP16")]:
        ax.axvline(ai, color=color, alpha=0.15, linewidth=6)
        ax.text(ai, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1,
                f" {prec}\nAI={ai:.0f}", fontsize=6.5, color=color, va="bottom", ha="left")

    # Marker styles by method
    MARKERS = {
        "Eager FP32":    ("o", "steelblue"),
        "Eager FP16":    ("o", "tomato"),
        "torch.compile": ("s", "steelblue"),
        "ORT CUDA EP":   ("D", "steelblue"),
        "TensorRT FP32": ("^", "green"),
        "TensorRT FP16": ("^", "firebrick"),
        "TVM MetaSchedule": ("P", "purple"),
    }
    offsets = {
        "Eager FP32":    (1.6, 0),
        "Eager FP16":    (1.6, 0),
        "torch.compile": (1.6, -0.06),
        "ORT CUDA EP":   (1.6, 0.06),
        "TensorRT FP32": (1.6, 0),
        "TensorRT FP16": (1.6, 0),
        "TVM MetaSchedule": (1.6, 0.06),
    }

    for label, lat_ms, prec in results:
        ai   = ai_fp16 if prec == "fp16" else ai_fp32
        perf = model_flops_g / (lat_ms / 1000)   # GFLOPs/s
        mark, color = MARKERS.get(label, ("o", "gray"))
        ax.loglog(ai, perf, mark, color=color, markersize=9,
                  markeredgecolor="white", markeredgewidth=0.8, zorder=5)
        if annotate_points:
            dx, dy = offsets.get(label, (1.6, 0))
            ax.annotate(
                f"{label}\n{lat_ms:.0f} ms",
                xy=(ai, perf), xytext=(ai * dx, perf * 10**dy),
                fontsize=6.5, color=color,
                arrowprops=dict(arrowstyle="-", color=color, lw=0.7),
                ha="left",
            )

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)", fontsize=10)
    ax.set_ylabel("Performance (GFLOPs/s)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xlim(0.5, 2000)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(500, peak_fp16 * 2.5)


def make_plots(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-model plots ─────────────────────────────────────────────────────────
    for fname, flops, dram32, dram16, results, title in [
        ("roofline_baseline.png",
         BASELINE_FLOPS_G, BASELINE_DRAM_FP32_GB, BASELINE_DRAM_FP16_GB,
         BASELINE_RESULTS,
         "Roofline — UNetSR Baseline (8.2 M params, 256→1024, RTX 3060)"),
        ("roofline_heavy.png",
         HEAVY_FLOPS_G, HEAVY_DRAM_FP32_GB, HEAVY_DRAM_FP16_GB,
         HEAVY_RESULTS,
         "Roofline — SRUNetHeavy 29 M (256→256, RTX 3060)"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_single(ax, flops, dram32, dram16, results, title)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_dir / fname}")

    # ── Combined side-by-side ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    plot_single(ax1, BASELINE_FLOPS_G, BASELINE_DRAM_FP32_GB, BASELINE_DRAM_FP16_GB,
                BASELINE_RESULTS, "UNetSR Baseline (8.2 M, 806 GFLOPs)", annotate_points=False)
    plot_single(ax2, HEAVY_FLOPS_G, HEAVY_DRAM_FP32_GB, HEAVY_DRAM_FP16_GB,
                HEAVY_RESULTS, "SRUNetHeavy 29 M (138 GFLOPs)", annotate_points=False)

    # Shared legend on combined
    handles = []
    for label, (mark, color) in {
        "Eager FP32/FP16": ("o", "steelblue"),
        "torch.compile":   ("s", "steelblue"),
        "ORT CUDA EP":     ("D", "steelblue"),
        "TensorRT FP32":   ("^", "green"),
        "TensorRT FP16":   ("^", "firebrick"),
        "TVM MetaSchedule":("P", "purple"),
    }.items():
        handles.append(plt.Line2D([0], [0], marker=mark, color="w",
                                   markerfacecolor=color, markersize=9, label=label))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Roofline Analysis — RTX 3060 (12.74 TFLOPs FP32, 101.9 TFLOPs FP16 TC, 360 GB/s)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    fig.savefig(out_dir / "roofline_combined.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_dir / 'roofline_combined.png'}")


if __name__ == "__main__":
    make_plots(Path("results"))
