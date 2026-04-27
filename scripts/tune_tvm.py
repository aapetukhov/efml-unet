"""
MetaSchedule autotuning for UNetSR x4 via TVM Relax (ONNX frontend).

Pipeline:
  1. ONNX → Relax IR (high-level R.nn.conv2d ops)
  2. LegalizeOps → call_tir with TIR PrimFuncs (33 unique tasks)
  3. tune_relax(lowered_mod) → MetaSchedule tunes the PrimFuncs
  4. Saves tuning database to results/tvm_tuning/

Usage (from efml-unet/):
  python scripts/tune_tvm.py [--trials 500] [--trials-per-task 64]
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.modeling import build_model
from src.config import load_config

CROP_LR   = 256
SCALE     = 4
ARCH      = "sm_86"
CONFIG    = "configs/sr_baseline_x4.yaml"
WORK_DIR  = "results/tvm_tuning"
ONNX_PATH = "results/unet_sr_x4.onnx"

# sm_86 (RTX 30xx / A10 / A30 / A40) hardware limits for MetaSchedule
_CUDA_TARGET = {
    "kind": "cuda",
    "arch": ARCH,
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,
}


def export_onnx_if_needed(model: torch.nn.Module) -> None:
    path = Path(ONNX_PATH)
    if path.exists():
        print(f"ONNX model already at {path}.")
        return
    print("Exporting model to ONNX…")
    dummy = torch.randn(1, 3, CROP_LR, CROP_LR)
    with torch.no_grad():
        torch.onnx.export(
            model.cpu().eval(), (dummy,), str(path),
            opset_version=17, input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        )
    print(f"  Saved ONNX to {path}")


def lower_mod(mod, target):
    """ONNX Relax IR → lowered call_tir IR (runs LegalizeOps + FuseOps)."""
    import tvm
    from tvm import relax
    return tvm.transform.Sequential([
        relax.transform.DecomposeOpsForInference(),
        relax.transform.CanonicalizeBindings(),
        relax.transform.LegalizeOps(),
        relax.transform.AnnotateTIROpPattern(),
        relax.transform.FoldConstant(),
        relax.transform.FuseOps(),
        relax.transform.FuseTIR(),
    ])(mod)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500,
                        help="Total tuning trials across all tasks")
    parser.add_argument("--trials-per-task", type=int, default=64,
                        help="Max trials per unique operator")
    parser.add_argument("--trials-per-iter", type=int, default=64,
                        help="Trials per tuning iteration (evolutionary batch size)")
    parser.add_argument("--strategy", default="replay-func",
                        choices=["evolutionary", "replay-func"],
                        help="Search strategy (replay-func is more stable for fused conv2d ops)")
    args = parser.parse_args()

    import tvm
    import onnx as onnx_lib
    from tvm.relax.frontend.onnx import from_onnx
    from tvm.s_tir import meta_schedule as ms

    config = load_config(CONFIG)
    model  = build_model(3, 3, 64, SCALE)
    ckpt   = torch.load(config["train"]["save_path"], map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    export_onnx_if_needed(model)

    target = tvm.target.Target(_CUDA_TARGET)

    print("Loading ONNX and converting to Relax IR…")
    onnx_model = onnx_lib.load(ONNX_PATH)
    mod = from_onnx(onnx_model, shape_dict={"input": [1, 3, CROP_LR, CROP_LR]},
                    keep_params_in_input=False)
    # Bind batch=1 so all TIR loops are static — MetaSchedule can't split symbolic vars
    mod = tvm.relax.transform.BindSymbolicVars({"batch": 1})(mod)

    print("Lowering to call_tir (LegalizeOps + FuseOps)…")
    mod_lowered = lower_mod(mod, target)

    tasks = ms.relax_integration.extract_tasks(mod_lowered, target, params=None)
    print(f"Found {len(tasks)} unique operator tasks to tune")
    for t in list(tasks)[:5]:
        print(f"  {t.task_name}")
    if len(tasks) > 5:
        print(f"  … and {len(tasks) - 5} more")

    Path(WORK_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\nTuning  max_trials_global={args.trials}  "
          f"max_trials_per_task={args.trials_per_task}  "
          f"num_trials_per_iter={args.trials_per_iter}  "
          f"strategy={args.strategy}")
    print(f"Work dir: {WORK_DIR}")
    print("Progress saved continuously to the work dir.\n")

    t0 = time.perf_counter()
    ms.relax_integration.tune_relax(
        mod=mod_lowered,
        params=None,
        target=target,
        work_dir=WORK_DIR,
        max_trials_global=args.trials,
        max_trials_per_task=args.trials_per_task,
        num_trials_per_iter=args.trials_per_iter,
        cost_model="xgb",
        strategy=args.strategy,
    )
    elapsed = time.perf_counter() - t0
    print(f"\nTuning finished in {elapsed/60:.1f} min")
    print(f"Database saved to {WORK_DIR}/")
    print("Run `python scripts/benchmark_tvm.py --mode tuned` to benchmark.")


if __name__ == "__main__":
    main()
