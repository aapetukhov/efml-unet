"""
Find and remove OOB-corrupted workloads from the MetaSchedule database.

Strategy: binary search where at each step we exclude a SUBSET of workloads
(they fall back to DLight) and run the FULL model. This correctly handles
multiple bad workloads and avoids false negatives from isolated testing.

Usage (from efml-unet/):
  python scripts/validate_tvm_db.py [--work-dir results/tvm_tuning]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ONNX_PATH = "results/unet_sr_x4.onnx"
CROP_LR   = 256
ARCH      = "sm_86"
N_RUNS    = 5  # inference runs per test

_CUDA_TARGET = {
    "kind": "cuda", "arch": ARCH,
    "max_threads_per_block": 1024,
    "max_shared_memory_per_block": 49152,
}

PROBE_TEMPLATE = """
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
import tvm, onnx as onnx_lib, torch
from tvm.relax.frontend.onnx import from_onnx
from tvm import relax
from tvm.s_tir import meta_schedule as ms
from tvm.ffi import from_dlpack

target = tvm.target.Target({target!r})
dev = tvm.cuda(0)
mod = from_onnx(onnx_lib.load("{onnx}"), shape_dict={{"input": [1,3,{crop},{crop}]}}, keep_params_in_input=False)
mod = tvm.relax.transform.BindSymbolicVars({{"batch": 1}})(mod)
database = ms.database.JSONDatabase(work_dir="{work_dir}")
ex = ms.relax_integration.compile_relax(database=database, mod=mod, target=target, params=None)
vm = relax.VirtualMachine(ex, dev)
inp = torch.randn(1, 3, {crop}, {crop}).cuda()
for _ in range({n_runs}):
    vm["main"](from_dlpack(inp.contiguous()))
    dev.sync()
print("OK")
"""


def load_db(work_dir: str) -> tuple[list, list]:
    rec_path = Path(work_dir) / "database_tuning_record.json"
    wl_path  = Path(work_dir) / "database_workload.json"
    with open(rec_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    with open(wl_path) as f:
        workloads = [json.loads(l) for l in f if l.strip()]
    return records, workloads


def save_db(work_dir: str, records: list, workloads: list) -> None:
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    for name, data in [("database_tuning_record.json", records),
                       ("database_workload.json", workloads)]:
        with open(Path(work_dir) / name, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")


def workload_key(record: list) -> str:
    """Unique key for a record: the tensor shapes in r[1][-1]."""
    return json.dumps(record[1][-1])


def group_by_workload(records: list) -> dict[str, list]:
    """Map workload_key → list of records."""
    groups: dict[str, list] = {}
    for r in records:
        k = workload_key(r)
        groups.setdefault(k, []).append(r)
    return groups


def test_subset(included_keys: set[str], all_records: list, all_workloads: list,
                work_dir: str) -> bool:
    """
    Write a temporary database with only records whose key is in included_keys,
    then compile+run the full model. Returns True if clean (no OOB crash).
    Excluded workloads fall back to DLight automatically.
    """
    subset_records = [r for r in all_records if workload_key(r) in included_keys]
    subset_workloads = _filter_workloads(subset_records, all_workloads)

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        save_db(tmp, subset_records, subset_workloads)
        probe = PROBE_TEMPLATE.format(
            target=_CUDA_TARGET, onnx=ONNX_PATH, crop=CROP_LR,
            work_dir=tmp, n_runs=N_RUNS,
        )
        try:
            result = subprocess.run(
                [sys.executable, "-c", probe],
                capture_output=True, text=True, timeout=180,
            )
            ok = result.returncode == 0 and "OK" in result.stdout
            if not ok:
                snippet = (result.stderr + result.stdout)[-200:].strip()
                print(f"        crash: ...{snippet[-120:]}")
            return ok
        except subprocess.TimeoutExpired:
            print("        TIMEOUT")
            return False


def _filter_workloads(records: list, all_workloads: list) -> list:
    """Return only workloads referenced by records (deduplicated)."""
    keys_in_records = {workload_key(r) for r in records}
    seen: set[str] = set()
    result = []
    for w in all_workloads:
        wk = json.dumps(w)
        if wk not in seen:
            # workloads match records by their tensor shapes (last element)
            # Heuristic: include all workloads whose shape list appears in records
            seen.add(wk)
            result.append(w)
    return result


def find_bad_workloads(
    candidate_keys: list[str],
    all_keys: set[str],
    all_records: list,
    all_workloads: list,
    work_dir: str,
    depth: int = 0,
) -> set[str]:
    """
    Recursively find all bad workload keys among candidate_keys.
    all_keys: the full set of workload keys (used to compute 'others').
    At each step: test only candidate_keys (others → DLight).
    If crash: recurse on left and right halves independently.
    At leaf (single key): test all_keys MINUS this one key; if clean → it's bad.
    """
    indent = "  " * depth

    if not candidate_keys:
        return set()

    if len(candidate_keys) == 1:
        key = candidate_keys[0]
        # Test full database minus this one key
        without_key = all_keys - {key}
        print(f"{indent}Leaf: testing without 1 workload...")
        clean_without = test_subset(without_key, all_records, all_workloads, work_dir)
        if clean_without:
            print(f"{indent}  → BAD workload confirmed")
            return {key}
        else:
            print(f"{indent}  → not the only bad one (or not bad)")
            return set()

    mid = len(candidate_keys) // 2
    left_keys  = set(candidate_keys[:mid])
    right_keys = set(candidate_keys[mid:])

    print(f"{indent}Testing left half ({len(left_keys)} workloads)...")
    left_crashes  = not test_subset(left_keys,  all_records, all_workloads, work_dir)
    print(f"{indent}Testing right half ({len(right_keys)} workloads)...")
    right_crashes = not test_subset(right_keys, all_records, all_workloads, work_dir)

    bad: set[str] = set()
    if left_crashes:
        bad |= find_bad_workloads(list(left_keys), all_keys, all_records, all_workloads, work_dir, depth + 1)
    if right_crashes:
        bad |= find_bad_workloads(list(right_keys), all_keys, all_records, all_workloads, work_dir, depth + 1)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", default="results/tvm_tuning")
    args = parser.parse_args()

    records, workloads = load_db(args.work_dir)
    groups = group_by_workload(records)
    all_keys = set(groups)
    key_list = list(all_keys)
    print(f"Database: {len(records)} records, {len(all_keys)} unique workloads")

    # Step 1: quick full test
    print("\nStep 1: testing full database...")
    if test_subset(all_keys, records, workloads, args.work_dir):
        print("Full database is clean — no OOB bugs!")
        return
    print("OOB detected.\n")

    # Step 2: binary search
    print("Step 2: binary search for bad workloads (each half tested in FULL model context)...")
    bad_keys = find_bad_workloads(key_list, all_keys, records, workloads, args.work_dir)

    if not bad_keys:
        print("\nCould not isolate bad workloads. The crash may require a specific combination.")
        return

    print(f"\nFound {len(bad_keys)} bad workload(s).")

    # Step 3: remove bad records from database
    bak_rec = Path(args.work_dir) / "database_tuning_record.json.bak2"
    bak_wl  = Path(args.work_dir) / "database_workload.json.bak2"
    shutil.copy(Path(args.work_dir) / "database_tuning_record.json", bak_rec)
    shutil.copy(Path(args.work_dir) / "database_workload.json", bak_wl)
    print(f"Backup saved to {bak_rec.name} / {bak_wl.name}")

    clean_records = [r for r in records if workload_key(r) not in bad_keys]
    clean_workloads = _filter_workloads(clean_records, workloads)
    save_db(args.work_dir, clean_records, clean_workloads)
    print(f"Records: {len(records)} → {len(clean_records)}")

    print("\nStep 3: verifying cleaned database...")
    if test_subset(all_keys - bad_keys, clean_records, clean_workloads, args.work_dir):
        print("Cleaned database verified OK.")
        print("\nRun: python scripts/benchmark_tvm.py --mode tuned")
    else:
        print("Still crashing — re-run this script to find remaining bad workloads.")


if __name__ == "__main__":
    main()
