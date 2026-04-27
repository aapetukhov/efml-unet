"""LSQ quantization SRUNetHeavy."""
from __future__ import annotations

import copy
import io
import logging
from typing import Optional

import torch
import torch.nn as nn

from src.acceleration.lsq import (
    LsqConv2d,
    LsqLinear,
    LsqQuantizer,
    count_int8_model_size_mb,
    finalize_lsq,
    get_lsq_params,
    get_quant_params_json,
    get_weight_params,
    lsq_summary,
    prepare_lsq,
)

log = logging.getLogger(__name__)


def count_model_size_mb(model: nn.Module) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 * 1024)


def apply_lsq_ptq(
    model: nn.Module, calibration_loader, device: str,
    n_bits_w: int = 8, n_bits_a: int = 8,
    quantize_activations: bool = False,
    skip_depthwise: bool = False,
    skip_first_last: bool = True,
    n_calibration_batches: int = 32,
) -> tuple[nn.Module, dict]:
    log.info("LSQ PTQ: inserting fake-quant layers...")
    prepare_lsq(
        model, n_bits_w=n_bits_w, n_bits_a=n_bits_a,
        quantize_activations=quantize_activations,
        skip_depthwise=skip_depthwise,
        skip_first_last=skip_first_last,
    )

    log.info(f"LSQ PTQ: calibrating with {n_calibration_batches} batches")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(calibration_loader):
            if i >= n_calibration_batches:
                break
            model(batch["lr"].to(device))
            if (i + 1) % 10 == 0:
                log.info(f"  Calibrated {i + 1}/{n_calibration_batches} batches")

    log.info("LSQ PTQ: calibration done")
    log.info(lsq_summary(model))

    int8_result = finalize_lsq(model)
    int8_size = count_int8_model_size_mb(int8_result)
    log.info(f"LSQ PTQ: INT8 model size = {int8_size:.2f} MB")

    return model, int8_result


def prepare_lsq_qat(
    model: nn.Module,
    n_bits_w: int = 8, n_bits_a: int = 8,
    quantize_activations: bool = False,
    skip_depthwise: bool = False,
    skip_first_last: bool = True,
) -> nn.Module:
    log.info("LSQ QAT: inserting fake-quant layers...")
    prepare_lsq(
        model, n_bits_w=n_bits_w, n_bits_a=n_bits_a,
        quantize_activations=quantize_activations,
        skip_depthwise=skip_depthwise,
        skip_first_last=skip_first_last,
    )
    log.info(lsq_summary(model))
    return model


def finalize_lsq_qat(model: nn.Module) -> dict:
    int8_result = finalize_lsq(model)
    int8_size = count_int8_model_size_mb(int8_result)
    log.info(f"LSQ QAT finalized: INT8 model size = {int8_size:.2f} MB")
    return int8_result


def compile_for_inference(model: nn.Module, mode: str = "max-autotune") -> nn.Module:
    compiled = torch.compile(model, mode=mode)
    log.info(f"Model compiled with mode='{mode}'")
    return compiled


def print_size_comparison(fp32_model: nn.Module, int8_result: dict) -> dict:
    fp32_mb = count_model_size_mb(fp32_model)
    int8_mb = count_int8_model_size_mb(int8_result)
    ratio = fp32_mb / max(int8_mb, 1e-6)
    log.info(f"FP32: {fp32_mb:.2f} MB | INT8: {int8_mb:.2f} MB | {ratio:.2f}x")
    return {"fp32_mb": round(fp32_mb, 2), "int8_mb": round(int8_mb, 2), "compression_ratio": round(ratio, 2)}
