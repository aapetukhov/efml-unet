"""LSQ"""
from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

_QMIN = -127
_QMAX = 127


class _LsqStepSizeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, step_size, qmin, qmax, grad_scale):
        x_div_s = x / step_size
        x_hat = x_div_s.round().clamp(qmin, qmax)
        ctx.save_for_backward(x_div_s, x_hat, step_size)
        ctx.other = (qmin, qmax, grad_scale)
        ctx.step_size_shape = step_size.shape
        return x_hat * step_size

    @staticmethod
    def backward(ctx, grad_output):
        x_div_s, x_hat, step_size = ctx.saved_tensors
        qmin, qmax, grad_scale = ctx.other
        ss_shape = ctx.step_size_shape

        in_range = (x_div_s >= qmin) & (x_div_s <= qmax)
        grad_x = grad_output * in_range.float()

        grad_s_elem = torch.where(
            in_range, x_hat - x_div_s,
            torch.where(x_div_s < qmin,
                        torch.full_like(x_div_s, qmin),
                        torch.full_like(x_div_s, qmax)),
        )
        raw = grad_output * grad_s_elem
        reduce_dims = [d for d in range(raw.dim()) if d >= len(ss_shape) or ss_shape[d] == 1]
        grad_s = raw.sum(dim=reduce_dims, keepdim=True) if reduce_dims else raw
        grad_s = grad_s.reshape(ss_shape) * grad_scale

        return grad_x, grad_s, None, None, None


def _lsq_quantize(x, step_size, qmin, qmax, grad_scale):
    return _LsqStepSizeFunc.apply(x, step_size, qmin, qmax, grad_scale)


class LsqQuantizer(nn.Module):
    def __init__(self, n_bits=8, per_channel=True, n_channels=1,
                 symmetric=True, is_activation=False):
        super().__init__()
        self.n_bits = n_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.is_activation = is_activation

        if symmetric:
            self.qmin = -(2 ** (n_bits - 1)) + 1
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** n_bits - 1

        shape = (n_channels,) if per_channel else (1,)
        self.step_size = nn.Parameter(torch.ones(shape))
        self.register_buffer("_initialized", torch.tensor(False))
        self.register_buffer("_grad_scale_factor", torch.tensor(1.0))

    def _init_step_size(self, x):
        with torch.no_grad():
            if self.per_channel and x.dim() >= 2:
                reduce_dims = list(range(1, x.dim()))
                mean_abs = x.abs().mean(dim=reduce_dims)
            else:
                mean_abs = x.abs().mean().unsqueeze(0)

            init_val = (2.0 * mean_abs / math.sqrt(self.qmax)).clamp(min=1e-8)
            self.step_size.data.copy_(init_val.reshape(self.step_size.shape))
            self._initialized.fill_(True)

            n_elements = x[0].numel() if x.dim() >= 2 else x.numel()
            self._grad_scale_factor.fill_(1.0 / math.sqrt(n_elements * self.qmax))

    def forward(self, x):
        if not self._initialized:
            self._init_step_size(x)

        if self.per_channel and x.dim() == 4:
            s = self.step_size.view(-1, 1, 1, 1)
        elif self.per_channel and x.dim() == 2:
            s = self.step_size.view(-1, 1)
        else:
            s = self.step_size

        s = s.abs().clamp(min=1e-8)
        return _lsq_quantize(x, s, self.qmin, self.qmax, float(self._grad_scale_factor))

    def extra_repr(self):
        return f"bits={self.n_bits}, per_channel={self.per_channel}, qrange=[{self.qmin}, {self.qmax}]"


class LsqConv2d(nn.Module):
    def __init__(self, conv, n_bits_w=8, n_bits_a=8, quantize_activations=False):
        super().__init__()
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.padding_mode = conv.padding_mode

        self.weight = conv.weight
        self.bias = conv.bias

        dev = conv.weight.device
        self.w_quantizer = LsqQuantizer(
            n_bits=n_bits_w, per_channel=True,
            n_channels=conv.out_channels, symmetric=True,
        ).to(dev)

        self.quantize_activations = quantize_activations
        if quantize_activations:
            self.a_quantizer = LsqQuantizer(
                n_bits=n_bits_a, per_channel=False,
                n_channels=1, symmetric=False, is_activation=True,
            ).to(dev)

    def forward(self, x):
        if self.quantize_activations:
            x = self.a_quantizer(x)
        w_q = self.w_quantizer(self.weight)
        return F.conv2d(x, w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return (f"{self.in_channels}, {self.out_channels}, k={self.kernel_size}, "
                f"g={self.groups}, w_bits={self.w_quantizer.n_bits}")


class LsqLinear(nn.Module):
    def __init__(self, linear, n_bits_w=8, n_bits_a=8, quantize_activations=False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias

        dev = linear.weight.device
        self.w_quantizer = LsqQuantizer(
            n_bits=n_bits_w, per_channel=True,
            n_channels=linear.out_features, symmetric=True,
        ).to(dev)

        self.quantize_activations = quantize_activations
        if quantize_activations:
            self.a_quantizer = LsqQuantizer(
                n_bits=n_bits_a, per_channel=False,
                n_channels=1, symmetric=False, is_activation=True,
            ).to(dev)

    def forward(self, x):
        if self.quantize_activations:
            x = self.a_quantizer(x)
        w_q = self.w_quantizer(self.weight)
        return F.linear(x, w_q, self.bias)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, w_bits={self.w_quantizer.n_bits}"


def prepare_lsq(
    model, n_bits_w=8, n_bits_a=8, quantize_activations=False,
    skip_depthwise=False, skip_first_last=True,
):
    all_convs = []
    for name, module in model.named_modules():
        for attr_name, child in module.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                full_name = f"{name}.{attr_name}" if name else attr_name
                all_convs.append((name, attr_name, module, child, full_name))

    if not all_convs:
        log.warning("No Conv2d or Linear layers found")
        return model

    stem_names, head_names = set(), set()
    if skip_first_last:
        for _, _, _, child, full_name in all_convs:
            if isinstance(child, nn.Conv2d):
                if full_name.startswith("stem"):
                    stem_names.add(full_name)
                if full_name.startswith("head"):
                    head_names.add(full_name)

    n_replaced, n_skipped = 0, 0
    for idx, (parent_name, attr_name, parent_module, child, full_name) in enumerate(all_convs):
        if isinstance(child, nn.Conv2d):
            if skip_first_last and (full_name in stem_names or full_name in head_names):
                n_skipped += 1
                continue
            if skip_depthwise and child.groups == child.in_channels and child.groups > 1:
                n_skipped += 1
                continue
            setattr(parent_module, attr_name, LsqConv2d(child, n_bits_w, n_bits_a, quantize_activations))
            n_replaced += 1
        elif isinstance(child, nn.Linear):
            setattr(parent_module, attr_name, LsqLinear(child, n_bits_w, n_bits_a, quantize_activations))
            n_replaced += 1

    log.info(f"LSQ: replaced {n_replaced} layers, skipped {n_skipped}")
    return model


def _count_lsq_layers(model):
    counts = {"LsqConv2d": 0, "LsqLinear": 0, "Conv2d_fp": 0, "Linear_fp": 0}
    for m in model.modules():
        if isinstance(m, LsqConv2d):
            counts["LsqConv2d"] += 1
        elif isinstance(m, LsqLinear):
            counts["LsqLinear"] += 1
        elif isinstance(m, nn.Conv2d):
            counts["Conv2d_fp"] += 1
        elif isinstance(m, nn.Linear):
            counts["Linear_fp"] += 1
    return counts


def get_lsq_params(model):
    return [p for n, p in model.named_parameters() if "step_size" in n]


def get_weight_params(model):
    return [p for n, p in model.named_parameters() if "step_size" not in n]


def finalize_lsq(model: nn.Module, scale_dtype: torch.dtype = torch.float16) -> dict:
    """Convert fake-quantized model to INT8 state dict.

    scale_dtype controls precision of scales/biases (default: float16).
    """
    int8_state = {}
    layer_info = {}

    for name, module in model.named_modules():
        if isinstance(module, (LsqConv2d, LsqLinear)):
            w = module.weight.data
            s = module.w_quantizer.step_size.data.abs().clamp(min=1e-8)

            if w.dim() == 4:
                s_broad = s.view(-1, 1, 1, 1)
            elif w.dim() == 2:
                s_broad = s.view(-1, 1)
            else:
                s_broad = s

            w_int = (w.float() / s_broad.float()).round().clamp(_QMIN, _QMAX).to(torch.int8)

            int8_state[f"{name}.weight_int8"] = w_int
            int8_state[f"{name}.weight_scale"] = s.to(scale_dtype)

            if module.bias is not None:
                int8_state[f"{name}.bias"] = module.bias.data.to(scale_dtype)

            layer_info[name] = {
                "type": type(module).__name__,
                "shape": list(w.shape),
                "n_bits": module.w_quantizer.n_bits,
                "scale_shape": list(s.shape),
            }

    for name, param in model.named_parameters():
        is_lsq = any(name.startswith(ln) for ln in layer_info)
        if not is_lsq:
            int8_state[name] = param.data.to(scale_dtype)

    for name, buf in model.named_buffers():
        if "_initialized" not in name and "_grad_scale" not in name:
            is_lsq = any(name.startswith(ln) for ln in layer_info)
            if not is_lsq:
                int8_state[name] = buf

    return {"int8_state_dict": int8_state, "layer_info": layer_info}


def count_int8_model_size_mb(int8_result: dict) -> float:
    total_bytes = 0
    for key, tensor in int8_result["int8_state_dict"].items():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / (1024 * 1024)


def get_quant_params_json(model, skip_first_last=True):
    result = {"global": {}, "layers": {}}

    for m in model.modules():
        if isinstance(m, (LsqConv2d, LsqLinear)):
            q = m.w_quantizer
            result["global"] = {
                "n_bits_w": q.n_bits, "symmetric": q.symmetric,
                "qmin": q.qmin, "qmax": q.qmax,
            }
            break

    for name, module in model.named_modules():
        if isinstance(module, LsqConv2d):
            q = module.w_quantizer
            s = q.step_size.data.abs().clamp(min=1e-8)
            result["layers"][name] = {
                "type": "LsqConv2d",
                "weight_shape": list(module.weight.shape),
                "scale_shape": list(s.shape),
                "scales": s.cpu().tolist(),
                "granularity": "per_channel" if q.per_channel else "per_tensor",
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": list(module.kernel_size),
                "groups": module.groups,
                "quantized": True,
            }
        elif isinstance(module, LsqLinear):
            q = module.w_quantizer
            s = q.step_size.data.abs().clamp(min=1e-8)
            result["layers"][name] = {
                "type": "LsqLinear",
                "weight_shape": list(module.weight.shape),
                "scale_shape": list(s.shape),
                "scales": s.cpu().tolist(),
                "granularity": "per_channel" if q.per_channel else "per_tensor",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "quantized": True,
            }
        elif isinstance(module, nn.Conv2d):
            result["layers"][name] = {
                "type": "Conv2d",
                "weight_shape": list(module.weight.shape),
                "quantized": False,
                "skip_reason": "stem/head layer" if skip_first_last else "not replaced",
            }
        elif isinstance(module, nn.Linear):
            result["layers"][name] = {
                "type": "Linear",
                "weight_shape": list(module.weight.shape),
                "quantized": False, "skip_reason": "not replaced",
            }

    return result


def lsq_summary(model):
    lines = ["LSQ Summary", "=" * 60]
    counts = _count_lsq_layers(model)
    lines.append(f"  LsqConv2d: {counts['LsqConv2d']}  LsqLinear: {counts['LsqLinear']}")
    lines.append(f"  FP Conv2d: {counts['Conv2d_fp']}  FP Linear: {counts['Linear_fp']}")

    total_params = sum(p.numel() for p in model.parameters())
    lsq_params = sum(p.numel() for n, p in model.named_parameters() if "step_size" in n)
    lines.append(f"  Total params: {total_params:,}  step_size params: {lsq_params:,}")
    lines.append("=" * 60)

    for name, module in model.named_modules():
        if isinstance(module, LsqConv2d):
            s = module.w_quantizer.step_size.data
            lines.append(f"  {name}: Conv2d({module.in_channels}→{module.out_channels}, "
                         f"k={module.kernel_size}, g={module.groups}) "
                         f"s_mean={s.mean():.6f}")
        elif isinstance(module, LsqLinear):
            s = module.w_quantizer.step_size.data
            lines.append(f"  {name}: Linear({module.in_features}→{module.out_features}) "
                         f"s_mean={s.mean():.6f}")

    return "\n".join(lines)
