"""Small, reproducible forward profiling helpers."""

from __future__ import annotations

import copy
import platform
import statistics
import time
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .schema import ProfileRecord


class BF16Conv2d(nn.Conv2d):
    """Inference-only BF16 Conv2d with an FP32 output boundary."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cpu":
            raise RuntimeError("the reference BF16 Conv2d candidate currently supports CPU only")
        output = F.conv2d(
            input.to(torch.bfloat16),
            self.weight.to(torch.bfloat16),
            None if self.bias is None else self.bias.to(torch.bfloat16),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return output.to(input.dtype)


def make_bf16_conv_candidate(model: nn.Module) -> tuple[nn.Module, int]:
    """Clone ``model`` and replace supported Conv2d modules with BF16 candidates."""

    candidate = copy.deepcopy(model)
    replaced = 0
    for parent in candidate.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Conv2d) and not isinstance(child, BF16Conv2d):
                replacement = BF16Conv2d(
                    child.in_channels,
                    child.out_channels,
                    child.kernel_size,
                    child.stride,
                    child.padding,
                    child.dilation,
                    child.groups,
                    child.bias is not None,
                    child.padding_mode,
                )
                replacement.load_state_dict(child.state_dict())
                replacement.train(child.training)
                setattr(parent, name, replacement)
                replaced += 1
    return candidate, replaced


@dataclass(frozen=True)
class ForwardProfile:
    reference: ProfileRecord
    candidate: ProfileRecord
    reference_output: torch.Tensor
    candidate_output: torch.Tensor


def profile_forward_pair(
    model: nn.Module,
    sample: torch.Tensor,
    *,
    candidate: nn.Module | None = None,
    candidate_name: str = "bf16",
    warmup: int = 2,
    repetitions: int = 5,
    data_range: float = 1.0,
) -> ForwardProfile:
    """Measure FP32 and an optional whole-model candidate on paired inputs."""

    if sample.device.type != "cpu":
        raise ValueError("the initial reference profiler supports CPU samples only")
    if repetitions < 1 or warmup < 0:
        raise ValueError("warmup must be non-negative and repetitions must be positive")
    reference_model = copy.deepcopy(model).cpu().eval()
    candidate_model = copy.deepcopy(candidate if candidate is not None else model).cpu().eval()
    context = {
        "model_class": type(model).__qualname__,
        "device": "cpu",
        "dtype": str(sample.dtype),
        "input_shape": list(sample.shape),
        "cpu": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "data_range": float(data_range),
    }

    with torch.inference_mode():
        reference_output = _extract_x_hat(reference_model(sample))
        candidate_output = _extract_x_hat(candidate_model(sample))
        reference_times = _measure(reference_model, sample, warmup, repetitions)
        candidate_times = _measure(candidate_model, sample, warmup, repetitions)

    psnr_reference = _psnr(sample, reference_output, data_range)
    psnr_candidate = _psnr(sample, candidate_output, data_range)
    psnr_drop = max(0.0, psnr_reference - psnr_candidate)
    reference_median = statistics.median(reference_times)
    candidate_median = statistics.median(candidate_times)
    shared_context = dict(context)
    shared_context["warmup"] = warmup
    shared_context["measurement_scope"] = "forward"
    return ForwardProfile(
        reference=ProfileRecord(
            kind="operator", target="model", candidate="fp32", context=shared_context,
            latency_ms=reference_median, latency_std_ms=statistics.pstdev(reference_times),
            repetitions=repetitions, warmup=warmup, quality_metric="psnr", quality_unit="dB",
            measurement_scope="forward",
        ),
        candidate=ProfileRecord(
            kind="operator", target="model", candidate=candidate_name, context=shared_context,
            latency_ms=candidate_median, latency_std_ms=statistics.pstdev(candidate_times),
            psnr_drop_db=psnr_drop, repetitions=repetitions, warmup=warmup,
            quality_metric="psnr_drop", quality_unit="dB", measurement_scope="forward",
        ),
        reference_output=reference_output,
        candidate_output=candidate_output,
    )


def _measure(model: nn.Module, sample: torch.Tensor, warmup: int, repetitions: int) -> list[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        times: list[float] = []
        for _ in range(repetitions):
            start = time.perf_counter()
            model(sample)
            times.append((time.perf_counter() - start) * 1000.0)
    return times


def _extract_x_hat(output: Any) -> torch.Tensor:
    if isinstance(output, Mapping) and isinstance(output.get("x_hat"), torch.Tensor):
        return output["x_hat"]
    if isinstance(output, torch.Tensor):
        return output
    raise ValueError("forward output must be a Tensor or mapping with a Tensor x_hat")


def _psnr(reference_input: torch.Tensor, reconstruction: torch.Tensor, data_range: float) -> float:
    if reference_input.shape != reconstruction.shape:
        raise ValueError("reference and reconstruction shapes must match for PSNR")
    mse = float(torch.mean((reference_input.float() - reconstruction.float()) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item()
