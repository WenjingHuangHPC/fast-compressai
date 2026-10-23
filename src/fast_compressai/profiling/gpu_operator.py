"""CUDA event profiling for individual Balle operator candidates."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..profiling.schema import ProfileRecord


@dataclass(frozen=True)
class GpuOperatorProfile:
    reference_record: ProfileRecord
    candidate_record: ProfileRecord
    reference_output: torch.Tensor
    candidate_output: torch.Tensor


def profile_gpu_operator_pair(
    reference: Callable[[torch.Tensor], torch.Tensor],
    candidate: Callable[[torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
    *,
    target: str,
    candidate_id: str,
    precision: str = "bf16",
    warmup: int = 5,
    repetitions: int = 30,
    data_range: float = 1.0,
) -> GpuOperatorProfile:
    if sample.device.type != "cuda":
        raise ValueError("GPU operator profiling requires a CUDA tensor")
    if warmup < 0 or repetitions < 2:
        raise ValueError("warmup must be non-negative and repetitions must be at least two")
    if data_range <= 0:
        raise ValueError("data_range must be positive")

    with torch.inference_mode():
        reference_output = reference(sample)
        candidate_output = candidate(sample)
        if reference_output.shape != candidate_output.shape:
            raise ValueError("reference and candidate output shapes differ")
        if not torch.isfinite(reference_output).all() or not torch.isfinite(candidate_output).all():
            raise ValueError("operator candidate produced non-finite output")
        reference_times = _measure(reference, sample, warmup, repetitions)
        candidate_times = _measure(candidate, sample, warmup, repetitions)

    delta = (candidate_output.float() - reference_output.float())
    mse = float(delta.square().mean().item())
    psnr_drop = 0.0 if mse == 0 else max(0.0, 10.0 * torch.log10(torch.tensor(data_range**2 / mse)).item())
    context = {
        "device": str(sample.device),
        "gpu_name": torch.cuda.get_device_name(sample.device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "input_shape": list(sample.shape),
        "input_dtype": str(sample.dtype),
        "input_range": [float(sample.float().amin().item()), float(sample.float().amax().item())],
        "candidate_id": candidate_id,
        "measurement_scope": "operator",
        "candidate_precision": precision,
        "data_range": float(data_range),
    }
    return GpuOperatorProfile(
        reference_record=ProfileRecord(
            kind="operator", target=target, candidate="fp32/pytorch-cudnn",
            context=context, latency_ms=statistics.median(reference_times),
            latency_std_ms=statistics.pstdev(reference_times), repetitions=repetitions,
            warmup=warmup, measurement_scope="operator", quality_metric="max_abs_error",
            quality_unit="input_dtype",
        ),
        candidate_record=ProfileRecord(
            kind="operator", target=target, candidate=candidate_id,
            context=context, latency_ms=statistics.median(candidate_times),
            latency_std_ms=statistics.pstdev(candidate_times), psnr_drop_db=psnr_drop,
            repetitions=repetitions, warmup=warmup, measurement_scope="operator",
            quality_metric="psnr_drop", quality_unit="dB",
        ),
        reference_output=reference_output,
        candidate_output=candidate_output,
    )


def _measure(
    operation: Callable[[torch.Tensor], torch.Tensor],
    sample: torch.Tensor,
    warmup: int,
    repetitions: int,
) -> list[float]:
    for _ in range(warmup):
        operation(sample)
    torch.cuda.synchronize(sample.device)
    results: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation(sample)
        end.record()
        end.synchronize()
        results.append(float(start.elapsed_time(end)))
    return results
