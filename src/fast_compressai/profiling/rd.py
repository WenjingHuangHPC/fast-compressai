"""Full FactorizedPrior + WarpDiv RD measurements for CAIEC selection."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from ..coders.factorized import FactorizedWarpDivCodec
from ..profiling.schema import ProfileRecord
from ..policies.caiec_precision import rd_cost


def calibration_fingerprint(samples: Sequence[torch.Tensor], dataset_name: str) -> str:
    import hashlib
    digest = hashlib.sha256(dataset_name.encode("utf-8"))
    for sample in samples:
        value = sample.detach().contiguous().cpu()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationData:
    samples: tuple[torch.Tensor, ...]
    raw_targets: tuple[torch.Tensor, ...]
    dataset_name: str
    dataset_fingerprint: str
    normalization: str
    denormalization: str
    data_range: float

    def __post_init__(self) -> None:
        if not self.samples or len(self.samples) != len(self.raw_targets):
            raise ValueError("calibration samples and raw targets must be non-empty and paired")
        if not self.dataset_name or not self.dataset_fingerprint:
            raise ValueError("calibration dataset identity is required")
        if self.normalization != "divide_by_255" or self.denormalization != "multiply_by_255":
            raise ValueError("unsupported calibration normalization contract")
        if self.data_range != 255.0:
            raise ValueError("Kodak calibration PSNR data_range must be 255")
        for sample, target in zip(self.samples, self.raw_targets):
            if sample.shape != target.shape or sample.ndim != 4:
                raise ValueError("calibration samples and targets must share NCHW shape")
            if sample.device.type != "cuda" or target.device.type != "cuda":
                raise ValueError("Kodak calibration tensors must already be on CUDA")


def load_kodak_f32(path, device: torch.device | str, dataset_name: str = "kodak") -> CalibrationData:
    """External data-loader helper; accelerate() itself never reads files."""
    from pathlib import Path
    import hashlib
    path = Path(path)
    expected_values = 24 * 3 * 512 * 768
    expected_bytes = expected_values * 4
    if path.stat().st_size != expected_bytes:
        raise ValueError(f"Kodak f32 file size mismatch: expected {expected_bytes} bytes")
    raw_array = __import__("numpy").fromfile(path, dtype="<f4")
    if raw_array.size != expected_values or not __import__("numpy").isfinite(raw_array).all():
        raise ValueError("Kodak f32 must contain exactly 24x3x512x768 finite float32 values")
    raw = torch.from_numpy(raw_array.copy()).reshape(24, 3, 512, 768)
    if raw.min().item() < 0.0 or raw.max().item() > 255.0:
        raise ValueError("Kodak raw pixels must be in [0,255]")
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
    raw_targets = tuple(sample.unsqueeze(0).to(device) for sample in raw)
    samples = tuple((target / 255.0).contiguous() for target in raw_targets)
    return CalibrationData(
        samples=samples,
        raw_targets=raw_targets,
        dataset_name=dataset_name,
        dataset_fingerprint=fingerprint,
        normalization="divide_by_255",
        denormalization="multiply_by_255",
        data_range=255.0,
    )


@dataclass(frozen=True)
class FactorizedRDMeasurement:
    bpp: float
    psnr_db: float
    rd_cost: float
    forward_latency_ms: float
    compress_latency_ms: float
    decompress_latency_ms: float
    compress_inference_ms: float = 0.0
    compress_codec_ms: float = 0.0
    decompress_codec_ms: float = 0.0
    decompress_inference_ms: float = 0.0
    sample_count: int = 0
    record: ProfileRecord | None = None


def measure_factorized_warpdiv_rd(
    model: nn.Module,
    calibration: CalibrationData,
    *,
    p_ch: int,
    model_name: str,
    reference_bpp: float,
    reference_psnr_db: float,
    warmup: int = 0,
    repetitions: int = 1,
) -> FactorizedRDMeasurement:
    samples = calibration.samples
    raw_targets = calibration.raw_targets
    dataset_name = calibration.dataset_name
    if not samples:
        raise ValueError("calibration_data must contain at least one sample")
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be non-negative and repetitions positive")
    device = next(model.parameters()).device
    if device.type != "cuda":
        raise ValueError("CAIEC FactorizedPrior RD profiling requires CUDA")
    codec = FactorizedWarpDivCodec(model, p_ch)
    forward_times: list[float] = []
    compress_times: list[float] = []
    decompress_times: list[float] = []
    compress_inference_times: list[float] = []
    compress_codec_times: list[float] = []
    decompress_codec_times: list[float] = []
    decompress_inference_times: list[float] = []
    total_bits = 0
    total_pixels = 0
    squared_error = 0.0
    value_count = 0
    with torch.inference_mode():
        for sample, raw_target in zip(samples, raw_targets):
            if sample.device != device or raw_target.device != device:
                raise ValueError("calibration samples/targets must be on the target CUDA device")
            if sample.shape[0] != 1:
                raise ValueError("calibration samples must contain one image each")
            for _ in range(warmup):
                warm_stream = codec.compress(sample)
                codec.decompress(warm_stream["strings"], warm_stream["shape"])
            for _ in range(repetitions):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                model(sample)
                torch.cuda.synchronize(device)
                forward_times.append((time.perf_counter() - start) * 1000.0)
                timing: dict[str, float] = {}
                start = time.perf_counter()
                encoded = codec.compress(sample, timing=timing)
                torch.cuda.synchronize(device)
                compress_times.append((time.perf_counter() - start) * 1000.0)
                compress_inference_times.append(timing["compress_inference_ms"])
                compress_codec_times.append(timing["compress_codec_ms"])
                timing = {}
                start = time.perf_counter()
                decoded = codec.decompress(encoded["strings"], encoded["shape"], timing=timing)["x_hat"]
                torch.cuda.synchronize(device)
                decompress_times.append((time.perf_counter() - start) * 1000.0)
                decompress_codec_times.append(timing["decompress_codec_ms"])
                decompress_inference_times.append(timing["decompress_inference_ms"])
                payload = encoded["strings"][0]
                total_bits += payload.packed.numel() * 8
                total_pixels += sample.numel()
                decoded_raw = decoded.float() * 255.0
                if decoded_raw.shape != raw_target.shape:
                    raise ValueError("decoded output shape differs from raw calibration target")
                squared_error += float((decoded_raw - raw_target.float()).square().sum().item())
                value_count += decoded.numel()
    bpp = total_bits / total_pixels
    mse = squared_error / value_count
    # Keep profile records finite.  The sensitivity CSV convention uses zero
    # for an exact reconstruction rather than serializing an infinite PSNR.
    psnr = 0.0 if mse == 0 else 20.0 * torch.log10(torch.tensor(255.0)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()
    if not math.isfinite(psnr):
        psnr = 0.0
    score = rd_cost(bpp, reference_bpp, psnr, reference_psnr_db)
    context = {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_fingerprint": calibration.dataset_fingerprint,
        "layout": "NCHW",
        "raw_range": [0.0, 255.0],
        "model_normalization": calibration.normalization,
        "reconstruction_denormalization": calibration.denormalization,
        "data_range": calibration.data_range,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "coder": "warp_div",
        "p_ch": p_ch,
        "sample_count": len(samples),
        "sample_shapes": [list(sample.shape) for sample in samples],
        "warmup": warmup,
        "repetitions_per_sample": repetitions,
        "forward_latency_ms_median": statistics.median(forward_times),
        "compress_latency_ms_median": statistics.median(compress_times),
        "decompress_latency_ms_median": statistics.median(decompress_times),
        "compress_inference_ms_median": statistics.median(compress_inference_times),
        "compress_codec_ms_median": statistics.median(compress_codec_times),
        "decompress_codec_ms_median": statistics.median(decompress_codec_times),
        "decompress_inference_ms_median": statistics.median(decompress_inference_times),
        "inference_total_ms_median": statistics.median(compress_inference_times) + statistics.median(decompress_inference_times),
        "codec_total_ms_median": statistics.median(compress_codec_times) + statistics.median(decompress_codec_times),
        "quality_metric": "caiec_rd_cost",
        "quality_unit": "epsilon",
    }
    total_latency = statistics.median(compress_times) + statistics.median(decompress_times)
    record = ProfileRecord(
        kind="coder", target=model_name, candidate=f"warp_div/p_ch={p_ch}",
        context=context, latency_ms=total_latency,
        repetitions=len(compress_times), warmup=warmup,
        measurement_scope="compress", quality_metric="caiec_rd_cost",
        quality_unit="epsilon", bpp=bpp, psnr_db=psnr, rd_cost=score,
        provenance="measured",
    )
    return FactorizedRDMeasurement(
        bpp=bpp, psnr_db=psnr, rd_cost=score,
        forward_latency_ms=statistics.median(forward_times),
        compress_latency_ms=statistics.median(compress_times),
        decompress_latency_ms=statistics.median(decompress_times),
        compress_inference_ms=statistics.median(compress_inference_times),
        compress_codec_ms=statistics.median(compress_codec_times),
        decompress_codec_ms=statistics.median(decompress_codec_times),
        decompress_inference_ms=statistics.median(decompress_inference_times),
        sample_count=len(samples), record=record,
    )
