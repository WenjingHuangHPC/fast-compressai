"""GA-ANS P_ch sweep measurements for final mixed-precision models."""

from __future__ import annotations

from typing import Sequence

from .rd import CalibrationData, measure_factorized_warpdiv_rd
from ..policies.p_ch import choose_p_ch, legal_p_candidates


def default_p_candidates(channel_count: int) -> tuple[int, ...]:
    values = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
    return legal_p_candidates(channel_count, values)


def sweep_p_ch(
    model,
    calibration: CalibrationData,
    *,
    model_name: str,
    candidates: Sequence[int] | None = None,
    warmup: int = 1,
    repetitions: int = 3,
    relative_tolerance: float = 0.10,
    plateau_delta: float = 0.05,
    plateau_patience: int = 2,
):
    """Measure P values on one fixed, already-selected transform model.

    Balance is based on coder compression time versus model forward time, which
    matches the CAIEC inference/coding co-design metric. Decompression remains
    in each result for auditability but does not drive P selection.
    """
    channel_count = int(model.entropy_bottleneck.channels)
    p_values = legal_p_candidates(channel_count, candidates or default_p_candidates(channel_count))
    measurements = []
    for p_ch in p_values:
        measurement = measure_factorized_warpdiv_rd(
            model, calibration, p_ch=p_ch, model_name=model_name,
            reference_bpp=1.0, reference_psnr_db=0.0,
            warmup=warmup, repetitions=repetitions,
        )
        context = measurement.record.context
        forward_ms = float(context["forward_latency_ms_median"])
        compress_ms = float(context["compress_latency_ms_median"])
        decompress_ms = float(context["decompress_latency_ms_median"])
        compress_inference_ms = float(context["compress_inference_ms_median"])
        compress_codec_ms = float(context["compress_codec_ms_median"])
        decompress_codec_ms = float(context["decompress_codec_ms_median"])
        decompress_inference_ms = float(context["decompress_inference_ms_median"])
        inference_total_ms = compress_inference_ms + decompress_inference_ms
        codec_total_ms = compress_codec_ms + decompress_codec_ms
        measurements.append({
            "p_ch": p_ch,
            "bpp": measurement.bpp,
            "psnr_db": measurement.psnr_db,
            "forward_latency_ms": forward_ms,
            "compress_latency_ms": compress_ms,
            "decompress_latency_ms": decompress_ms,
            "compress_inference_ms": compress_inference_ms,
            "compress_codec_ms": compress_codec_ms,
            "decompress_codec_ms": decompress_codec_ms,
            "decompress_inference_ms": decompress_inference_ms,
            "inference_total_ms": inference_total_ms,
            "codec_total_ms": codec_total_ms,
            "codec_to_inference_ratio": codec_total_ms / max(inference_total_ms, 1e-12),
            "rd_cost": measurement.rd_cost,
            "sample_count": measurement.sample_count,
            "warmup": warmup,
            "repetitions_per_sample": repetitions,
            "payload_semantics": "device_local_warp_div_packed_payload",
        })
    selection = choose_p_ch(
        measurements,
        relative_tolerance=relative_tolerance,
        plateau_delta=plateau_delta,
        plateau_patience=plateau_patience,
    )
    for item in measurements:
        ratio = item["codec_to_inference_ratio"]
        item["relative_time_error"] = abs(ratio - 1.0)
        item["passed_balance"] = item["relative_time_error"] <= relative_tolerance
        item["selected"] = item["p_ch"] == selection.selected_p_ch
    return selection, tuple(measurements)
