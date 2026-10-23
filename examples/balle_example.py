"""Run a CompressAI Balle model with Fast-CompressAI.

Prerequisites:
    - install CompressAI and Fast-CompressAI
    - build Fast-CompressAI's optional CUDA extension
    - run on a CUDA device supported by that extension

Examples:
    python examples/balle_example.py \\
        --image path/to/image.png \\
        --weights /hwj/data/bmshj2018-factorized/bmshj2018-factorized-1.pth \\
        --p-ch 4

    python examples/balle_example.py \\
        --kodak /hwj/data/caiec_test_data/Kodak_24x3x512x768.f32 \\
        --weights /hwj/data/bmshj2018-factorized/bmshj2018-factorized-1.pth \\
        --precision-budget 1.0 \\
        --p-ch 4
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image
from torchvision.transforms import functional as TF
from compressai.zoo import bmshj2018_factorized

import fast_compressai


def load_kodak(path: Path, device: torch.device) -> fast_compressai.CalibrationData:
    return fast_compressai.load_kodak_f32(path, device, dataset_name="kodak")


def load_image(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        tensor = TF.to_tensor(image).unsqueeze(0)
    height, width = tensor.shape[-2:]
    height_pad = (-height) % 16
    width_pad = (-width) % 16
    if height_pad or width_pad:
        tensor = torch.nn.functional.pad(tensor, (0, width_pad, 0, height_pad), mode="replicate")
    return tensor.to(device)


def _psnr_db(reconstruction: torch.Tensor, target: torch.Tensor, data_range: float) -> float:
    mse = float((reconstruction.float() - target.float()).square().mean().item())
    if mse == 0.0:
        return 0.0
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


def _payload_elements(payload: Any) -> int:
    if hasattr(payload, "packed"):
        return int(payload.packed.numel())
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    if isinstance(payload, torch.Tensor):
        return int(payload.numel() * payload.element_size())
    if isinstance(payload, (list, tuple)):
        return sum(_payload_elements(item) for item in payload)
    raise TypeError(f"unsupported payload type: {type(payload).__name__}")


def _bpp(payload: Any, sample: torch.Tensor) -> float:
    return float(_payload_elements(payload) * 8 / sample.numel())


def _measure(
    label: str,
    operation: Callable[[], Any],
    repeats: int,
    throughput_bytes: int,
) -> dict[str, float | str]:
    samples: list[float] = []
    for _ in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        operation()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    median_ms = statistics.median(samples)
    throughput_gb_s = (
        throughput_bytes / (median_ms / 1000.0) / 1e9
        if median_ms else float("inf")
    )
    return {
        "operation": label,
        "median_ms": median_ms,
        "min_ms": min(samples),
        "throughput_gb_s": throughput_gb_s,
        "throughput_bytes_per_call": throughput_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--kodak", type=Path, help="external Kodak .f32 calibration dataset")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/hwj/data/bmshj2018-factorized/bmshj2018-factorized-1.pth"),
        help="local CompressAI checkpoint; no network download is performed",
    )
    parser.add_argument("--quality", type=int, default=1, help="CompressAI Balle model quality")
    parser.add_argument("--p-ch", type=int, default=4, help="latent channels per GA-ANS chunk")
    parser.add_argument("--repeats", type=int, default=10, help="timed repetitions after warmup")
    parser.add_argument("--precision-budget", type=float, default=None, help="CAIEC RD Cost epsilon for Kodak selection")
    parser.add_argument("--precision-config", type=Path, help="reuse a saved per-operator precision config")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if bool(args.image) == bool(args.kodak):
        parser.error("provide exactly one of --image or --kodak")
    if args.precision_config and args.kodak:
        parser.error("--precision-config and --kodak are mutually exclusive")
    if args.kodak and args.precision_budget is None:
        parser.error("--kodak requires --precision-budget unless --precision-config is used")

    if not torch.cuda.is_available():
        raise SystemExit("warp_div requires CUDA; install/build the optional CUDA extension first")
    if not args.weights.is_file():
        raise SystemExit(f"local checkpoint does not exist: {args.weights}")
    device = torch.device("cuda")
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model = bmshj2018_factorized(quality=args.quality, pretrained=False)
    model.load_state_dict(checkpoint, strict=True)
    model = model.to(device).eval()
    model.update(force=True)
    calibration = load_kodak(args.kodak, device) if args.kodak else None
    x = load_image(args.image, device) if args.image else calibration.samples[0]
    precision_config_path = args.precision_config
    precision_config_output_dir = (
        Path(__file__).resolve().parent / "precision_configs"
        if calibration is not None and precision_config_path is None
        else None
    )

    fast_model = fast_compressai.accelerate(
        model,
        P=args.p_ch,
        budget=args.precision_budget,
        device=str(device),
        calibration_data=calibration,
        dataset_name="kodak" if calibration is not None else None,
        precision_config_path=str(precision_config_path) if precision_config_path else None,
        precision_config_output_dir=(
            str(precision_config_output_dir) if precision_config_output_dir else None
        ),
    )

    with torch.inference_mode():
        reference_encoded = model.compress(x)
        reference_decoded = model.decompress(
            reference_encoded["strings"], reference_encoded["shape"]
        )["x_hat"]
        encoded = fast_model.compress(x)
        decoded = fast_model.decompress(encoded["strings"], encoded["shape"])["x_hat"]

    max_abs_error = float((decoded - reference_decoded).abs().max().item())
    mean_abs_error = float((decoded - reference_decoded).abs().mean().item())
    reference_payload = reference_encoded["strings"][0]
    payload = encoded["strings"][0]
    target = calibration.raw_targets[0] if calibration is not None else x
    reference_psnr_input = reference_decoded * 255.0 if calibration is not None else reference_decoded
    accelerated_psnr_input = decoded * 255.0 if calibration is not None else decoded
    data_range = 255.0 if calibration is not None else 1.0
    reference_psnr = _psnr_db(reference_psnr_input, target, data_range)
    accelerated_psnr = _psnr_db(accelerated_psnr_input, target, data_range)
    reference_bpp = _bpp(reference_payload, x)
    accelerated_bpp = _bpp(payload, x)
    warmup = 2
    with torch.inference_mode():
        for _ in range(warmup):
            model.compress(x)
            model.decompress(reference_encoded["strings"], reference_encoded["shape"])
            fast_model.compress(x)
            fast_model.decompress(encoded["strings"], encoded["shape"])
        timings = [
            _measure(
                "reference_compress", lambda: model.compress(x), args.repeats,
                x.numel() * x.element_size(),
            ),
            _measure(
                "reference_decompress",
                lambda: model.decompress(reference_encoded["strings"], reference_encoded["shape"]),
                args.repeats, x.numel() * x.element_size(),
            ),
            _measure(
                "warp_div_compress", lambda: fast_model.compress(x), args.repeats,
                x.numel() * x.element_size(),
            ),
            _measure(
                "warp_div_decompress",
                lambda: fast_model.decompress(encoded["strings"], encoded["shape"]),
                args.repeats, x.numel() * x.element_size(),
            ),
        ]

    by_name = {item["operation"]: item for item in timings}
    for operation in ("compress", "decompress"):
        reference_ms = float(by_name[f"reference_{operation}"]["median_ms"])
        warp_div_ms = float(by_name[f"warp_div_{operation}"]["median_ms"])
        by_name[f"warp_div_{operation}"]["speedup_vs_reference"] = (
            reference_ms / warp_div_ms if warp_div_ms else float("inf")
        )
    reference_compress = by_name["reference_compress"]
    reference_decompress = by_name["reference_decompress"]
    accelerated_compress = by_name["warp_div_compress"]
    accelerated_decompress = by_name["warp_div_decompress"]
    summary = {
        "reference": {
            "compress_throughput_gb_s": reference_compress["throughput_gb_s"],
            "decompress_throughput_gb_s": reference_decompress["throughput_gb_s"],
            "bpp": reference_bpp,
            "psnr_db": reference_psnr,
        },
        "accelerated": {
            "compress_throughput_gb_s": accelerated_compress["throughput_gb_s"],
            "decompress_throughput_gb_s": accelerated_decompress["throughput_gb_s"],
            "bpp": accelerated_bpp,
            "psnr_db": accelerated_psnr,
        },
        "compress_speedup": accelerated_compress["speedup_vs_reference"],
        "decompress_speedup": accelerated_decompress["speedup_vs_reference"],
    }
    report_root = Path(__file__).resolve().parent / "reports"
    report_path = report_root / f"{type(model).__name__}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')}.json"
    report = {
        "summary": summary,
        "input": {
            "model": type(model).__name__, "quality": args.quality, "p_ch": args.p_ch,
            "device": str(device), "shape": list(x.shape), "repeats": args.repeats,
            "warmup": warmup, "data_range": data_range,
        },
        "timings": timings,
        "quality": {
            "reference_psnr_db": reference_psnr,
            "accelerated_psnr_db": accelerated_psnr,
            "psnr_delta_db": accelerated_psnr - reference_psnr,
            "max_abs_error_between_outputs": max_abs_error,
            "mean_abs_error_between_outputs": mean_abs_error,
        },
        "payload": {
            "type": type(payload).__name__,
            "reference_bytes_device_local": _payload_elements(reference_payload),
            "accelerated_bytes_device_local": _payload_elements(payload),
            "reference_bpp": reference_bpp,
            "accelerated_bpp": accelerated_bpp,
        },
        "dataset": ({
            "name": calibration.dataset_name,
            "fingerprint": calibration.dataset_fingerprint,
            "samples": len(calibration.samples),
            "normalization": calibration.normalization,
            "denormalization": calibration.denormalization,
        } if calibration is not None else None),
        "precision_config_path": fast_model.acceleration_report.precision_config_path,
        "acceleration_report": fast_model.acceleration_report.to_dict(),
    }
    report_root.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Detailed report: {report_path}")
    print("Note: WarpDivPayload is device-local and is not a persistent bytes bitstream.")


if __name__ == "__main__":
    main()
