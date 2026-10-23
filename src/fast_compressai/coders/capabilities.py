"""Coder capability discovery without loading optional native extensions."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CoderCapability:
    backend: str
    available: bool
    reason: str | None
    device: str | None
    persistent_payload: bool


def warp_div_capability(device: torch.device | str | None = None) -> CoderCapability:
    requested = torch.device(device) if device is not None else None
    if not torch.cuda.is_available():
        return CoderCapability("warp_div", False, "CUDA runtime/device unavailable", None, False)
    if requested is not None and requested.type != "cuda":
        return CoderCapability("warp_div", False, "warp_div requires a CUDA device", str(requested), False)
    spec = importlib.util.find_spec("fast_compressai.coders._warp_div")
    if spec is None:
        return CoderCapability("warp_div", False, "optional CUDA extension is not built", str(requested or torch.device("cuda")), False)
    return CoderCapability("warp_div", True, None, str(requested or torch.device("cuda")), False)


def coder_catalog() -> tuple[str, ...]:
    """Return exactly the supported entropy coder backend identifiers."""

    return ("compressai-rans", "warp_div")


def create_coder(name: str):
    """Construct one supported backend without loading unrelated backends."""

    if name == "compressai-rans":
        from .compressai_ans import CompressAIANSBackend

        return CompressAIANSBackend()
    if name == "warp_div":
        from .warp_div import WarpDivBackend

        return WarpDivBackend()
    raise ValueError(f"unsupported coder {name!r}; supported coders are {coder_catalog()}")
