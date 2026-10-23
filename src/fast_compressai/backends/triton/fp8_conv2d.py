"""Experimental Triton FP8 Conv2d candidate.

Triton owns FP8 quantization; the matrix multiply intentionally reuses
PyTorch's validated FP8 scaled-mm primitive until a Triton GEMM kernel is
profiled as a separate implementation.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - capability reports the reason
    triton = None
    tl = None

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
# Stay one representable step below E4M3FN's endpoint: Triton conversion may
# encode rounded-over endpoint values as NaN on some compiler/device builds.
FP8_CLAMP = 416.0


if triton is not None:
    @triton.jit
    def _quantize_fp8_kernel(
        x_ptr, out_ptr, scale_ptr, n_elements,
        FP8_LIMIT: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr)
        q = tl.maximum(tl.minimum(x / scale, FP8_LIMIT), -FP8_LIMIT)
        q = tl.where(x == 0.0, 0.0, q)
        tl.store(out_ptr + offsets, q, mask=mask)
else:
    _quantize_fp8_kernel = None


def fp8_triton_conv2d_capability(module: nn.Module, context: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(module, nn.Conv2d):
        return False, "Triton FP8 candidate requires nn.Conv2d"
    if triton is None:
        return False, "Triton is not installed"
    if module.weight.device.type != "cuda":
        return False, "Triton FP8 candidate requires CUDA"
    if torch.cuda.get_device_capability(module.weight.device) < (9, 0):
        return False, "Triton FP8 candidate is validated only on sm90+"
    if module.training or context.get("training", module.training):
        return False, "Triton FP8 candidate is inference-only"
    if module.groups != 1 or module.padding_mode != "zeros":
        return False, "Triton FP8 candidate supports groups=1 and zero padding only"
    if tuple(module.dilation) != (1, 1) or tuple(module.stride) not in {(1, 1), (2, 2)}:
        return False, "Triton FP8 candidate supports dilation=1 and stride 1/2"
    if not hasattr(torch, "_scaled_mm"):
        return False, "PyTorch FP8 scaled-mm primitive is unavailable"
    return True, None


def _triton_quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    # Triton produces scaled/clamped FP32 values; Torch performs the final
    # E4M3FN cast because Triton 3.4's direct FP8 store can encode saturation
    # edge values as NaNs on this toolchain.
    output = torch.empty_like(tensor, dtype=torch.float32)
    n_elements = tensor.numel()
    block = 1 << min(12, max(5, (n_elements - 1).bit_length()))
    _quantize_fp8_kernel[(triton.cdiv(n_elements, block),)](
        tensor.reshape(-1), output.reshape(-1), scale, n_elements,
        FP8_LIMIT=FP8_CLAMP, BLOCK=block,
    )
    return output.to(FP8_DTYPE)


class FP8TritonConv2d(nn.Conv2d):
    backend = "triton"
    precision = "fp8"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cuda":
            raise RuntimeError("FP8TritonConv2d requires CUDA")
        n, _, height, width = input.shape
        kernel_h, kernel_w = self.kernel_size
        output_height = (height + 2 * self.padding[0] - kernel_h) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - kernel_w) // self.stride[1] + 1
        patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        patches = patches.transpose(1, 2).reshape(-1, patches.shape[1]).contiguous()
        weight = self.weight.reshape(self.out_channels, -1).transpose(0, 1).contiguous()
        k_original, out_original = patches.shape[1], weight.shape[1]
        k_padded = ((k_original + 15) // 16) * 16
        out_padded = ((out_original + 15) // 16) * 16
        patches = F.pad(patches, (0, k_padded - k_original))
        weight = F.pad(weight, (0, out_padded - out_original, 0, k_padded - k_original))
        input_scale = torch.clamp(patches.float().abs().amax() / FP8_MAX,
                                  min=torch.finfo(torch.float32).tiny)
        weight_scale = torch.clamp(weight.float().abs().amax() / FP8_MAX,
                                   min=torch.finfo(torch.float32).tiny)
        a = _triton_quantize(patches.float(), input_scale).contiguous()
        b = _triton_quantize(weight.float(), weight_scale).t().contiguous().t()
        output = torch._scaled_mm(a, b, input_scale, weight_scale, out_dtype=torch.bfloat16)
        output = output[:, :out_original]
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        output = output.reshape(n, output_height * output_width, out_original)
        output = output.transpose(1, 2).reshape(n, out_original, output_height, output_width)
        return output.to(input.dtype)


def fp8_triton_conv2d_factory(module: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    supported, reason = fp8_triton_conv2d_capability(module, context)
    if not supported:
        raise RuntimeError(reason or "Triton FP8 Conv2d unavailable")
    return FP8TritonConv2d(
        module.in_channels, module.out_channels, module.kernel_size, module.stride,
        module.padding, module.dilation, module.groups, module.bias is not None,
        module.padding_mode, device=module.weight.device, dtype=module.weight.dtype,
    )
