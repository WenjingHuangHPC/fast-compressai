"""Triton-quantized FP8 candidates for Balle floating operators.

The elementwise FP8 quantization is implemented by Triton kernels. Matrix
multiplication intentionally uses PyTorch's scaled-mm primitive, matching the
existing Triton Conv2d candidate while keeping the backend identity explicit.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

from .fp8_conv2d import FP8_CLAMP, FP8_DTYPE, FP8_MAX

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - capability reports the reason
    triton = None
    tl = None


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


def _base_capability(module: nn.Module, context: Mapping[str, Any], expected: type[nn.Module]) -> tuple[bool, str | None]:
    if not isinstance(module, expected):
        return False, f"Triton FP8 candidate requires {expected.__name__}"
    if triton is None:
        return False, "Triton is not installed"
    parameter = next(module.parameters(), None)
    if parameter is None or parameter.device.type != "cuda":
        return False, "Triton FP8 candidate requires CUDA"
    if torch.cuda.get_device_capability(parameter.device) < (9, 0):
        return False, "Triton FP8 candidate is validated only on sm90+"
    if module.training or context.get("training", module.training):
        return False, "Triton FP8 candidate is inference-only"
    if not hasattr(torch, "_scaled_mm"):
        return False, "PyTorch FP8 scaled-mm primitive is unavailable"
    return True, None


def _quantize(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(tensor, dtype=torch.float32)
    n_elements = tensor.numel()
    block = 1 << min(12, max(5, (n_elements - 1).bit_length()))
    _quantize_fp8_kernel[(triton.cdiv(n_elements, block),)](
        tensor.reshape(-1), output.reshape(-1), scale, n_elements,
        FP8_LIMIT=FP8_CLAMP, BLOCK=block,
    )
    return output.to(FP8_DTYPE)


def _scale(tensor: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor.detach().float().abs().amax() / FP8_MAX,
                       min=torch.finfo(torch.float32).tiny)


def _scaled_mm(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_scale, right_scale = _scale(left), _scale(right)
    left_fp8 = _quantize(left, left_scale).contiguous()
    right_fp8 = _quantize(right, right_scale).t().contiguous().t()
    return torch._scaled_mm(left_fp8, right_fp8, left_scale, right_scale,
                            out_dtype=torch.bfloat16)


def fp8_triton_conv_transpose2d_capability(module: nn.Module, context: Mapping[str, Any]) -> tuple[bool, str | None]:
    supported, reason = _base_capability(module, context, nn.ConvTranspose2d)
    if not supported:
        return supported, reason
    if module.groups != 1 or module.padding_mode != "zeros" or tuple(module.dilation) != (1, 1):
        return False, "Triton FP8 ConvTranspose2d supports groups=1, zero padding and dilation=1 only"
    if tuple(module.stride) not in {(1, 1), (2, 2)}:
        return False, "Triton FP8 ConvTranspose2d supports stride 1 or 2"
    if any(op >= stride for op, stride in zip(module.output_padding, module.stride)):
        return False, "Triton FP8 ConvTranspose2d output_padding must be smaller than stride"
    return True, None


class FP8TritonConvTranspose2d(nn.ConvTranspose2d):
    backend = "triton"
    precision = "fp8"

    def forward(self, input: torch.Tensor, output_size=None) -> torch.Tensor:
        n, _, height, width = input.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h = (height - 1) * stride_h - 2 * self.padding[0] + kernel_h + self.output_padding[0]
        out_w = (width - 1) * stride_w - 2 * self.padding[1] + kernel_w + self.output_padding[1]
        if output_size is not None and tuple(output_size[-2:]) != (out_h, out_w):
            raise RuntimeError("Triton FP8 ConvTranspose2d custom output_size is unsupported")
        positions = height * width
        rows = input.reshape(n, self.in_channels, positions).transpose(1, 2).reshape(-1, self.in_channels)
        weight = self.weight.permute(0, 2, 3, 1).reshape(self.in_channels, -1)
        k_original, out_original = rows.shape[1], weight.shape[1]
        k_padded = ((k_original + 15) // 16) * 16
        out_padded = ((out_original + 15) // 16) * 16
        rows = F.pad(rows, (0, k_padded - k_original))
        weight = F.pad(weight, (0, out_padded - out_original, 0, k_padded - k_original))
        products = _scaled_mm(rows, weight)[:, :out_original]
        products = products.float().reshape(n, positions, kernel_h, kernel_w, self.out_channels)
        result = torch.zeros((n, self.out_channels, out_h, out_w), device=input.device, dtype=torch.float32)
        for ky in range(kernel_h):
            for kx in range(kernel_w):
                oy = torch.arange(height, device=input.device) * stride_h - self.padding[0] + ky
                ox = torch.arange(width, device=input.device) * stride_w - self.padding[1] + kx
                valid_y, valid_x = (oy >= 0) & (oy < out_h), (ox >= 0) & (ox < out_w)
                values = products[:, :, ky, kx, :].reshape(n, height, width, self.out_channels)
                values = values[:, valid_y][:, :, valid_x].permute(0, 3, 1, 2)
                result[:, :, oy[valid_y, None], ox[None, valid_x]] += values
        if self.bias is not None:
            result += self.bias.float().reshape(1, -1, 1, 1)
        return result.to(input.dtype)


def fp8_triton_conv_transpose2d_factory(module: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    supported, reason = fp8_triton_conv_transpose2d_capability(module, context)
    if not supported:
        raise RuntimeError(reason or "Triton FP8 ConvTranspose2d unavailable")
    return FP8TritonConvTranspose2d(
        module.in_channels, module.out_channels, module.kernel_size, module.stride,
        module.padding, module.output_padding, module.groups, module.bias is not None,
        module.dilation, module.padding_mode, device=module.weight.device, dtype=module.weight.dtype,
    )


def fp8_triton_gdn_capability(module: nn.Module, context: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not hasattr(module, "beta") or not hasattr(module, "gamma"):
        return False, "Triton FP8 GDN requires beta/gamma parameters"
    supported, reason = _base_capability(module, context, nn.Module)
    if not supported:
        return supported, reason
    channels = int(module.beta.numel())
    workspace = int(context.get("input_elements", 0)) * max(1, channels) * 4
    limit = int(context.get("max_workspace_bytes", 256 * 1024 * 1024))
    if workspace > limit:
        return False, f"estimated Triton FP8 GDN workspace {workspace} exceeds limit {limit}"
    return True, None


class FP8TritonGDN(nn.Module):
    backend = "triton"
    precision = "fp8"

    def __init__(self, original: nn.Module) -> None:
        super().__init__()
        self.channels = int(original.beta.numel())
        self.inverse = bool(original.inverse)
        self.gdn1 = type(original).__name__ == "GDN1"
        self.beta_reparam = copy.deepcopy(original.beta_reparam)
        self.gamma_reparam = copy.deepcopy(original.gamma_reparam)
        self.register_parameter("beta", nn.Parameter(original.beta.detach().clone(), requires_grad=original.beta.requires_grad))
        self.register_parameter("gamma", nn.Parameter(original.gamma.detach().clone(), requires_grad=original.gamma.requires_grad))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, channels, height, width = x.shape
        values = x.float()
        norm_input = values.abs() if self.gdn1 else values.square()
        rows = norm_input.permute(0, 2, 3, 1).reshape(-1, channels).contiguous()
        activation = values.permute(0, 2, 3, 1).reshape(-1, channels).contiguous()
        gamma = self.gamma_reparam(self.gamma).reshape(channels, channels).float().transpose(0, 1)
        beta = self.beta_reparam(self.beta).reshape(channels).float()
        padded = ((channels + 15) // 16) * 16
        norm = _scaled_mm(F.pad(rows, (0, padded - channels)),
                          F.pad(gamma, (0, padded - channels, 0, padded - channels)))
        norm = norm[:, :channels].float() + beta
        activation_scale = _scale(activation)
        activation = _quantize(activation, activation_scale).float() * activation_scale
        if self.gdn1:
            if not self.inverse:
                norm = norm.reciprocal()
        elif self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)
        return (activation * norm).reshape(n, height, width, channels).permute(0, 3, 1, 2).to(x.dtype)


def fp8_triton_gdn_factory(module: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    supported, reason = fp8_triton_gdn_capability(module, context)
    if not supported:
        raise RuntimeError(reason or "Triton FP8 GDN unavailable")
    return FP8TritonGDN(module)


fp8_triton_igdn_factory = fp8_triton_gdn_factory
