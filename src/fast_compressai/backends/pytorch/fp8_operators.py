"""Experimental FP8 Conv2d candidate for H100-class CUDA devices.

This is intentionally not a general Conv2d implementation: it uses an
unfold/im2col materialization plus PyTorch's private scaled-mm primitive.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)


def fp8_conv2d_capability(module: nn.Module, context: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(module, nn.Conv2d):
        return False, "FP8 candidate requires nn.Conv2d"
    if module.weight.device.type != "cuda":
        return False, "FP8 candidate requires CUDA"
    capability = torch.cuda.get_device_capability(module.weight.device)
    if capability < (9, 0):
        return False, "FP8 scaled-mm candidate is validated only on sm90+"
    if not hasattr(torch, "_scaled_mm"):
        return False, "PyTorch scaled-mm primitive is unavailable"
    if module.training or context.get("training", False):
        return False, "FP8 candidate is inference-only"
    if module.groups != 1:
        return False, "FP8 candidate currently supports groups=1 only"
    if module.padding_mode != "zeros":
        return False, "FP8 candidate currently supports zero padding only"
    if tuple(module.dilation) != (1, 1):
        return False, "FP8 candidate currently supports dilation=1 only"
    if tuple(module.stride) not in {(1, 1), (2, 2)}:
        return False, "FP8 candidate currently supports stride 1 or 2"
    return True, None


def _symmetric_scale(tensor: torch.Tensor) -> torch.Tensor:
    maximum = tensor.detach().float().abs().amax()
    return torch.clamp(maximum / FP8_MAX, min=torch.finfo(torch.float32).tiny)


def _quantize_fp8(tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.clamp(tensor.float() / scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)


class FP8Conv2d(nn.Conv2d):
    """Experimental FP8 Conv2d with FP32 parameter storage."""

    backend = "pytorch-scaled-mm"
    precision = "fp8"

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.device.type != "cuda":
            raise RuntimeError("FP8Conv2d requires CUDA")
        if self.groups != 1 or self.padding_mode != "zeros" or tuple(self.dilation) != (1, 1):
            raise RuntimeError("unsupported FP8Conv2d configuration")
        n, _, height, width = input.shape
        kernel_h, kernel_w = self.kernel_size
        output_height = (height + 2 * self.padding[0] - kernel_h) // self.stride[0] + 1
        output_width = (width + 2 * self.padding[1] - kernel_w) // self.stride[1] + 1
        if output_height <= 0 or output_width <= 0:
            raise RuntimeError("FP8Conv2d output shape is invalid")

        patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride)
        # [N, K, L] -> [N*L, K], row-major A for scaled_mm.
        patches = patches.transpose(1, 2).reshape(-1, patches.shape[1]).contiguous()
        weight = self.weight.reshape(self.out_channels, -1).transpose(0, 1).contiguous()
        k_original = patches.shape[1]
        out_original = weight.shape[1]
        k_padded = ((k_original + 15) // 16) * 16
        out_padded = ((out_original + 15) // 16) * 16
        if k_padded != k_original or out_padded != out_original:
            patches = F.pad(patches, (0, k_padded - k_original))
            weight = F.pad(weight, (0, out_padded - out_original, 0, k_padded - k_original))

        input_scale = _symmetric_scale(patches)
        weight_scale = _symmetric_scale(weight)
        patches_fp8 = _quantize_fp8(patches, input_scale)
        weight_fp8 = _quantize_fp8(weight, weight_scale)
        # scaled_mm requires row-major activations and column-major weights.
        weight_fp8 = weight_fp8.t().contiguous().t()
        output = torch._scaled_mm(
            patches_fp8,
            weight_fp8,
            input_scale,
            weight_scale,
            out_dtype=torch.bfloat16,
        )
        output = output[:, :out_original]
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        output = output.reshape(n, output_height * output_width, out_original)
        output = output.transpose(1, 2).reshape(n, out_original, output_height, output_width)
        return output.to(input.dtype)


def fp8_conv_transpose2d_capability(module: nn.Module, context: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not isinstance(module, nn.ConvTranspose2d):
        return False, "FP8 candidate requires nn.ConvTranspose2d"
    if module.weight.device.type != "cuda" or torch.cuda.get_device_capability(module.weight.device) < (9, 0):
        return False, "FP8 ConvTranspose2d requires an sm90+ CUDA device"
    if module.training or context.get("training", module.training):
        return False, "FP8 candidate is inference-only"
    if module.groups != 1 or module.padding_mode != "zeros" or tuple(module.dilation) != (1, 1):
        return False, "FP8 ConvTranspose2d supports groups=1, zero padding and dilation=1 only"
    if tuple(module.stride) not in {(1, 1), (2, 2)}:
        return False, "FP8 ConvTranspose2d supports stride 1 or 2"
    if any(op >= stride for op, stride in zip(module.output_padding, module.stride)):
        return False, "FP8 ConvTranspose2d output_padding must be smaller than stride"
    if not hasattr(torch, "_scaled_mm"):
        return False, "PyTorch scaled-mm primitive is unavailable"
    return True, None


class FP8ConvTranspose2d(nn.ConvTranspose2d):
    """Experimental FP8 transposed convolution via scatter-add of scaled GEMM."""

    backend = "pytorch-scaled-mm"
    precision = "fp8"

    def forward(self, input: torch.Tensor, output_size=None) -> torch.Tensor:
        if input.device.type != "cuda":
            raise RuntimeError("FP8ConvTranspose2d requires CUDA")
        if (self.groups != 1 or tuple(self.dilation) != (1, 1)
                or any(op >= stride for op, stride in zip(self.output_padding, self.stride))):
            raise RuntimeError("unsupported FP8ConvTranspose2d configuration")
        n, _, height, width = input.shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        out_h = (height - 1) * stride_h - 2 * self.padding[0] + kernel_h + self.output_padding[0]
        out_w = (width - 1) * stride_w - 2 * self.padding[1] + kernel_w + self.output_padding[1]
        if output_size is not None:
            requested = tuple(output_size[-2:])
            if requested != (out_h, out_w):
                raise RuntimeError("FP8ConvTranspose2d custom output_size is unsupported")
        positions = height * width
        patches = input.reshape(n, self.in_channels, positions).transpose(1, 2).reshape(-1, self.in_channels)
        weight = self.weight.permute(0, 2, 3, 1).reshape(self.in_channels, -1)
        k_original, out_original = patches.shape[1], weight.shape[1]
        k_padded = ((k_original + 15) // 16) * 16
        out_padded = ((out_original + 15) // 16) * 16
        if k_padded != k_original:
            patches = F.pad(patches, (0, k_padded - k_original))
        if k_padded != k_original or out_padded != out_original:
            weight = F.pad(weight, (0, out_padded - out_original, 0, k_padded - k_original))
        input_scale, weight_scale = _symmetric_scale(patches), _symmetric_scale(weight)
        a = _quantize_fp8(patches, input_scale).contiguous()
        b = _quantize_fp8(weight, weight_scale).t().contiguous().t()
        products = torch._scaled_mm(a, b, input_scale, weight_scale, out_dtype=torch.bfloat16)[:, :out_original]
        products = products.to(torch.float32).reshape(n, positions, kernel_h, kernel_w, self.out_channels)
        result = torch.zeros((n, self.out_channels, out_h, out_w), device=input.device, dtype=torch.float32)
        # Scatter each input-position/kernel-position contribution into the
        # transposed-convolution output lattice. This is experimental and may
        # be slower than cuDNN because it materializes and scatters columns.
        for ky in range(kernel_h):
            for kx in range(kernel_w):
                oy = torch.arange(height, device=input.device) * stride_h - self.padding[0] + ky
                ox = torch.arange(width, device=input.device) * stride_w - self.padding[1] + kx
                valid_y = (oy >= 0) & (oy < out_h)
                valid_x = (ox >= 0) & (ox < out_w)
                yidx = oy[valid_y]
                xidx = ox[valid_x]
                values = products[:, :, ky, kx, :].reshape(n, height, width, self.out_channels)
                values = values[:, valid_y][:, :, valid_x].permute(0, 3, 1, 2)
                result[:, :, yidx[:, None], xidx[None, :]] += values
        if self.bias is not None:
            result += self.bias.float().reshape(1, -1, 1, 1)
        return result.to(input.dtype)


def fp8_conv_transpose2d_factory(module: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    supported, reason = fp8_conv_transpose2d_capability(module, context)
    if not supported:
        raise RuntimeError(reason or "FP8 ConvTranspose2d unavailable")
    return FP8ConvTranspose2d(
        module.in_channels, module.out_channels, module.kernel_size, module.stride,
        module.padding, module.output_padding, module.groups, module.bias is not None,
        module.dilation, module.padding_mode, device=module.weight.device, dtype=module.weight.dtype,
    )


class FP8GDN(nn.Module):
    """Experimental FP8 channel-mixing matrix multiply with FP32 nonlinearities."""

    backend = "pytorch-scaled-mm"
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
        if x.device.type != "cuda":
            raise RuntimeError("FP8GDN requires CUDA")
        n, channels, height, width = x.shape
        dtype = x.dtype
        values = x.float()
        norm_input = values.abs() if self.gdn1 else values.square()
        rows = norm_input.permute(0, 2, 3, 1).reshape(-1, channels).contiguous()
        activation_rows = values.permute(0, 2, 3, 1).reshape(-1, channels)
        # CompressAI GDN uses a channel-mixing 1x1 convolution with weight
        # [C_out, C_in, 1, 1]. Flatten its channel axes as [C_in, C_out] for
        # the column-major RHS expected by torch._scaled_mm.
        # Conv2d weight is [out_channel, in_channel]; GEMM RHS must be its transpose.
        gamma = self.gamma_reparam(self.gamma).reshape(channels, channels).float().transpose(0, 1)
        beta = self.beta_reparam(self.beta).reshape(channels).float()
        kpad = ((channels + 15) // 16) * 16
        npad = ((channels + 15) // 16) * 16
        rows_padded = F.pad(rows, (0, kpad - channels))
        gamma_padded = F.pad(gamma, (0, npad - channels, 0, kpad - channels))
        input_scale, weight_scale = _symmetric_scale(rows_padded), _symmetric_scale(gamma_padded)
        a = _quantize_fp8(rows_padded, input_scale).contiguous()
        b = _quantize_fp8(gamma_padded, weight_scale).t().contiguous().t()
        norm = torch._scaled_mm(a, b, input_scale, weight_scale, out_dtype=torch.bfloat16)
        norm = norm[:, :channels].float() + beta
        # Keep the final activation multiply at BF16 so the candidate reflects
        # the FP8/FP16-style module boundary instead of FP32 identity behavior.
        activation_fp8 = _quantize_fp8(activation_rows, _symmetric_scale(activation_rows))
        activation_bf16 = (activation_fp8.float() * _symmetric_scale(activation_rows)).to(torch.bfloat16).float()
        if self.gdn1:
            if not self.inverse:
                norm = norm.reciprocal()
        elif self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)
        output = activation_bf16 * norm
        return output.reshape(n, height, width, channels).permute(0, 3, 1, 2).to(dtype)


def fp8_gdn_capability(module: nn.Module, context: Mapping[str, Any]) -> tuple[bool, str | None]:
    if not hasattr(module, "beta") or not hasattr(module, "gamma"):
        return False, "FP8 GDN requires beta/gamma parameters"
    if module.beta.device.type != "cuda" or torch.cuda.get_device_capability(module.beta.device) < (9, 0):
        return False, "FP8 GDN requires an sm90+ CUDA device"
    if module.training or context.get("training", module.training):
        return False, "FP8 GDN is inference-only"
    if not hasattr(torch, "_scaled_mm"):
        return False, "PyTorch scaled-mm primitive is unavailable"
    channels = int(module.beta.numel())
    workspace = int(context.get("input_elements", 0)) * max(1, channels) * 4
    limit = int(context.get("max_workspace_bytes", 256 * 1024 * 1024))
    if workspace > limit:
        return False, f"estimated FP8 GDN workspace {workspace} exceeds limit {limit}"
    return True, None


def fp8_gdn_factory(module: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    supported, reason = fp8_gdn_capability(module, context)
    if not supported:
        raise RuntimeError(reason or "FP8 GDN unavailable")
    return FP8GDN(module)


def fp8_igdn_factory(module: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    if not bool(getattr(module, "inverse", False)):
        raise RuntimeError("FP8 IGDN candidate requires inverse=True")
    return fp8_gdn_factory(module, context)


def fp8_conv2d_factory(module: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    supported, reason = fp8_conv2d_capability(module, context)
    if not supported:
        raise RuntimeError(reason or "FP8 Conv2d is unavailable")
    if not isinstance(module, nn.Conv2d):
        raise TypeError("FP8 factory requires nn.Conv2d")
    return FP8Conv2d(
        module.in_channels, module.out_channels, module.kernel_size, module.stride,
        module.padding, module.dilation, module.groups, module.bias is not None,
        module.padding_mode, device=module.weight.device, dtype=module.weight.dtype,
    )
