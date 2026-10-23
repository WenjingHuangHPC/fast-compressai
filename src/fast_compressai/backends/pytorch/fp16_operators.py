"""CUDA FP16 module candidates for Balle floating-point operators."""

from __future__ import annotations

import copy
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


class FP16Conv2d(nn.Conv2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input.to(torch.float16), self.weight.to(torch.float16),
            None if self.bias is None else self.bias.to(torch.float16),
            self.stride, self.padding, self.dilation, self.groups,
        ).to(input.dtype)


class FP16ConvTranspose2d(nn.ConvTranspose2d):
    def forward(self, input: torch.Tensor, output_size=None) -> torch.Tensor:
        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size,
            2, self.dilation,
        )
        return F.conv_transpose2d(
            input.to(torch.float16), self.weight.to(torch.float16),
            None if self.bias is None else self.bias.to(torch.float16),
            self.stride, self.padding, output_padding, self.groups, self.dilation,
        ).to(input.dtype)


class FP16GDN(nn.Module):
    """FP16 floating GDN/IGDN compute with FP32 parameter storage."""

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
        values = x.to(torch.float16)
        beta = self.beta_reparam(self.beta).to(torch.float16)
        gamma = self.gamma_reparam(self.gamma).reshape(self.channels, self.channels, 1, 1).to(torch.float16)
        norm = F.conv2d(values.abs() if self.gdn1 else values.square(), gamma, beta)
        if self.gdn1:
            if not self.inverse:
                norm = norm.reciprocal()
        elif self.inverse:
            norm = torch.sqrt(norm)
        else:
            norm = torch.rsqrt(norm)
        return (values * norm).to(x.dtype)


def _capability(module: nn.Module, context: Mapping) -> tuple[bool, str | None]:
    parameter = next(module.parameters(), None)
    if parameter is None or parameter.device.type != "cuda":
        return False, "FP16 candidate requires a CUDA module"
    if module.training or context.get("training", module.training):
        return False, "FP16 candidate is inference-only"
    return True, None


def conv2d_factory(module: nn.Module, context: Mapping) -> nn.Module:
    del context
    if not isinstance(module, nn.Conv2d):
        raise TypeError("FP16 Conv2d factory requires nn.Conv2d")
    return FP16Conv2d(module.in_channels, module.out_channels, module.kernel_size,
                      module.stride, module.padding, module.dilation, module.groups,
                      module.bias is not None, module.padding_mode,
                      device=module.weight.device, dtype=module.weight.dtype)


def conv_transpose2d_factory(module: nn.Module, context: Mapping) -> nn.Module:
    del context
    if not isinstance(module, nn.ConvTranspose2d):
        raise TypeError("FP16 ConvTranspose2d factory requires nn.ConvTranspose2d")
    return FP16ConvTranspose2d(module.in_channels, module.out_channels, module.kernel_size,
                               module.stride, module.padding, module.output_padding,
                               module.groups, module.bias is not None, module.dilation,
                               module.padding_mode, device=module.weight.device,
                               dtype=module.weight.dtype)


def gdn_factory(module: nn.Module, context: Mapping) -> nn.Module:
    del context
    if not hasattr(module, "beta") or not hasattr(module, "gamma"):
        raise TypeError("FP16 GDN factory requires beta/gamma parameters")
    return FP16GDN(module)


capability = _capability
