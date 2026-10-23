"""Balle operator inventory and extensible precision/backend catalog."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import nn

from .candidates import CandidateKey, OperatorCandidate, OperatorInstance, OperatorSpec
from .errors import RegistryError
from ..backends.triton.fp8_conv2d import (
    fp8_triton_conv2d_capability,
    fp8_triton_conv2d_factory,
)
from ..backends.triton.fp8_operators import (
    fp8_triton_conv_transpose2d_capability,
    fp8_triton_conv_transpose2d_factory,
    fp8_triton_gdn_capability,
    fp8_triton_gdn_factory,
)
from ..backends.pytorch.fp8_operators import (
    fp8_conv2d_capability,
    fp8_conv2d_factory,
    fp8_conv_transpose2d_capability,
    fp8_conv_transpose2d_factory,
    fp8_gdn_capability,
    fp8_gdn_factory,
    fp8_igdn_factory,
)
from ..backends.pytorch.bf16_operators import (
    conv2d_factory as conv2d_bf16_factory,
    conv_transpose2d_factory as conv_transpose2d_bf16_factory,
    cuda_bf16_capability,
    gdn_factory as gdn_bf16_factory,
)
from ..backends.pytorch.fp16_operators import (
    capability as cuda_fp16_capability,
    conv2d_factory as conv2d_fp16_factory,
    conv_transpose2d_factory as conv_transpose2d_fp16_factory,
    gdn_factory as gdn_fp16_factory,
)


@dataclass(frozen=True)
class CatalogEntry:
    spec: OperatorSpec
    candidates: tuple[OperatorCandidate, ...]
    precision_locked: bool = False
    notes: str = ""


@dataclass(frozen=True)
class CandidateApplication:
    instance_id: str
    module_path: str
    logical_op: str
    candidate_id: str
    precision: str
    backend: str
    applied: bool


class OperatorCatalog:
    """Internal model/operator catalog; paths are runtime identities only."""

    def __init__(self) -> None:
        self._by_logical_op: dict[str, CatalogEntry] = {}

    def register(self, entry: CatalogEntry) -> None:
        if entry.spec.logical_op in self._by_logical_op:
            raise RegistryError(f"logical operator {entry.spec.logical_op!r} is already registered")
        candidate_ids = [candidate.key.stable_id for candidate in entry.candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise RegistryError(f"operator {entry.spec.logical_op!r} has duplicate candidates")
        self._by_logical_op[entry.spec.logical_op] = entry

    def get(self, logical_op: str) -> CatalogEntry:
        try:
            return self._by_logical_op[logical_op]
        except KeyError as exc:
            raise RegistryError(f"logical operator {logical_op!r} is not registered") from exc

    def entries(self) -> tuple[CatalogEntry, ...]:
        return tuple(self._by_logical_op.values())

    def inventory(self, model: nn.Module) -> tuple[OperatorInstance, ...]:
        inventory: list[OperatorInstance] = []
        for path, module in model.named_modules(remove_duplicate=False):
            if not path:
                continue
            logical_op = _logical_op(module)
            if logical_op is None or logical_op not in self._by_logical_op:
                continue
            spec = self.get(logical_op).spec
            inventory.append(OperatorInstance(
                instance_id=f"{type(model).__module__}.{type(model).__qualname__}:{path}",
                module_path=path,
                logical_op=logical_op,
                module_type=f"{type(module).__module__}.{type(module).__qualname__}",
                config=_module_config(module),
                stateful=spec.stateful,
            ))
        return tuple(inventory)

    def candidate(self, logical_op: str, precision: str, backend: str | None = "pytorch-cudnn",
                 implementation: str | None = None) -> OperatorCandidate:
        for candidate in self.get(logical_op).candidates:
            key = candidate.key
            if (key.precision == precision and (backend is None or key.backend == backend)
                    and (implementation is None or key.implementation == implementation)):
                return candidate
        raise RegistryError(f"no {precision}/{backend}/{implementation} candidate registered for {logical_op}")

    def candidate_by_key(self, key: CandidateKey) -> OperatorCandidate:
        for candidate in self.get(key.logical_op).candidates:
            if candidate.key == key:
                return candidate
        raise RegistryError(f"candidate {key.stable_id} is not registered")

    def build(self, module: nn.Module, logical_op: str, precision: str,
              backend: str | None = "pytorch-cudnn",
              context: Mapping[str, Any] | None = None) -> nn.Module:
        return self.candidate(logical_op, precision, backend).build(module, context)

    def apply(self, model: nn.Module, assignments: Mapping[str, str | CandidateKey], *,
              backend: str | None = "pytorch-cudnn") -> tuple[nn.Module, tuple[CandidateApplication, ...]]:
        """Clone the model and apply an internal instance-id to candidate map."""
        candidate_model = copy.deepcopy(model)
        instances = {instance.instance_id: instance for instance in self.inventory(candidate_model)}
        unknown = set(assignments) - set(instances)
        if unknown:
            raise RegistryError(f"unknown operator instance assignment(s): {sorted(unknown)}")
        modules = dict(candidate_model.named_modules(remove_duplicate=False))
        replacements: list[tuple[str, nn.Module, CandidateApplication]] = []
        for instance_id, choice in assignments.items():
            instance = instances[instance_id]
            original = modules[instance.module_path]
            if isinstance(choice, CandidateKey):
                candidate = self.candidate_by_key(choice)
                if candidate.key.logical_op != instance.logical_op:
                    raise RegistryError(f"candidate {choice.stable_id} is not registered for {instance.logical_op}")
            else:
                candidate = self.candidate(instance.logical_op, choice, backend)
            context = {
                "device": next(original.parameters(), torch.empty(0)).device.type,
                "training": original.training,
                **dict(instance.config),
            }
            replacement = candidate.build(original, context)
            replacements.append((instance.module_path, replacement, CandidateApplication(
                instance_id=instance.instance_id,
                module_path=instance.module_path,
                logical_op=instance.logical_op,
                candidate_id=candidate.key.stable_id,
                precision=candidate.key.precision,
                backend=candidate.key.backend,
                applied=True,
            )))
        records: list[CandidateApplication] = []
        for path, replacement, record in replacements:
            _set_submodule(candidate_model, path, replacement)
            records.append(record)
        if model.training:
            candidate_model.train()
        return candidate_model, tuple(records)

    def catalog_fingerprint(self) -> str:
        from ..configs.precision import fingerprint_json
        definitions = []
        for logical_op, entry in sorted(self._by_logical_op.items()):
            definitions.append({
                "logical_op": logical_op,
                "op_type": f"{entry.spec.op_type.__module__}.{entry.spec.op_type.__qualname__}",
                "stateful": entry.spec.stateful,
                "attributes": dict(entry.spec.attributes),
                "precision_locked": entry.precision_locked,
                "candidates": [candidate.key.to_dict() for candidate in sorted(
                    entry.candidates, key=lambda item: item.key.stable_id
                )],
            })
        return fingerprint_json(definitions)


def _set_submodule(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if name not in parent._modules:
        raise RegistryError(f"module path {path!r} is no longer present in model copy")
    parent._modules[name] = replacement


def _logical_op(module: nn.Module) -> str | None:
    if isinstance(module, nn.Conv2d):
        return "conv2d"
    if isinstance(module, nn.ConvTranspose2d):
        return "conv_transpose2d"
    cls = type(module)
    if cls.__name__ in {"GDN", "GDN1"} and cls.__module__.startswith("compressai."):
        suffix = "1" if cls.__name__ == "GDN1" else ""
        return ("igdn" if bool(getattr(module, "inverse", False)) else "gdn") + suffix
    if cls.__name__ == "EntropyBottleneck" and cls.__module__.startswith("compressai."):
        return "entropy_bottleneck"
    return None


def _module_config(module: nn.Module) -> dict[str, Any]:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        return {
            "in_channels": module.in_channels, "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size), "stride": list(module.stride),
            "padding": list(module.padding), "dilation": list(module.dilation),
            "groups": module.groups, "bias": module.bias is not None,
            "device": str(module.weight.device), "dtype": str(module.weight.dtype),
        }
    if hasattr(module, "inverse") and hasattr(module, "beta") and hasattr(module, "gamma"):
        return {"channels": int(module.beta.numel()), "inverse": bool(module.inverse),
                "module_type": type(module).__name__, "device": str(module.beta.device),
                "dtype": str(module.beta.dtype)}
    if hasattr(module, "channels") and hasattr(module, "quantiles"):
        return {"channels": int(module.channels), "discrete_coder_boundary": True,
                "cdf_initialized": bool(getattr(module, "_quantized_cdf", torch.empty(0)).numel()),
                "device": str(module.quantiles.device)}
    return {}


def _reference_factory(module: nn.Module, context: Mapping[str, Any]) -> nn.Module:
    del context
    return copy.deepcopy(module)


def build_balle_catalog() -> OperatorCatalog:
    """Create Balle's GPU reference and low-precision candidate catalog."""
    catalog = OperatorCatalog()
    conv_specs = (
        ("conv2d", nn.Conv2d, conv2d_fp16_factory, conv2d_bf16_factory),
        ("conv_transpose2d", nn.ConvTranspose2d, conv_transpose2d_fp16_factory, conv_transpose2d_bf16_factory),
    )
    for logical_op, module_type, fp16_factory, bf16_factory in conv_specs:
        candidates = [
            OperatorCandidate(CandidateKey(logical_op, "fp32", "pytorch-cudnn", "reference"),
                             _reference_factory,
                             lambda module, context: (module.weight.device.type == "cuda", "FP32 GPU reference requires CUDA")),
            OperatorCandidate(CandidateKey(logical_op, "fp16", "pytorch-cudnn", "fp16-module-boundary",
                                           {"parameter_storage": "fp32", "output_dtype": "input"}),
                             fp16_factory, cuda_fp16_capability),
            OperatorCandidate(CandidateKey(logical_op, "bf16", "pytorch-cudnn", "bf16-module-boundary",
                                           {"parameter_storage": "fp32", "output_dtype": "input"}),
                             bf16_factory, cuda_bf16_capability),
        ]
        fp8_candidates = (
            ("pytorch-scaled-mm", "fp8-e4m3fn-scaled-mm", fp8_conv2d_capability, fp8_conv2d_factory)
            if logical_op == "conv2d" else
            ("pytorch-scaled-mm", "fp8-e4m3fn-scaled-mm", fp8_conv_transpose2d_capability, fp8_conv_transpose2d_factory)
        )
        backend, implementation, capability, factory = fp8_candidates
        candidates.append(OperatorCandidate(
            CandidateKey(logical_op, "fp8", backend, implementation,
                         {"scale_mode": "per_tensor_symmetric", "compute_dtype": "bf16", "alignment": 16}),
            factory, capability,
        ))
        if logical_op == "conv2d":
            candidates.append(OperatorCandidate(
                CandidateKey(logical_op, "fp8", "triton", "fp8-e4m3fn-im2col",
                             {"scale_mode": "per_tensor_symmetric", "compute_dtype": "bf16", "alignment": 16}),
                fp8_triton_conv2d_factory, fp8_triton_conv2d_capability,
            ))
        else:
            candidates.append(OperatorCandidate(
                CandidateKey(logical_op, "fp8", "triton", "fp8-e4m3fn-transpose-scatter",
                             {"scale_mode": "per_tensor_symmetric", "compute_dtype": "bf16", "alignment": 16}),
                fp8_triton_conv_transpose2d_factory, fp8_triton_conv_transpose2d_capability,
            ))
        catalog.register(CatalogEntry(OperatorSpec(logical_op, module_type, attributes={"floating": True}),
                                      tuple(candidates), notes="GPU floating transform candidates."))

    try:
        from compressai.layers import GDN
    except ImportError:
        GDN = None
    if GDN is not None:
        for logical_op, inverse in (("gdn", False), ("igdn", True)):
            candidates = [
                OperatorCandidate(CandidateKey(logical_op, "fp32", "pytorch-cudnn", "compressai-reference"),
                                 _reference_factory,
                                 lambda module, context, expected=inverse: (module.beta.device.type == "cuda" and bool(module.inverse) == expected,
                                                                           "GDN reference requires matching inverse mode on CUDA")),
                OperatorCandidate(CandidateKey(logical_op, "fp16", "pytorch-cudnn", "fp16-parameterized-gdn",
                                               {"parameter_storage": "fp32", "output_dtype": "input"}),
                                 gdn_fp16_factory,
                                 lambda module, context, expected=inverse: (bool(module.inverse) == expected and cuda_fp16_capability(module, context)[0],
                                                                           "GDN FP16 candidate requires CUDA and matching inverse mode")),
                OperatorCandidate(CandidateKey(logical_op, "bf16", "pytorch-cudnn", "bf16-parameterized-gdn",
                                               {"parameter_storage": "fp32", "output_dtype": "input"}),
                                 gdn_bf16_factory,
                                 lambda module, context, expected=inverse: (bool(module.inverse) == expected and cuda_bf16_capability(module, context)[0],
                                                                           "GDN BF16 candidate requires CUDA and matching inverse mode")),
            ]
            candidates.append(OperatorCandidate(
                CandidateKey(logical_op, "fp8", "pytorch-scaled-mm", "fp8-e4m3fn-scaled-mm-gdn",
                             {"scale_mode": "per_tensor_symmetric", "compute_dtype": "bf16", "alignment": 16}),
                fp8_igdn_factory if inverse else fp8_gdn_factory,
                lambda module, context, expected=inverse: (
                    bool(getattr(module, "inverse", False)) == expected
                    and fp8_gdn_capability(module, context)[0],
                    "FP8 GDN candidate requires matching inverse mode and CUDA capability",
                ),
            ))
            candidates.append(OperatorCandidate(
                CandidateKey(logical_op, "fp8", "triton", "fp8-e4m3fn-triton-gdn",
                             {"scale_mode": "per_tensor_symmetric", "compute_dtype": "bf16", "alignment": 16}),
                fp8_triton_gdn_factory,
                lambda module, context, expected=inverse: (
                    bool(getattr(module, "inverse", False)) == expected
                    and fp8_triton_gdn_capability(module, context)[0],
                    "Triton FP8 GDN candidate requires matching inverse mode and CUDA capability",
                ),
            ))
            catalog.register(CatalogEntry(OperatorSpec(logical_op, GDN, attributes={"inverse": inverse, "floating": True}),
                                          tuple(candidates), notes="GDN/IGDN precision candidates; evaluate full RD before use."))

    try:
        from compressai.entropy_models import EntropyBottleneck
    except ImportError:
        EntropyBottleneck = None
    if EntropyBottleneck is not None:
        catalog.register(CatalogEntry(
            OperatorSpec("entropy_bottleneck", EntropyBottleneck, stateful=True,
                         attributes={"floating_likelihood": True, "discrete_coder_locked": True}),
            (OperatorCandidate(CandidateKey("entropy_bottleneck", "fp32", "pytorch-cudnn", "compressai-reference",
                                            {"coder_state": "integer-locked"}),
                               _reference_factory,
                               lambda module, context: (module.quantiles.device.type == "cuda", "entropy reference requires CUDA")),),
            precision_locked=True,
            notes="Entropy quantization/CDF/index/coder state is precision-locked.",
        ))
    return catalog
