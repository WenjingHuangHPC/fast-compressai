"""Structured operator candidate descriptors for backend/precision expansion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping

from torch import nn

from .errors import RegistryError


@dataclass(frozen=True)
class CandidateKey:
    logical_op: str
    precision: str
    backend: str
    implementation: str
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("logical_op", "precision", "backend", "implementation"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RegistryError(f"candidate {name} must be a non-empty string")
        try:
            encoded = json.dumps(dict(self.config), sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RegistryError("candidate config must be finite JSON data") from exc
        object.__setattr__(self, "config", json.loads(encoded))

    @property
    def stable_id(self) -> str:
        config_id = json.dumps(dict(self.config), sort_keys=True, separators=(",", ":"))
        return f"{self.logical_op}/{self.precision}/{self.backend}/{self.implementation}/{config_id}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CandidateKey":
        if set(value) != {"logical_op", "precision", "backend", "implementation", "config"}:
            raise RegistryError("candidate key has missing or unknown fields")
        return cls(**dict(value))


@dataclass(frozen=True)
class OperatorInstance:
    instance_id: str
    module_path: str
    logical_op: str
    module_type: str
    config: Mapping[str, Any]
    stateful: bool = False


@dataclass(frozen=True)
class OperatorSpec:
    logical_op: str
    op_type: type[nn.Module]
    reference_backend: str = "pytorch-cudnn"
    stateful: bool = False
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OperatorCandidate:
    key: CandidateKey
    factory: Callable[[nn.Module, Mapping[str, Any]], nn.Module]
    capability: Callable[[nn.Module, Mapping[str, Any]], tuple[bool, str | None]]
    inference_only: bool = True

    def __post_init__(self) -> None:
        if not callable(self.factory) or not callable(self.capability):
            raise RegistryError("operator candidate factory and capability must be callable")

    def build(self, original: nn.Module, context: Mapping[str, Any] | None = None) -> nn.Module:
        context = {} if context is None else dict(context)
        supported, reason = self.capability(original, context)
        if not supported:
            raise RegistryError(reason or f"candidate {self.key.stable_id} is not applicable")
        replacement = self.factory(original, context)
        if not isinstance(replacement, nn.Module):
            raise RegistryError(f"candidate {self.key.stable_id} factory must return nn.Module")
        replacement.load_state_dict(original.state_dict(), strict=True)
        replacement.train(original.training)
        for source, target in zip(original.parameters(), replacement.parameters()):
            target.requires_grad_(source.requires_grad)
        return replacement
