"""Stable contracts used by the explicit operator registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping

from .errors import RegistryError

Capability = Callable[["OperatorContext"], bool]
Implementation = Callable[..., Any]


@dataclass(frozen=True)
class OperatorContext:
    """Read-only metadata used to decide whether an implementation is usable."""

    device: str | None = None
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    requested_precision: str | None = None

    def with_precision(self, precision: str | None) -> "OperatorContext":
        return OperatorContext(self.device, self.dtype, self.shape, precision)

    @classmethod
    def from_value(cls, value: "OperatorContext | Mapping[str, Any] | None") -> "OperatorContext":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            valid = {"device", "dtype", "shape", "requested_precision"}
            unknown = set(value) - valid
            if unknown:
                names = ", ".join(sorted(map(str, unknown)))
                raise RegistryError(f"unknown operator context field(s): {names}")
            shape = value.get("shape")
            if shape is not None:
                try:
                    shape = tuple(int(item) for item in shape)
                except (TypeError, ValueError) as exc:
                    raise RegistryError("operator context shape must be an integer sequence") from exc
            return cls(
                device=None if value.get("device") is None else str(value.get("device")),
                dtype=None if value.get("dtype") is None else str(value.get("dtype")),
                shape=shape,
                requested_precision=value.get("requested_precision"),
            )
        raise RegistryError("context must be None, OperatorContext, or a mapping")

    @classmethod
    def from_inputs(cls, *args: Any, **kwargs: Any) -> "OperatorContext":
        """Infer metadata from the first tensor-like argument, if available."""

        def find(value: Any) -> Any:
            if hasattr(value, "device") and hasattr(value, "dtype") and hasattr(value, "shape"):
                return value
            if isinstance(value, (tuple, list)):
                for item in value:
                    found = find(item)
                    if found is not None:
                        return found
            if isinstance(value, Mapping):
                for item in value.values():
                    found = find(item)
                    if found is not None:
                        return found
            return None

        tensor = find(args)
        if tensor is None:
            tensor = find(kwargs)
        if tensor is None:
            return cls()
        return cls(
            device=str(tensor.device),
            dtype=str(tensor.dtype),
            shape=tuple(int(item) for item in tensor.shape),
        )


@dataclass(frozen=True)
class OperatorImplementation:
    """One callable candidate for a registered operator."""

    precision: str
    function: Implementation
    capability: Capability | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise RegistryError("implementation precision must be a non-empty string")
        if not callable(self.function):
            raise RegistryError(f"implementation {self.precision!r} must be callable")
        if self.capability is not None and not callable(self.capability):
            raise RegistryError(f"capability for {self.precision!r} must be callable")


@dataclass(frozen=True)
class OperatorDefinition:
    """Immutable registration record for one canonical operator key."""

    key: str
    reference: Implementation
    implementations: tuple[OperatorImplementation, ...] = ()
    fallback: str = "fp32"
    op_type: type | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise RegistryError("operator key must be a non-empty string")
        if not callable(self.reference):
            raise RegistryError("operator reference must be callable")
        if self.fallback != "fp32":
            raise RegistryError("the only supported fallback is 'fp32'")
        if self.op_type is not None and not isinstance(self.op_type, type):
            raise RegistryError("op_type must be a type or None")
        by_precision: dict[str, OperatorImplementation] = {}
        for implementation in self.implementations:
            if not isinstance(implementation, OperatorImplementation):
                raise RegistryError("implementations must contain OperatorImplementation values")
            if implementation.precision == "fp32":
                raise RegistryError("'fp32' is reserved for the reference implementation")
            if implementation.precision in by_precision:
                raise RegistryError(f"duplicate implementation precision {implementation.precision!r}")
            by_precision[implementation.precision] = implementation
        object.__setattr__(self, "implementations", tuple(by_precision.values()))

    @property
    def implementation_map(self) -> Mapping[str, OperatorImplementation]:
        return MappingProxyType({item.precision: item for item in self.implementations})


@dataclass(frozen=True)
class OperatorDecision:
    """The explainable result of one implementation selection."""

    key: str
    requested_precision: str | None
    selected_precision: str
    implementation: str
    accelerated: bool
    fallback_used: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
