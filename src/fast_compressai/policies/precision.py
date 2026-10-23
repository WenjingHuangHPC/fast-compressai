"""Serializable, reporting-only precision policies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping


SCHEMA_VERSION = 1
SUPPORTED_PRECISIONS = frozenset({"fp32", "fp16", "bf16", "fp8"})


class PrecisionPolicyError(ValueError):
    """Raised when a precision policy is invalid or cannot be serialized."""


class PrecisionPolicyKind(str, Enum):
    FP32 = "fp32"
    UNIFORM = "uniform"
    COMPONENT = "component"
    OPERATOR = "operator"
    CUSTOM = "custom"


@dataclass(frozen=True)
class PrecisionRequest:
    """Stable context passed to a custom policy resolver."""

    component: str | None = None
    operator: str | None = None
    available_precisions: tuple[str, ...] = ()
    context: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PrecisionResolution:
    """Requested precision and its policy source."""

    precision: str
    source: str


@dataclass(frozen=True)
class PrecisionPolicy:
    """An immutable precision request policy.

    This object describes selections for reporting. It never mutates a model
    and never performs dtype conversion.
    """

    kind: PrecisionPolicyKind
    precision: str | None = None
    assignments: tuple[tuple[str, str], ...] = ()
    default: str = "fp32"
    name: str | None = None
    resolver: Callable[[PrecisionRequest], str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PrecisionPolicyKind):
            try:
                object.__setattr__(self, "kind", PrecisionPolicyKind(self.kind))
            except (TypeError, ValueError) as exc:
                raise PrecisionPolicyError(f"unknown precision policy kind: {self.kind!r}") from exc
        if self.kind == PrecisionPolicyKind.CUSTOM:
            if not callable(self.resolver):
                raise PrecisionPolicyError("custom policy requires a callable resolver")
        elif self.resolver is not None:
            raise PrecisionPolicyError("resolver is only valid for custom policies")

        if self.precision is not None:
            _validate_precision(self.precision)
        _validate_precision(self.default)
        if self.kind == PrecisionPolicyKind.UNIFORM and self.precision is None:
            raise PrecisionPolicyError("uniform policy requires a precision")
        if self.kind != PrecisionPolicyKind.UNIFORM and self.precision is not None:
            raise PrecisionPolicyError("precision is only valid for uniform policies")
        if self.kind == PrecisionPolicyKind.CUSTOM and self.assignments:
            raise PrecisionPolicyError("custom policy cannot contain assignments")
        if self.kind in (PrecisionPolicyKind.FP32, PrecisionPolicyKind.UNIFORM) and self.assignments:
            raise PrecisionPolicyError(f"{self.kind.value} policy cannot contain assignments")

        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for key, value in self.assignments:
            if not isinstance(key, str) or not key.strip():
                raise PrecisionPolicyError("policy assignment keys must be non-empty strings")
            if key in seen:
                raise PrecisionPolicyError(f"duplicate policy assignment {key!r}")
            _validate_precision(value)
            seen.add(key)
            normalized.append((key, value))
        object.__setattr__(self, "assignments", tuple(sorted(normalized)))

    @classmethod
    def fp32(cls) -> "PrecisionPolicy":
        return cls(PrecisionPolicyKind.FP32, default="fp32")

    @classmethod
    def uniform(cls, precision: str) -> "PrecisionPolicy":
        _validate_precision(precision)
        return cls(PrecisionPolicyKind.UNIFORM, precision=precision, default="fp32")

    @classmethod
    def component(
        cls,
        assignments: Mapping[str, str],
        *,
        default: str = "fp32",
    ) -> "PrecisionPolicy":
        return cls(
            PrecisionPolicyKind.COMPONENT,
            assignments=tuple(assignments.items()),
            default=default,
        )

    @classmethod
    def operator(
        cls,
        assignments: Mapping[str, str],
        *,
        default: str = "fp32",
    ) -> "PrecisionPolicy":
        return cls(
            PrecisionPolicyKind.OPERATOR,
            assignments=tuple(assignments.items()),
            default=default,
        )

    @classmethod
    def custom(
        cls,
        resolver: Callable[[PrecisionRequest], str],
        *,
        name: str = "custom",
    ) -> "PrecisionPolicy":
        if not isinstance(name, str) or not name.strip():
            raise PrecisionPolicyError("custom policy name must be a non-empty string")
        return cls(PrecisionPolicyKind.CUSTOM, name=name, resolver=resolver)

    @property
    def assignments_map(self) -> Mapping[str, str]:
        return dict(self.assignments)

    @property
    def serializable(self) -> bool:
        return self.kind != PrecisionPolicyKind.CUSTOM

    def resolve(self, request: PrecisionRequest) -> PrecisionResolution:
        """Resolve a request without consulting implementations or a registry."""

        if not isinstance(request, PrecisionRequest):
            raise PrecisionPolicyError("policy requests must be PrecisionRequest instances")
        if self.kind == PrecisionPolicyKind.FP32:
            return PrecisionResolution("fp32", "default")
        if self.kind == PrecisionPolicyKind.UNIFORM:
            return PrecisionResolution(self.precision or "fp32", "policy")
        if self.kind == PrecisionPolicyKind.COMPONENT:
            if request.component in self.assignments_map:
                return PrecisionResolution(self.assignments_map[request.component], "policy")
            return PrecisionResolution(self.default, "default")
        if self.kind == PrecisionPolicyKind.OPERATOR:
            if request.operator in self.assignments_map:
                return PrecisionResolution(self.assignments_map[request.operator], "policy")
            return PrecisionResolution(self.default, "default")
        assert self.resolver is not None
        try:
            value = self.resolver(request)
        except Exception as exc:
            raise PrecisionPolicyError(f"custom precision policy failed: {exc}") from exc
        _validate_precision(value)
        return PrecisionResolution(value, "custom")

    def to_dict(self) -> dict[str, Any]:
        if not self.serializable:
            raise PrecisionPolicyError(
                "custom precision policies are runtime-only and cannot be serialized"
            )
        policy: dict[str, Any] = {"type": self.kind.value}
        if self.kind == PrecisionPolicyKind.UNIFORM:
            policy["precision"] = self.precision
        elif self.kind in (PrecisionPolicyKind.COMPONENT, PrecisionPolicyKind.OPERATOR):
            policy["assignments"] = dict(self.assignments)
            policy["default"] = self.default
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "precision_policy",
            "policy": policy,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrecisionPolicy":
        if not isinstance(value, Mapping):
            raise PrecisionPolicyError("serialized precision policy must be a mapping")
        if set(value) != {"schema_version", "kind", "policy"}:
            raise PrecisionPolicyError(
                "serialized precision policy must contain exactly schema_version, kind, and policy"
            )
        if value["schema_version"] != SCHEMA_VERSION:
            raise PrecisionPolicyError(
                f"unsupported precision policy schema version: {value['schema_version']!r}"
            )
        if value["kind"] != "precision_policy" or not isinstance(value["policy"], Mapping):
            raise PrecisionPolicyError("invalid precision policy envelope")
        payload = value["policy"]
        policy_type = payload.get("type")
        if policy_type == PrecisionPolicyKind.FP32.value:
            expected = {"type"}
            if set(payload) != expected:
                raise PrecisionPolicyError("fp32 policy contains unknown fields")
            return cls.fp32()
        if policy_type == PrecisionPolicyKind.UNIFORM.value:
            if set(payload) != {"type", "precision"}:
                raise PrecisionPolicyError("uniform policy contains unknown fields")
            return cls.uniform(payload["precision"])
        if policy_type in (PrecisionPolicyKind.COMPONENT.value, PrecisionPolicyKind.OPERATOR.value):
            if set(payload) != {"type", "assignments", "default"}:
                raise PrecisionPolicyError(f"{policy_type} policy contains unknown fields")
            assignments = payload["assignments"]
            if not isinstance(assignments, Mapping):
                raise PrecisionPolicyError("policy assignments must be a mapping")
            factory = cls.component if policy_type == PrecisionPolicyKind.COMPONENT.value else cls.operator
            return factory(dict(assignments), default=payload["default"])
        raise PrecisionPolicyError(f"unsupported serialized precision policy type: {policy_type!r}")


def _validate_precision(value: Any) -> None:
    if not isinstance(value, str) or value not in SUPPORTED_PRECISIONS:
        allowed = ", ".join(sorted(SUPPORTED_PRECISIONS))
        raise PrecisionPolicyError(
            f"unsupported precision {value!r}; supported precisions are: {allowed}"
        )
