"""Public configuration objects for Fast-CompressAI."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when a compile configuration is invalid."""


@dataclass(frozen=True)
class AccelerationConfig:
    """Public constraints for the automatic acceleration workflow.

    ``P`` is represented internally as ``p_ch``: latent channels per
    independently coded GA-ANS chunk. ``budget`` is an RD/quality budget and
    is intentionally independent of coding-time constraints.
    """

    p_ch: int | None = None
    budget: float = 0.0
    coder: str = "warp_div"
    device: str | None = None
    profile_path: str | None = None
    precision_config_path: str | None = None
    execution_mode: str = "sequential"

    def __post_init__(self) -> None:
        if self.p_ch is not None and (type(self.p_ch) is not int or self.p_ch <= 0):
            raise ConfigurationError("P (p_ch) must be a positive integer or None")
        if self.coder not in {"warp_div", "rans"}:
            raise ConfigurationError("coder must be 'warp_div' or 'rans'")
        if isinstance(self.budget, bool) or not isinstance(self.budget, (int, float)):
            raise ConfigurationError("budget must be a non-negative finite RD/quality value")
        if self.budget < 0 or not math.isfinite(float(self.budget)):
            raise ConfigurationError("budget must be a non-negative finite RD/quality value")
        if self.device is not None and not isinstance(self.device, str):
            raise ConfigurationError("device must be a string or None")
        if self.profile_path is not None and not isinstance(self.profile_path, str):
            raise ConfigurationError("profile_path must be a string or None")
        if self.precision_config_path is not None and not isinstance(self.precision_config_path, str):
            raise ConfigurationError("precision_config_path must be a string or None")
        if self.execution_mode not in {"sequential", "overlap"}:
            raise ConfigurationError("execution_mode must be 'sequential' or 'overlap'")

    @classmethod
    def from_value(
        cls,
        value: "AccelerationConfig | Mapping[str, Any] | None",
        *,
        P: int | None = None,
        budget: float | None = None,
        device: str | None = None,
        coder: str | None = None,
        precision_config_path: str | None = None,
    ) -> "AccelerationConfig":
        if value is None:
            return cls(p_ch=P, budget=0.0 if budget is None else budget,
                       device=device, coder="warp_div" if coder is None else coder)
        if isinstance(value, cls):
            updates: dict[str, Any] = {}
            if P is not None:
                updates["p_ch"] = P
            if budget is not None:
                updates["budget"] = budget
            if device is not None:
                updates["device"] = device
            if coder is not None:
                updates["coder"] = coder
            if precision_config_path is not None:
                updates["precision_config_path"] = precision_config_path
            if updates:
                raise ConfigurationError("do not combine config with P/budget/device/coder overrides")
            return value
        if isinstance(value, Mapping):
            data = dict(value)
            if "P" in data:
                if "p_ch" in data:
                    raise ConfigurationError("use only one of P and p_ch")
                data["p_ch"] = data.pop("P")
            for key, override in (("p_ch", P), ("budget", budget), ("device", device), ("coder", coder), ("precision_config_path", precision_config_path)):
                if override is not None:
                    if key in data:
                        raise ConfigurationError(f"{key} specified both in config and as an argument")
                    data[key] = override
            valid = {"p_ch", "budget", "coder", "device", "profile_path", "precision_config_path", "execution_mode"}
            unknown = set(data) - valid
            if unknown:
                names = ", ".join(sorted(map(str, unknown)))
                raise ConfigurationError(f"unknown acceleration config field(s): {names}")
            try:
                return cls(**data)
            except TypeError as exc:
                raise ConfigurationError(str(exc)) from exc
        raise ConfigurationError("config must be None, AccelerationConfig, or a mapping")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompileConfig:
    """Configuration for the reference-first compile path."""

    device: str | None = None
    backend: str = "reference"
    enable_reference_fallback: bool = True
    enable_operator_fallback: bool = True
    default_precision: str = "fp32"

    def __post_init__(self) -> None:
        if self.device is not None and not isinstance(self.device, str):
            try:
                normalized_device = str(self.device)
            except Exception as exc:  # pragma: no cover - defensive guard
                raise ConfigurationError("device must be a string or None") from exc
            object.__setattr__(self, "device", normalized_device)

        if self.backend != "reference":
            raise ConfigurationError(
                f"unsupported backend {self.backend!r}; the MVP only supports 'reference'"
            )
        if not isinstance(self.enable_reference_fallback, bool):
            raise ConfigurationError("enable_reference_fallback must be a bool")
        if not isinstance(self.enable_operator_fallback, bool):
            raise ConfigurationError("enable_operator_fallback must be a bool")
        if self.default_precision != "fp32":
            raise ConfigurationError(
                f"unsupported default precision {self.default_precision!r}; "
                "the default must remain 'fp32'"
            )

    @classmethod
    def from_value(cls, value: "CompileConfig | Mapping[str, Any] | None") -> "CompileConfig":
        """Normalize ``None``, a config instance, or a mapping to a config."""

        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            valid_fields = {
                "device",
                "backend",
                "enable_reference_fallback",
                "enable_operator_fallback",
                "default_precision",
            }
            unknown = set(value) - valid_fields
            if unknown:
                names = ", ".join(sorted(map(str, unknown)))
                raise ConfigurationError(f"unknown compile configuration field(s): {names}")
            try:
                return cls(**dict(value))
            except TypeError as exc:
                raise ConfigurationError(str(exc)) from exc
        raise ConfigurationError(
            "config must be None, CompileConfig, or a mapping of configuration fields"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the configuration."""

        return asdict(self)
