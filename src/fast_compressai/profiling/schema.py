"""Portable, versioned profile records for precision and coder selection."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

PROFILE_SCHEMA_VERSION = 3
_PROFILE_KINDS = {"operator", "coder"}
_PROVENANCE = {"measured", "estimated", "imported"}


class ProfileError(ValueError):
    """Raised when a profiling record is malformed or incompatible."""


@dataclass(frozen=True)
class ProfileRecord:
    """One candidate measurement in a reproducible model/device context.

    ``psnr_drop_db`` and ``bpp_delta`` are deliberately separate metrics.
    ``quality_delta`` remains a deprecated input alias for schema-v1 callers.
    """

    kind: str
    target: str
    candidate: str
    context: Mapping[str, Any]
    latency_ms: float | None = None
    throughput: float | None = None
    compression_ratio: float | None = None
    psnr_drop_db: float | None = None
    psnr_db: float | None = None
    bpp: float | None = None
    bpp_delta: float | None = None
    rd_cost: float | None = None
    thr_gain: float | None = None
    provenance: str = "measured"
    repetitions: int = 1
    warmup: int = 0
    latency_std_ms: float | None = None
    setup_cost_ms: float | None = None
    measurement_scope: str = "unknown"
    quality_metric: str | None = None
    quality_unit: str | None = None
    unavailable_reason: str | None = None
    quality_delta: float | None = None  # deprecated v1 compatibility alias

    def __post_init__(self) -> None:
        if self.kind not in _PROFILE_KINDS:
            raise ProfileError("profile kind must be 'operator' or 'coder'")
        for name, value in (("target", self.target), ("candidate", self.candidate)):
            if not isinstance(value, str) or not value.strip():
                raise ProfileError(f"{name} must be a non-empty string")
        if self.provenance not in _PROVENANCE:
            raise ProfileError(f"provenance must be one of {sorted(_PROVENANCE)}")
        if type(self.repetitions) is not int or self.repetitions < 1:
            raise ProfileError("repetitions must be a positive integer")
        if type(self.warmup) is not int or self.warmup < 0:
            raise ProfileError("warmup must be a non-negative integer")
        if not isinstance(self.context, Mapping):
            raise ProfileError("context must be a mapping")
        _json_value(dict(self.context), "context")
        numeric_fields = (
            "latency_ms", "throughput", "compression_ratio", "psnr_drop_db", "psnr_db",
            "bpp", "bpp_delta", "rd_cost", "thr_gain", "latency_std_ms", "setup_cost_ms", "quality_delta",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            ):
                raise ProfileError(f"{name} must be a finite number or None")
        for name in ("latency_ms", "throughput", "latency_std_ms", "setup_cost_ms"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ProfileError(f"{name} must be non-negative")
        if self.compression_ratio is not None and self.compression_ratio <= 0:
            raise ProfileError("compression_ratio must be positive")
        if self.psnr_drop_db is not None and self.psnr_drop_db < 0:
            raise ProfileError("psnr_drop_db must be non-negative")
        if self.bpp is not None and self.bpp <= 0:
            raise ProfileError("bpp must be positive")
        if self.measurement_scope not in {"operator", "forward", "compress", "decompress", "unknown"}:
            raise ProfileError("measurement_scope is not recognized")
        for name in ("quality_metric", "quality_unit", "unavailable_reason"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ProfileError(f"{name} must be a string or None")
        if self.psnr_drop_db is None and self.quality_delta is not None:
            object.__setattr__(self, "psnr_drop_db", float(self.quality_delta))
        has_measurement = any(
            getattr(self, name) is not None
            for name in ("latency_ms", "throughput", "compression_ratio", "psnr_drop_db", "psnr_db", "bpp", "bpp_delta", "rd_cost", "thr_gain")
        )
        if self.unavailable_reason is None and not has_measurement:
            raise ProfileError("profile record needs measurements or an unavailable_reason")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileRecord":
        expected = {
            "kind", "target", "candidate", "context", "latency_ms", "throughput",
            "compression_ratio", "psnr_drop_db", "psnr_db", "bpp", "bpp_delta", "rd_cost", "thr_gain", "provenance", "repetitions",
            "warmup", "latency_std_ms", "setup_cost_ms", "measurement_scope", "quality_metric",
            "quality_unit", "unavailable_reason", "quality_delta",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ProfileError("profile record has missing or unknown fields for schema v3")
        return cls(**dict(value))


@dataclass(frozen=True)
class ProfileSet:
    """A versioned, JSON-safe set of comparable profile records."""

    records: tuple[ProfileRecord, ...]
    schema_version: int = PROFILE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ProfileError(f"unsupported profile schema version {self.schema_version!r}")
        if any(not isinstance(record, ProfileRecord) for record in self.records):
            raise ProfileError("records must contain ProfileRecord values")

    def to_dict(self) -> dict[str, Any]:
        records = sorted(
            (record.to_dict() for record in self.records),
            key=lambda record: (record["kind"], record["target"], record["candidate"]),
        )
        return {
            "schema_version": self.schema_version,
            "kind": "fast_compressai_profile_set",
            "records": records,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ProfileSet":
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "kind", "records"}:
            raise ProfileError("profile set has missing or unknown fields")
        if value["kind"] != "fast_compressai_profile_set":
            raise ProfileError("invalid profile set kind")
        records = value["records"]
        if not isinstance(records, list):
            raise ProfileError("profile set records must be a list")
        return cls(
            records=tuple(ProfileRecord.from_dict(record) for record in records),
            schema_version=value["schema_version"],
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str) -> "ProfileSet":
        try:
            decoded = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProfileError("profile set is not valid JSON") from exc
        return cls.from_dict(decoded)


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ProfileError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProfileError(f"{path} keys must be strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    raise ProfileError(f"{path} contains unsupported value {type(value).__name__}")
