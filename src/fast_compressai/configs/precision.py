"""Versioned JSON persistence for CAIEC per-instance precision assignments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

PRECISION_CONFIG_SCHEMA_VERSION = 3
RD_COST_FORMULA = "log10(bpp/bpp_ref)+(psnr_ref-psnr)"


class PrecisionConfigError(ValueError):
    """Raised when a precision config cannot safely be loaded or applied."""


@dataclass(frozen=True)
class PrecisionAssignment:
    instance_id: str
    module_path: str
    logical_op: str
    candidate_key: Mapping[str, Any]
    precision: str
    backend: str
    category: str
    probe_rd_cost: float
    probe_thr_gain: float
    fallback_reason: str | None = None
    selection_reason: str | None = None

    def __post_init__(self) -> None:
        for name in ("instance_id", "module_path", "logical_op", "precision", "backend", "category"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise PrecisionConfigError(f"assignment {name} must be non-empty")
        _finite(self.probe_rd_cost, "probe_rd_cost")
        _finite(self.probe_thr_gain, "probe_thr_gain")
        if not isinstance(self.candidate_key, Mapping):
            raise PrecisionConfigError("candidate_key must be a mapping")
        _json_value(dict(self.candidate_key), "candidate_key")


@dataclass(frozen=True)
class PrecisionConfig:
    model_name: str
    model_fingerprint: str
    dataset_name: str
    dataset_fingerprint: str
    device_type: str
    device_capability: tuple[int, int]
    coder: str
    p_ch: int | None
    epsilon: float
    rd_cost_formula: str
    selector_version: str
    catalog_fingerprint: str
    assignments: tuple[PrecisionAssignment, ...]
    probe_records: tuple[Mapping[str, Any], ...]
    validated_rd_cost: float
    budget_met: bool
    created_utc: str
    dataset_sample_count: int
    profile_cost_ms: float
    recovery_order: tuple[str, ...] = ()
    recovery_steps: tuple[Mapping[str, Any], ...] = ()
    records_path: str | None = None
    recovery_path: str | None = None
    schema_version: int = PRECISION_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != PRECISION_CONFIG_SCHEMA_VERSION:
            raise PrecisionConfigError(f"unsupported precision config schema {self.schema_version!r}")
        for name in ("model_name", "dataset_name", "device_type", "coder", "selector_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise PrecisionConfigError(f"{name} must be a non-empty string")
        for name in ("model_fingerprint", "dataset_fingerprint", "catalog_fingerprint"):
            if not _is_sha256(getattr(self, name)):
                raise PrecisionConfigError(f"{name} must be a SHA-256 hex digest")
        if self.rd_cost_formula != RD_COST_FORMULA:
            raise PrecisionConfigError("unsupported RD Cost formula")
        _finite(self.epsilon, "epsilon")
        _finite(self.validated_rd_cost, "validated_rd_cost")
        _finite(self.profile_cost_ms, "profile_cost_ms")
        if self.epsilon < 0 or self.profile_cost_ms < 0:
            raise PrecisionConfigError("epsilon and profile cost must be non-negative")
        if self.p_ch is not None and (type(self.p_ch) is not int or self.p_ch <= 0):
            raise PrecisionConfigError("p_ch must be a positive integer or None")
        if type(self.dataset_sample_count) is not int or self.dataset_sample_count < 1:
            raise PrecisionConfigError("dataset_sample_count must be positive")
        if len(self.device_capability) != 2 or any(type(item) is not int for item in self.device_capability):
            raise PrecisionConfigError("device_capability must be an integer pair")
        for name, records in (("probe_records", self.probe_records), ("recovery_steps", self.recovery_steps)):
            for record in records:
                if not isinstance(record, Mapping):
                    raise PrecisionConfigError(f"{name} must contain mappings")
                _json_value(dict(record), name)
        for name in ("records_path", "recovery_path"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise PrecisionConfigError(f"{name} must be a non-empty string or None")
        paths = [assignment.module_path for assignment in self.assignments]
        if len(paths) != len(set(paths)):
            raise PrecisionConfigError("precision assignments contain duplicate module paths")
        if any(not isinstance(item, PrecisionAssignment) for item in self.assignments):
            raise PrecisionConfigError("assignments must contain PrecisionAssignment values")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "fast_compressai_precision_config",
            "model_name": self.model_name,
            "model_fingerprint": self.model_fingerprint,
            "dataset_name": self.dataset_name,
            "dataset_fingerprint": self.dataset_fingerprint,
            "dataset_sample_count": self.dataset_sample_count,
            "device_type": self.device_type,
            "device_capability": list(self.device_capability),
            "coder": self.coder,
            "p_ch": self.p_ch,
            "epsilon": self.epsilon,
            "rd_cost_formula": self.rd_cost_formula,
            "selector_version": self.selector_version,
            "catalog_fingerprint": self.catalog_fingerprint,
            "assignments": [asdict(assignment) for assignment in self.assignments],
            "probe_records": [_json_value(dict(record), "probe_record") for record in self.probe_records],
            "validated_rd_cost": self.validated_rd_cost,
            "budget_met": self.budget_met,
            "recovery_order": list(self.recovery_order),
            "recovery_steps": [_json_value(dict(record), "recovery_step") for record in self.recovery_steps],
            "records_path": self.records_path,
            "recovery_path": self.recovery_path,
            "profile_cost_ms": self.profile_cost_ms,
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PrecisionConfig":
        expected = {
            "schema_version", "kind", "model_name", "model_fingerprint", "dataset_name",
            "dataset_fingerprint", "dataset_sample_count", "device_type", "device_capability",
            "coder", "p_ch", "epsilon", "rd_cost_formula", "selector_version",
            "catalog_fingerprint", "assignments", "probe_records", "validated_rd_cost", "budget_met",
            "recovery_order", "recovery_steps", "records_path", "recovery_path", "profile_cost_ms", "created_utc",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise PrecisionConfigError("precision config has missing or unknown fields")
        if value["kind"] != "fast_compressai_precision_config":
            raise PrecisionConfigError("invalid precision config kind")
        try:
            return cls(
                schema_version=value["schema_version"],
                model_name=value["model_name"],
                model_fingerprint=value["model_fingerprint"],
                dataset_name=value["dataset_name"],
                dataset_fingerprint=value["dataset_fingerprint"],
                dataset_sample_count=value["dataset_sample_count"],
                device_type=value["device_type"],
                device_capability=tuple(value["device_capability"]),
                coder=value["coder"],
                p_ch=value["p_ch"],
                epsilon=value["epsilon"],
                rd_cost_formula=value["rd_cost_formula"],
                selector_version=value["selector_version"],
                catalog_fingerprint=value["catalog_fingerprint"],
                assignments=tuple(PrecisionAssignment(**item) for item in value["assignments"]),
                probe_records=tuple(value["probe_records"]),
                validated_rd_cost=value["validated_rd_cost"],
                budget_met=value["budget_met"],
                recovery_order=tuple(value["recovery_order"]),
                recovery_steps=tuple(value["recovery_steps"]),
                records_path=value["records_path"],
                recovery_path=value["recovery_path"],
                profile_cost_ms=value["profile_cost_ms"],
                created_utc=value["created_utc"],
            )
        except (TypeError, ValueError, KeyError) as exc:
            if isinstance(exc, PrecisionConfigError):
                raise
            raise PrecisionConfigError(f"invalid precision config: {exc}") from exc

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> "PrecisionConfig":
        try:
            data = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise PrecisionConfigError("precision config is not valid JSON") from exc
        return cls.from_dict(data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(self.to_json())
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        return path

    @classmethod
    def load(cls, path: str | Path) -> "PrecisionConfig":
        path = Path(path)
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise PrecisionConfigError(f"cannot read precision config {path}: {exc}") from exc

    @classmethod
    def default_path(cls, root: str | Path, model_name: str, dataset_name: str) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return Path(root) / _safe_name(model_name) / f"{_safe_name(dataset_name)}-{stamp}.json"

    def validate_compatibility(self, **actual: Any) -> None:
        fields = (
            "model_fingerprint", "dataset_fingerprint", "device_type", "device_capability",
            "coder", "p_ch", "catalog_fingerprint",
        )
        mismatches = [name for name in fields if getattr(self, name) != actual.get(name)]
        if mismatches:
            raise PrecisionConfigError("precision config is incompatible: " + ", ".join(mismatches))


def model_fingerprint(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fingerprint_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PrecisionConfigError(f"{name} must be finite")


def _json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        _finite(value, path)
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PrecisionConfigError(f"{path} keys must be strings")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    raise PrecisionConfigError(f"{path} contains unsupported value {type(value).__name__}")


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not value or value in {".", ".."}:
        raise PrecisionConfigError("model/dataset name is not a safe path component")
    return value[:120]
