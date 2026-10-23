"""Versioned, JSON-safe coder bitstream container."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import BitstreamError

MAGIC = "fast-compressai-coder"
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Bitstream:
    """Immutable payload and metadata produced by a coder backend."""

    backend: str
    payload: bytes
    symbol_count: int
    shape: tuple[int, ...] = ()
    metadata: Mapping[str, Any] | None = None
    magic: str = MAGIC
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.magic != MAGIC:
            raise BitstreamError(f"unsupported bitstream magic: {self.magic!r}")
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise BitstreamError(f"unsupported bitstream schema version: {self.schema_version!r}")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise BitstreamError("bitstream backend must be a non-empty string")
        if not isinstance(self.payload, bytes):
            raise BitstreamError("bitstream payload must be bytes")
        if type(self.symbol_count) is not int or self.symbol_count < 0:
            raise BitstreamError("bitstream symbol_count must be a non-negative integer")
        if isinstance(self.shape, (str, bytes)):
            raise BitstreamError("bitstream shape must be an integer sequence")
        try:
            shape = tuple(self.shape)
        except TypeError as exc:
            raise BitstreamError("bitstream shape must be an integer sequence") from exc
        if any(type(value) is not int for value in shape):
            raise BitstreamError("bitstream shape dimensions must be integers")
        if any(value < 0 for value in shape):
            raise BitstreamError("bitstream shape values must be non-negative")
        if shape and _product(shape) != self.symbol_count:
            raise BitstreamError("bitstream shape product must equal symbol_count")
        object.__setattr__(self, "shape", shape)
        normalized_metadata = _normalize_json_value(dict(self.metadata or {}), "metadata")
        object.__setattr__(self, "metadata", normalized_metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "magic": self.magic,
            "schema_version": self.schema_version,
            "backend": self.backend,
            "payload": base64.b64encode(self.payload).decode("ascii"),
            "symbol_count": self.symbol_count,
            "shape": list(self.shape),
            "metadata": _normalize_json_value(dict(self.metadata or {}), "metadata"),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Bitstream":
        if not isinstance(value, Mapping):
            raise BitstreamError("serialized bitstream must be a mapping")
        expected = {"magic", "schema_version", "backend", "payload", "symbol_count", "shape", "metadata"}
        if set(value) != expected:
            raise BitstreamError("serialized bitstream has missing or unknown fields")
        payload = value["payload"]
        if not isinstance(payload, str):
            raise BitstreamError("serialized bitstream payload must be base64 text")
        try:
            decoded = base64.b64decode(payload.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise BitstreamError("serialized bitstream payload is not valid base64") from exc
        shape = value["shape"]
        if not isinstance(shape, (list, tuple)):
            raise BitstreamError("serialized bitstream shape must be a list")
        return cls(
            magic=value["magic"],
            schema_version=value["schema_version"],
            backend=value["backend"],
            payload=decoded,
            symbol_count=value["symbol_count"],
            shape=tuple(shape),
            metadata=value["metadata"],
        )


def _normalize_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if type(value) is int:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise BitstreamError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_normalize_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, tuple):
        return [_normalize_json_value(item, f"{path}[]") for item in value]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise BitstreamError(f"{path} keys must be strings")
            result[key] = _normalize_json_value(item, f"{path}.{key}")
        return result
    raise BitstreamError(f"{path} contains a non-JSON value of type {type(value).__name__}")


def _product(values: tuple[int, ...]) -> int:
    result = 1
    for value in values:
        result *= value
    return result
