"""Compatibility adapter for CompressAI-style models.

The adapter deliberately performs inspection and transparent delegation only.
It does not rewrite graphs, normalize bitstreams, or call ``update`` during
construction.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from torch import nn

from ..errors import UnsupportedOperationError


class ContractFamily(str, Enum):
    """The strongest model contract detected without executing the model."""

    GENERIC = "generic"
    COMPRESSION_MODEL = "compression_model"
    LATENT_CODEC = "latent_codec"
    COMPRESSAI_STYLE = "compressai_style"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AdapterInfo:
    """JSON-friendly static compatibility information."""

    framework: str | None
    framework_version: str | None
    model_class: str
    model_module: str
    contract_family: str
    available_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]
    update_available: bool
    update_called: bool
    entropy_models: tuple[str, ...]
    signature_level: str
    compress_signature: str | None
    decompress_signature: str | None
    validation_level: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompressAIAdapter:
    """Inspect and transparently delegate a PyTorch compression model."""

    def __init__(self, model: nn.Module) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("CompressAIAdapter requires a torch.nn.Module")
        self.model = model
        self._update_called = False
        self._info = self._inspect()

    @property
    def info(self) -> AdapterInfo:
        return self._info

    def _inspect(self) -> AdapterInfo:
        compressai = _safe_import("compressai")
        compression_model_types = _find_types(
            ("compressai.models", "CompressionModel"),
            ("compressai.models.base", "CompressionModel"),
        )
        latent_codec_types = _find_types(
            ("compressai.latent_codecs", "LatentCodec"),
            ("compressai.latent_codecs.base", "LatentCodec"),
        )

        compress = getattr(self.model, "compress", None)
        decompress = getattr(self.model, "decompress", None)
        update = getattr(self.model, "update", None)
        forward = getattr(self.model, "forward", None)
        available = tuple(
            name for name, value in (
                ("forward", forward),
                ("compress", compress),
                ("decompress", decompress),
                ("update", update),
                ("state_dict", getattr(self.model, "state_dict", None)),
                ("load_state_dict", getattr(self.model, "load_state_dict", None)),
            )
            if callable(value)
        )
        missing = tuple(name for name in ("compress", "decompress") if not callable(getattr(self.model, name, None)))

        if compression_model_types and isinstance(self.model, compression_model_types):
            family = ContractFamily.COMPRESSION_MODEL
            validation = "typed"
        elif latent_codec_types and isinstance(self.model, latent_codec_types):
            family = ContractFamily.LATENT_CODEC
            validation = "typed"
        elif callable(compress) and callable(decompress):
            family = ContractFamily.COMPRESSAI_STYLE
            validation = "structural"
        elif callable(forward):
            family = ContractFamily.GENERIC
            validation = "forward_only"
        else:
            family = ContractFamily.UNKNOWN
            validation = "unvalidated"

        compress_signature = _signature(compress)
        decompress_signature = _signature(decompress)
        if compress_signature is not None and decompress_signature is not None:
            signature_level = "exact"
        elif compress_signature is not None or decompress_signature is not None:
            signature_level = "partial"
        elif callable(compress) or callable(decompress):
            signature_level = "opaque"
        else:
            signature_level = "missing"

        return AdapterInfo(
            framework="compressai" if compressai is not None else None,
            framework_version=getattr(compressai, "__version__", None) if compressai is not None else None,
            model_class=type(self.model).__qualname__,
            model_module=type(self.model).__module__,
            contract_family=family.value,
            available_paths=available,
            missing_paths=missing,
            update_available=callable(update),
            update_called=self._update_called,
            entropy_models=_find_entropy_models(self.model),
            signature_level=signature_level,
            compress_signature=compress_signature,
            decompress_signature=decompress_signature,
            validation_level=validation,
        )

    def _refresh_update_state(self) -> None:
        info = self._info
        self._info = AdapterInfo(
            **{**info.to_dict(), "update_called": self._update_called}
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def compress(self, *args: Any, **kwargs: Any) -> Any:
        return _call_path(self.model, "compress", *args, **kwargs)

    def decompress(self, *args: Any, **kwargs: Any) -> Any:
        return _call_path(self.model, "decompress", *args, **kwargs)

    def update(self, *args: Any, **kwargs: Any) -> Any:
        result = _call_path(self.model, "update", *args, **kwargs)
        self._update_called = True
        self._refresh_update_state()
        return result

    def state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return _call_path(self.model, "state_dict", *args, **kwargs)

    def load_state_dict(self, *args: Any, **kwargs: Any) -> Any:
        return _call_path(self.model, "load_state_dict", *args, **kwargs)


def _safe_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError):
        return None


def _find_types(*candidates: tuple[str, str]) -> tuple[type, ...]:
    found: list[type] = []
    for module_name, type_name in candidates:
        module = _safe_import(module_name)
        candidate = getattr(module, type_name, None) if module is not None else None
        if isinstance(candidate, type) and candidate not in found:
            found.append(candidate)
    return tuple(found)


def _signature(method: Any) -> str | None:
    if not callable(method):
        return None
    try:
        return str(inspect.signature(method))
    except (TypeError, ValueError):
        return None


def _find_entropy_models(model: nn.Module) -> tuple[str, ...]:
    paths: list[str] = []
    for name in ("entropy_bottleneck", "gaussian_conditional", "entropy_models"):
        value = getattr(model, name, None)
        if value is not None:
            paths.append(name)
    for name, module in model.named_modules():
        if not name:
            continue
        class_name = type(module).__name__.lower()
        if "entropy" in class_name or "gaussianconditional" in class_name:
            if name not in paths:
                paths.append(name)
    return tuple(paths)


def _call_path(model: nn.Module, name: str, *args: Any, **kwargs: Any) -> Any:
    method = getattr(model, name, None)
    if not callable(method):
        raise UnsupportedOperationError(f"{name} is not available: the wrapped model does not define {name}() or it is not callable")
    return method(*args, **kwargs)
