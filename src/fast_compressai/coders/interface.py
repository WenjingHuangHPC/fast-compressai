"""Stable coder backend protocols and shared validation helpers."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .bitstream import Bitstream
from .errors import CoderInputError

CDF_PRECISION = 16
CDF_TOTAL = 1 << CDF_PRECISION


class EncoderSession(Protocol):
    """One-shot whole-stream encoder session."""

    def encode(
        self,
        symbols: Sequence[int],
        indexes: Sequence[int],
        cdfs: Sequence[Sequence[int]],
        offsets: Sequence[int],
    ) -> None: ...

    def finalize(self) -> Bitstream: ...


class CoderBackend(Protocol):
    name: str
    schema_version: int

    def create_session(
        self,
        *,
        symbol_count: int,
        shape: Sequence[int] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> EncoderSession: ...

    def decode(
        self,
        bitstream: Bitstream,
        *,
        indexes: Sequence[int],
        cdfs: Sequence[Sequence[int]],
        offsets: Sequence[int],
    ) -> list[int]: ...


def normalize_coding_metadata(
    indexes: Sequence[int],
    cdfs: Sequence[Sequence[int]],
    offsets: Sequence[int],
) -> tuple[list[int], list[list[int]], list[int]]:
    try:
        indexes = list(indexes)
        cdfs = [list(cdf) for cdf in cdfs]
        offsets = list(offsets)
    except (TypeError, ValueError) as exc:
        raise CoderInputError("indexes, cdfs, and offsets must be integer sequences") from exc
    _require_ints(indexes, "indexes")
    _require_ints(offsets, "offsets")
    for row, cdf in enumerate(cdfs):
        _require_ints(cdf, f"CDF {row}")
        if len(cdf) < 2:
            raise CoderInputError(f"CDF {row} must contain at least two values")
        if cdf[0] != 0 or cdf[-1] != CDF_TOTAL:
            raise CoderInputError(
                f"CDF {row} must start at zero and end at {CDF_TOTAL} (precision {CDF_PRECISION})"
            )
        if any(left >= right for left, right in zip(cdf, cdf[1:])):
            raise CoderInputError(f"CDF {row} must be strictly increasing")
    if not cdfs and indexes:
        raise CoderInputError("cdfs must contain at least one CDF")
    if len(offsets) != len(cdfs):
        raise CoderInputError("offsets length must match the number of CDFs")
    for index in indexes:
        if index < 0 or index >= len(cdfs):
            raise CoderInputError(f"CDF index {index} is out of range")
    return indexes, cdfs, offsets


def validate_coding_inputs(
    symbols: Sequence[int],
    indexes: Sequence[int],
    cdfs: Sequence[Sequence[int]],
    offsets: Sequence[int],
) -> tuple[list[int], list[int], list[list[int]], list[int]]:
    try:
        symbols = list(symbols)
    except (TypeError, ValueError) as exc:
        raise CoderInputError("symbols must be an integer sequence") from exc
    _require_ints(symbols, "symbols")
    indexes, cdfs, offsets = normalize_coding_metadata(indexes, cdfs, offsets)
    if len(symbols) != len(indexes):
        raise CoderInputError("symbols and indexes must have equal lengths")
    # CompressAI RANS supports out-of-range values through escape/bypass coding.
    return symbols, indexes, cdfs, offsets


def _require_ints(values: Sequence[Any], name: str) -> None:
    if any(type(value) is not int for value in values):
        raise CoderInputError(f"{name} must contain integers (bool is not accepted)")
