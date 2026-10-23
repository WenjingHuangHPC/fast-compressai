"""CompressAI RANS-backed reference coder."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from .bitstream import Bitstream
from .errors import BitstreamError, CoderDependencyError, CoderError, CoderInputError, CoderStateError
from .interface import normalize_coding_metadata, validate_coding_inputs


class CompressAIANSBackend:
    """CPU reference backend backed by CompressAI's fixed-precision RANS extension."""

    name = "compressai-rans"
    schema_version = 1

    def __init__(self) -> None:
        try:
            from compressai import ans
        except (ImportError, ModuleNotFoundError) as exc:
            raise CoderDependencyError(
                "CompressAI RANS is unavailable; install fast-compressai[compressai]"
            ) from exc
        encoder = getattr(ans, "RansEncoder", None)
        decoder = getattr(ans, "RansDecoder", None)
        if encoder is None or decoder is None:
            raise CoderDependencyError("the installed CompressAI package has no RANS encoder/decoder")
        self._encoder_type = encoder
        self._decoder_type = decoder

    def create_session(
        self,
        *,
        symbol_count: int,
        shape: Sequence[int] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "CompressAIANSEncoderSession":
        if type(symbol_count) is not int or symbol_count < 0:
            raise CoderInputError("symbol_count must be a non-negative integer")
        return CompressAIANSEncoderSession(
            self._encoder_type,
            self.name,
            symbol_count,
            shape,
            metadata,
        )

    def decode(
        self,
        bitstream: Bitstream,
        *,
        indexes: Sequence[int],
        cdfs: Sequence[Sequence[int]],
        offsets: Sequence[int],
    ) -> list[int]:
        if not isinstance(bitstream, Bitstream):
            raise CoderInputError("decode expects a Bitstream")
        if bitstream.backend != self.name:
            raise CoderInputError(
                f"bitstream backend {bitstream.backend!r} does not match {self.name!r}"
            )
        indexes, cdfs, normalized_offsets = normalize_coding_metadata(indexes, cdfs, offsets)
        if len(indexes) != bitstream.symbol_count:
            raise CoderInputError("indexes length must equal bitstream symbol_count")
        expected_fingerprint = (bitstream.metadata or {}).get("coding_fingerprint")
        if expected_fingerprint is not None:
            actual = _coding_fingerprint(indexes, cdfs, normalized_offsets)
            if expected_fingerprint != actual:
                raise BitstreamError("coding metadata does not match bitstream fingerprint")
        if bitstream.symbol_count == 0:
            return []
        try:
            decoder = self._decoder_type()
            return list(
                decoder.decode_with_indexes(
                    bitstream.payload,
                    indexes,
                    cdfs,
                    [len(cdf) for cdf in cdfs],
                    normalized_offsets,
                )
            )
        except Exception as exc:
            raise CoderError("CompressAI RANS decode failed") from exc


class CompressAIANSEncoderSession:
    """One-shot encoder: one full symbol sequence per session."""

    def __init__(self, encoder_type: type, backend: str, symbol_count: int, shape: Sequence[int], metadata: Mapping[str, Any] | None) -> None:
        self._encoder_type = encoder_type
        self._backend = backend
        self._symbol_count = symbol_count
        self._shape = tuple(shape)
        self._metadata = dict(metadata or {})
        self._payload: bytes | None = None
        self._failed = False
        self._finalized = False
        self._encode_attempted = False

    def encode(
        self,
        symbols: Sequence[int],
        indexes: Sequence[int],
        cdfs: Sequence[Sequence[int]],
        offsets: Sequence[int],
    ) -> None:
        self._ensure_open()
        if self._encode_attempted:
            raise CoderStateError("one-shot coder session accepts only one encode() call")
        self._encode_attempted = True
        try:
            symbols, indexes, cdfs, offsets = validate_coding_inputs(
                symbols, indexes, cdfs, offsets
            )
            if len(symbols) != self._symbol_count:
                raise CoderInputError("encoded symbol count does not match session symbol_count")
            self._metadata["coding_fingerprint"] = _coding_fingerprint(indexes, cdfs, offsets)
            if not symbols:
                self._payload = b""
                return
            encoder = self._encoder_type()
            self._payload = bytes(
                encoder.encode_with_indexes(
                    symbols,
                    indexes,
                    cdfs,
                    [len(cdf) for cdf in cdfs],
                    offsets,
                )
            )
        except Exception:
            self._failed = True
            raise

    def finalize(self) -> Bitstream:
        self._ensure_open()
        if not self._encode_attempted:
            if self._symbol_count != 0:
                raise CoderStateError("cannot finalize a non-empty session before encoding its symbols")
            self._encode_attempted = True
            self._payload = b""
        if self._payload is None:
            raise CoderStateError("coder session has no finalized payload")
        self._finalized = True
        return Bitstream(
            backend=self._backend,
            payload=self._payload,
            symbol_count=self._symbol_count,
            shape=self._shape,
            metadata=self._metadata,
        )

    def _ensure_open(self) -> None:
        if self._failed:
            raise CoderStateError("coder session failed and cannot be reused")
        if self._finalized:
            raise CoderStateError("coder session is already finalized")


def _coding_fingerprint(
    indexes: Sequence[int],
    cdfs: Sequence[Sequence[int]],
    offsets: Sequence[int],
) -> str:
    canonical = json.dumps(
        {"indexes": list(indexes), "cdfs": [list(cdf) for cdf in cdfs], "offsets": list(offsets), "precision": 16},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()
