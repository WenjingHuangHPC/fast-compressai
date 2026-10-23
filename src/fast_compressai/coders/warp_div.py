"""Optional WarpANS V2 fast-division GPU backend.

The backend returns device-local packed tensors. They are not serialized as the
CPU ``Bitstream`` format and are only valid with matching CUDA coder metadata.
"""

from __future__ import annotations

import hashlib
import importlib
from dataclasses import dataclass
from typing import Sequence

import torch

from .capabilities import warp_div_capability
from .errors import CoderDependencyError, CoderInputError, CoderStateError


@dataclass(frozen=True)
class WarpDivPayload:
    packed: torch.Tensor
    max_rounds: torch.Tensor
    header_bytes: int
    chunk_len: int
    p_ch: int
    batch: int
    symbols_per_batch: int
    cdf_count: int
    coding_fingerprint: str | None = None
    format_id: str = "warpans-v2-warp-div-v1-decode"
    format_version: int = 1

    def __post_init__(self) -> None:
        if self.format_id != "warpans-v2-warp-div-v1-decode" or self.format_version != 1:
            raise CoderInputError("unsupported warp_div payload format")
        if self.packed.device.type != "cuda" or self.packed.dtype != torch.uint8 or not self.packed.is_contiguous():
            raise CoderInputError("packed payload must be contiguous CUDA uint8")
        if (self.max_rounds.device != self.packed.device or self.max_rounds.dtype != torch.uint32
                or not self.max_rounds.is_contiguous()):
            raise CoderInputError("max_rounds must be contiguous CUDA uint32 on the packed payload device")
        if self.coding_fingerprint is not None and (
            not isinstance(self.coding_fingerprint, str) or len(self.coding_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.coding_fingerprint)
        ):
            raise CoderInputError("coding_fingerprint must be a SHA-256 hex digest")
        if min(self.header_bytes, self.chunk_len, self.p_ch, self.batch, self.symbols_per_batch, self.cdf_count) <= 0:
            raise CoderInputError("warp_div metadata values must be positive")
        if self.p_ch > self.cdf_count or self.symbols_per_batch % self.cdf_count:
            raise CoderInputError("warp_div payload P_ch/channel metadata is inconsistent")
        expected_chunks = (self.cdf_count + self.p_ch - 1) // self.p_ch
        if tuple(self.max_rounds.shape) != (self.batch, expected_chunks):
            raise CoderInputError("warp_div max_rounds shape does not match payload metadata")
        if self.chunk_len != self.p_ch * (self.symbols_per_batch // self.cdf_count):
            raise CoderInputError("warp_div chunk_len does not match P_ch and symbol dimensions")
        if self.header_bytes > self.packed.numel():
            raise CoderInputError("warp_div header exceeds packed payload size")
        if self.header_bytes % 16 or self.header_bytes < 32 + self.max_rounds.numel() * 4:
            raise CoderInputError("warp_div header size does not match chunk metadata")
        if self.packed.numel() % 4:
            raise CoderInputError("warp_div packed payload must be uint32 aligned")
        rounds = self.max_rounds.to(torch.int64)
        if bool((rounds < 2).any().item()):
            raise CoderInputError("warp_div chunk rounds must include two initial state words")
        expected_size = self.header_bytes + int(rounds.sum().item()) * 32 * 4
        if expected_size != self.packed.numel():
            raise CoderInputError("warp_div packed payload length does not match chunk rounds")


class WarpDivBackend:
    """WarpANS V2 fast-division encoder with V1-compatible decode."""

    name = "warp_div"
    format_id = "warpans-v2-warp-div-v1-decode"
    format_version = 1

    def __init__(self) -> None:
        capability = warp_div_capability()
        if not capability.available:
            raise CoderDependencyError(f"warp_div unavailable: {capability.reason}")
        try:
            self._extension = importlib.import_module("fast_compressai.coders._warp_div")
        except (ImportError, OSError) as exc:
            raise CoderDependencyError(f"warp_div extension could not be loaded: {exc}") from exc

    def encode(
        self,
        symbols: torch.Tensor,
        indexes: torch.Tensor,
        cdfs: torch.Tensor,
        cdf_sizes: torch.Tensor,
        offsets: torch.Tensor,
        *,
        p_ch: int,
    ) -> WarpDivPayload:
        _validate_inputs(symbols, indexes, cdfs, cdf_sizes, offsets, p_ch)
        try:
            packed, max_rounds, header_bytes, chunk_len, p_value = self._extension.encode_warp_div(
                symbols, indexes, cdfs, cdf_sizes, offsets, int(p_ch)
            )
        except Exception as exc:
            raise CoderInputError(f"warp_div encode failed: {exc}") from exc
        return WarpDivPayload(
            packed=packed,
            max_rounds=max_rounds,
            header_bytes=int(header_bytes.item()),
            chunk_len=int(chunk_len.item()),
            p_ch=int(p_value.item()),
            batch=int(symbols.shape[0]),
            symbols_per_batch=int(symbols.shape[1]),
            cdf_count=int(cdfs.shape[0]),
            coding_fingerprint=_coding_fingerprint(indexes, cdfs, cdf_sizes, offsets),
        )

    def decode(
        self,
        payload: WarpDivPayload,
        indexes: torch.Tensor,
        cdfs: torch.Tensor,
        cdf_sizes: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(payload, WarpDivPayload):
            raise CoderInputError("warp_div decode expects WarpDivPayload")
        _validate_inputs(indexes, indexes, cdfs, cdf_sizes, offsets, payload.p_ch)
        if tuple(indexes.shape) != (payload.batch, payload.symbols_per_batch):
            raise CoderInputError("indexes shape does not match payload metadata")
        if cdfs.shape[0] != payload.cdf_count or indexes.device != payload.packed.device:
            raise CoderInputError("CDF count or CUDA device does not match payload metadata")
        if payload.coding_fingerprint is not None and payload.coding_fingerprint != _coding_fingerprint(indexes, cdfs, cdf_sizes, offsets):
            raise CoderInputError("coding tables or indexes do not match warp_div payload")
        _validate_payload_header(payload)
        header = torch.tensor([payload.header_bytes], dtype=torch.int64, device="cpu")
        chunk = torch.tensor([payload.chunk_len], dtype=torch.int32, device="cpu")
        p_value = torch.tensor([payload.p_ch], dtype=torch.int32, device="cpu")
        try:
            return self._extension.decode_warp_div(
                payload.packed,
                payload.max_rounds,
                header,
                chunk,
                p_value,
                indexes,
                cdfs,
                cdf_sizes,
                offsets,
            )
        except Exception as exc:
            raise CoderInputError(f"warp_div decode failed: {exc}") from exc


def _coding_fingerprint(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        cpu = tensor.detach().contiguous().cpu()
        digest.update(bytes(str(tuple(cpu.shape)), "ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _validate_payload_header(payload: WarpDivPayload) -> None:
    # V2 writes eight uint32 header words followed by B*K round counts.
    words = payload.packed[:payload.header_bytes].cpu().view(torch.int32)
    expected_chunks = payload.max_rounds.shape[1]
    batch = payload.batch
    if (
        int(words[0]) != payload.symbols_per_batch
        or int(words[1]) != payload.chunk_len
        or int(words[2]) != ((batch << 16) | expected_chunks)
        or int(words[3]) != ((32 << 16) | 2)
        or int(words[4]) != 0
        or int(words[5]) != 0
    ):
        raise CoderInputError("warp_div packed header is inconsistent with payload metadata")
    if not torch.equal(words[8:8 + payload.max_rounds.numel()], payload.max_rounds.reshape(-1).cpu().to(torch.int32)):
        raise CoderInputError("warp_div packed header round counts do not match payload metadata")


def _validate_inputs(
    symbols: torch.Tensor,
    indexes: torch.Tensor,
    cdfs: torch.Tensor,
    cdf_sizes: torch.Tensor,
    offsets: torch.Tensor,
    p_ch: int,
) -> None:
    if not isinstance(symbols, torch.Tensor) or not isinstance(indexes, torch.Tensor):
        raise CoderInputError("symbols and indexes must be torch tensors")
    tensors: Sequence[torch.Tensor] = (symbols, indexes, cdfs, cdf_sizes, offsets)
    if any(not isinstance(value, torch.Tensor) for value in tensors):
        raise CoderInputError("indexes, CDFs, CDF sizes, and offsets must be tensors")
    if type(p_ch) is not int or p_ch <= 0:
        raise CoderInputError("p_ch must be a positive integer")
    if symbols.device.type != "cuda" or any(value.device != symbols.device for value in tensors):
        raise CoderInputError("all coder inputs must be CUDA tensors on the same device")
    if any(value.dtype != torch.int32 for value in tensors):
        raise CoderInputError("all warp_div input tensors must use torch.int32")
    if any(not value.is_contiguous() for value in tensors):
        raise CoderInputError("all warp_div input tensors must be contiguous")
    if symbols.ndim != 2 or indexes.shape != symbols.shape:
        raise CoderInputError("symbols and indexes must have matching [B, N] shape")
    if cdfs.ndim != 2 or cdf_sizes.ndim != 1 or offsets.ndim != 1:
        raise CoderInputError("CDF must be [M,L] and CDF sizes/offsets must be [M]")
    if cdfs.shape[0] <= 0 or cdfs.shape[1] < 2:
        raise CoderInputError("CDF table cannot be empty")
    if cdf_sizes.numel() != cdfs.shape[0] or offsets.numel() != cdfs.shape[0]:
        raise CoderInputError("CDF, CDF sizes, and offsets must have the same table count")
    if symbols.shape[0] <= 0 or symbols.shape[1] <= 0:
        raise CoderInputError("empty symbols are not supported by this WarpANS format")
    if symbols.shape[1] % cdfs.shape[0] != 0:
        raise CoderInputError("symbols per batch must be divisible by CDF count for WarpANS channel layout")
    if p_ch > cdfs.shape[0]:
        raise CoderInputError("p_ch cannot exceed the number of CDF channels")
    if indexes.numel() and (int(indexes.min().item()) < 0 or int(indexes.max().item()) >= cdfs.shape[0]):
        raise CoderInputError("indexes contain a value outside the CDF table range")
