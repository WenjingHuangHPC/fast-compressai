"""FactorizedPrior entropy path using the local warp_div backend."""

from __future__ import annotations

import torch

from .errors import CoderInputError
from .warp_div import WarpDivBackend, WarpDivPayload


class FactorizedWarpDivCodec:
    """Keep Balle's transforms/quantizer while replacing its entropy coder."""

    def __init__(self, model, p_ch: int) -> None:
        self.model = model
        self.p_ch = p_ch
        self.backend = WarpDivBackend()
        entropy = model.entropy_bottleneck
        if not 1 <= p_ch <= entropy.channels:
            raise CoderInputError(f"P must be between 1 and {entropy.channels} for this model")
        self._tables(entropy)

    @staticmethod
    def _tables(entropy):
        cdfs = entropy._quantized_cdf
        lengths = entropy._cdf_length
        offsets = entropy._offset
        if (cdfs.ndim != 2 or cdfs.numel() == 0 or lengths.numel() != cdfs.size(0)
                or offsets.numel() != cdfs.size(0)):
            raise CoderInputError("entropy tables are not initialized; call model.update() before accelerate()")
        return cdfs.contiguous(), lengths.contiguous(), offsets.contiguous()

    def compress(self, x: torch.Tensor, timing: dict[str, float] | None = None):
        if not isinstance(x, torch.Tensor) or x.device.type != "cuda" or x.ndim != 4:
            raise CoderInputError("warp_div FactorizedPrior.compress expects a CUDA NCHW tensor")
        with torch.no_grad():
            start = torch.cuda.Event(enable_timing=True) if timing is not None else None
            end = torch.cuda.Event(enable_timing=True) if timing is not None else None
            if timing is not None:
                start.record()
            y = self.model.g_a(x)
            entropy = self.model.entropy_bottleneck
            if y.size(1) != entropy.channels:
                raise CoderInputError("latent channel count does not match entropy bottleneck")
            cdfs, lengths, offsets = self._tables(entropy)
            indexes = entropy._build_indexes(y.size(), device=y.device)
            medians = entropy._extend_ndims(entropy._get_medians().detach(), y.ndim - 2)
            medians = medians.expand(y.size(0), *([-1] * (y.ndim - 1)))
            symbols = entropy.quantize(y, "symbols", medians)
            if timing is not None:
                end.record()
                end.synchronize()
                timing["compress_inference_ms"] = float(start.elapsed_time(end))
                start.record()
            payload = self.backend.encode(
                symbols.reshape(y.size(0), -1).to(torch.int32).contiguous(),
                indexes.reshape(y.size(0), -1).to(torch.int32).contiguous(),
                cdfs, lengths, offsets, p_ch=self.p_ch,
            )
            if timing is not None:
                end.record()
                end.synchronize()
                timing["compress_codec_ms"] = float(start.elapsed_time(end))
            return {"strings": [payload], "shape": tuple(y.shape[-2:])}

    def decompress(self, strings, shape, timing: dict[str, float] | None = None):
        if (not isinstance(strings, (list, tuple)) or len(strings) != 1
                or not isinstance(strings[0], WarpDivPayload)):
            raise CoderInputError("warp_div decompress expects strings=[WarpDivPayload]")
        if (not isinstance(shape, (tuple, list, torch.Size)) or len(shape) != 2
                or any(type(dimension) is not int or dimension <= 0 for dimension in shape)):
            raise CoderInputError("warp_div decompress expects a positive (height, width) shape")
        payload = strings[0]
        if payload.p_ch != self.p_ch:
            raise CoderInputError("payload P_ch does not match the configured coder")
        entropy = self.model.entropy_bottleneck
        channels = entropy.channels
        if payload.symbols_per_batch != channels * shape[0] * shape[1]:
            raise CoderInputError("payload symbol count does not match the latent shape")
        cdfs, lengths, offsets = self._tables(entropy)
        if payload.packed.device != cdfs.device:
            raise CoderInputError("payload and entropy model must be on the same CUDA device")
        with torch.no_grad():
            output_shape = (payload.batch, channels, *shape)
            indexes = entropy._build_indexes(output_shape, device=cdfs.device)
            start = torch.cuda.Event(enable_timing=True) if timing is not None else None
            end = torch.cuda.Event(enable_timing=True) if timing is not None else None
            if timing is not None:
                start.record()
            symbols = self.backend.decode(
                payload, indexes.reshape(payload.batch, -1).contiguous(),
                cdfs, lengths, offsets,
            ).reshape(output_shape)
            if timing is not None:
                end.record()
                end.synchronize()
                timing["decompress_codec_ms"] = float(start.elapsed_time(end))
                start.record()
            medians = entropy._extend_ndims(entropy._get_medians().detach(), len(shape))
            medians = medians.expand(payload.batch, *([-1] * (len(shape) + 1)))
            y_hat = entropy.dequantize(symbols, medians, medians.dtype)
            result = {"x_hat": self.model.g_s(y_hat).clamp_(0, 1)}
            if timing is not None:
                end.record()
                end.synchronize()
                timing["decompress_inference_ms"] = float(start.elapsed_time(end))
            return result
