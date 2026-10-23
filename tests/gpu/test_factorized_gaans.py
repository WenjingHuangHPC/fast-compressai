"""End-to-end Balle FactorizedPrior tests for the local GA-ANS coder."""

import unittest
from dataclasses import replace

import torch

try:
    from compressai.models import FactorizedPrior
except ImportError:
    FactorizedPrior = None

from fast_compressai import accelerate
from fast_compressai.coders import WarpDivPayload, warp_div_capability
from fast_compressai.coders.errors import CoderInputError
from fast_compressai.config import ConfigurationError


@unittest.skipUnless(torch.cuda.is_available() and FactorizedPrior is not None,
                     "FactorizedPrior and CUDA are required")
class FactorizedGAANSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        capability = warp_div_capability("cuda:0")
        if not capability.available:
            raise unittest.SkipTest(capability.reason)

    def setUp(self):
        torch.manual_seed(20261015)
        self.model = FactorizedPrior(8, 12).cuda().eval()
        self.model.update(force=True)

    def test_gaans_matches_rans_for_multiple_p_and_batch_sizes(self):
        for batch in (1, 2):
            x = torch.rand(batch, 3, 64, 64, device="cuda:0")
            with torch.inference_mode():
                original = self.model.compress(x)
                reference = self.model.decompress(original["strings"], original["shape"])["x_hat"]
                reference_symbols = self.model.entropy_bottleneck.decompress(
                    original["strings"][0], original["shape"]
                )
                y = self.model.g_a(x)
                eb = self.model.entropy_bottleneck
                medians = eb._extend_ndims(eb._get_medians().detach(), 2).expand(batch, -1, -1, -1)
                expected_symbols = eb.quantize(y, "symbols", medians)
            for p_ch in (1, 4, 5, 12):
                with self.subTest(batch=batch, p_ch=p_ch):
                    source = {key: value.clone() for key, value in self.model.state_dict().items()}
                    fast = accelerate(self.model, P=p_ch, device="cuda:0")
                    self.assertEqual(fast.acceleration_report.coder_status, "warp_div")
                    self.assertTrue(fast.acceleration_report.p_ch_applied)
                    self.assertEqual(fast.acceleration_report.runtime_status, "not_applied")
                    self.assertEqual(next(fast.parameters()).device.type, "cuda")
                    with torch.inference_mode():
                        stream = fast.compress(x)
                        payload = stream["strings"][0]
                        self.assertIsInstance(payload, WarpDivPayload)
                        self.assertEqual(payload.p_ch, p_ch)
                        reconstructed = fast.decompress(stream["strings"], stream["shape"])["x_hat"]
                        fast_eb = fast.model.entropy_bottleneck
                        indexes = fast_eb._build_indexes((batch, 12, *stream["shape"]), device="cuda:0")
                        symbols = fast._coder.backend.decode(
                            payload, indexes.reshape(batch, -1).contiguous(),
                            fast_eb._quantized_cdf.contiguous(), fast_eb._cdf_length.contiguous(),
                            fast_eb._offset.contiguous(),
                        ).reshape_as(expected_symbols)
                    self.assertTrue(torch.equal(expected_symbols, symbols))
                    self.assertTrue(torch.equal(reference_symbols, self.model.entropy_bottleneck.dequantize(symbols, self.model.entropy_bottleneck._get_medians(), self.model.entropy_bottleneck._get_medians().dtype)))
                    self.assertTrue(torch.allclose(reference, reconstructed, atol=1e-6, rtol=1e-6))
                    self.assertEqual(set(source), set(fast.state_dict()))
                    self.assertTrue(all(torch.equal(value, self.model.state_dict()[key]) for key, value in source.items()))
                    self.assertEqual(next(self.model.parameters()).device.type, "cuda")

    def test_invalid_model_state_and_metadata_are_rejected(self):
        fresh = FactorizedPrior(8, 12).cuda().eval()
        with self.assertRaisesRegex(CoderInputError, "update"):
            accelerate(fresh, P=4, device="cuda:0")
        with self.assertRaisesRegex(CoderInputError, "P"):
            accelerate(self.model, P=13, device="cuda:0")
        with self.assertRaises(ConfigurationError):
            accelerate(self.model, P=4, device="cpu")
        fast = accelerate(self.model, P=4, device="cuda:0")
        with torch.inference_mode():
            stream = fast.compress(torch.rand(1, 3, 64, 64, device="cuda:0"))
        payload = stream["strings"][0]
        with self.assertRaises(CoderInputError):
            fast.decompress(stream["strings"], (8, 8))
        with self.assertRaises(CoderInputError):
            fast.decompress([[b"cpu-rans"]], stream["shape"])
        with self.assertRaisesRegex(CoderInputError, "match"):
            fast.decompress([replace(payload, p_ch=3, chunk_len=12)], stream["shape"])
        with self.assertRaisesRegex(CoderInputError, "match"):
            fast.decompress([replace(payload, coding_fingerprint="0" * 64)], stream["shape"])
        with self.assertRaisesRegex(CoderInputError, "header"):
            corrupted = payload.packed.clone()
            corrupted[0] ^= 1
            fast.decompress([replace(payload, packed=corrupted)], stream["shape"])


if __name__ == "__main__":
    unittest.main()
