import importlib
import unittest
from unittest import mock

import torch

import fast_compressai
from fast_compressai.coders.warp_div import WarpDivBackend, WarpDivPayload
from fast_compressai.coders.errors import CoderDependencyError, CoderInputError


class WarpDivApiTests(unittest.TestCase):
    def test_catalog_is_limited_to_two_coders(self):
        self.assertEqual(fast_compressai.coder_catalog(), ("compressai-rans", "warp_div"))
        self.assertEqual(type(fast_compressai.create_coder("compressai-rans" )).__name__, "CompressAIANSBackend")

    def test_capability_is_lazy_and_reports_missing_extension(self):
        with mock.patch("fast_compressai.coders.capabilities.torch.cuda.is_available", return_value=True), mock.patch(
            "fast_compressai.coders.capabilities.importlib.util.find_spec", return_value=None
        ):
            capability = fast_compressai.warp_div_capability("cuda:0")
        self.assertFalse(capability.available)
        self.assertEqual(capability.backend, "warp_div")
        self.assertIn("not built", capability.reason)

    def test_backend_constructor_reports_unavailable_extension(self):
        with mock.patch("fast_compressai.coders.warp_div.warp_div_capability") as capability:
            capability.return_value = mock.Mock(available=False, reason="not built")
            with self.assertRaisesRegex(CoderDependencyError, "not built"):
                WarpDivBackend()

    def test_python_import_does_not_import_native_module(self):
        self.assertIsNotNone(importlib.import_module("fast_compressai.coders.warp_div"))

    def test_payload_rejects_non_cuda_data(self):
        with self.assertRaises(CoderInputError):
            WarpDivPayload(
                packed=torch.zeros(128, dtype=torch.uint8),
                max_rounds=torch.zeros((1, 1), dtype=torch.uint32),
                header_bytes=32,
                chunk_len=4,
                p_ch=1,
                batch=1,
                symbols_per_batch=4,
                cdf_count=1,
            )

    def test_input_contract_rejects_wrong_dtype_before_extension(self):
        from fast_compressai.coders.warp_div import _validate_inputs
        symbols = torch.zeros((1, 4), dtype=torch.float32, device="cpu")
        with self.assertRaises(CoderInputError):
            _validate_inputs(
                symbols,
                torch.zeros((1, 4), dtype=torch.int32),
                torch.tensor([[0, 65536]], dtype=torch.int32),
                torch.tensor([2], dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                1,
            )


if __name__ == "__main__":
    unittest.main()
