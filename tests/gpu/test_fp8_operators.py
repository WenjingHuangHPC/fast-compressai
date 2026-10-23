"""Experimental H100 FP8 Conv2d candidate tests."""

import unittest

import torch
from torch import nn

import fast_compressai
from fast_compressai.backends.pytorch.fp8_operators import FP8Conv2d, fp8_conv2d_capability


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for FP8 tests")
class FP8OperatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability("cuda:0") < (9, 0):
            raise unittest.SkipTest("FP8 candidate is validated only on sm90+")
        if not hasattr(torch, "_scaled_mm"):
            raise unittest.SkipTest("torch._scaled_mm is unavailable")

    def test_non_aligned_conv2d_fp8_candidate(self):
        torch.manual_seed(3)
        source = nn.Conv2d(3, 5, 3, padding=1).cuda().eval()
        supported, reason = fp8_conv2d_capability(source, {})
        self.assertTrue(supported, reason)
        candidate = fast_compressai.OperatorCatalog if False else FP8Conv2d(
            3, 5, 3, padding=1, device="cuda", dtype=torch.float32
        ).eval()
        candidate.load_state_dict(source.state_dict())
        x = torch.randn(2, 3, 16, 16, device="cuda")
        with torch.inference_mode():
            reference = source(x)
            output = candidate(x)
        self.assertEqual(output.shape, reference.shape)
        self.assertEqual(output.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(float((output - reference).abs().mean()), 0.2)

    def test_fp8_negative_capabilities(self):
        cpu_module = nn.Conv2d(3, 5, 3)
        supported, reason = fp8_conv2d_capability(cpu_module, {})
        self.assertFalse(supported)
        self.assertIn("CUDA", reason)
        grouped = nn.Conv2d(4, 4, 3, groups=2).cuda().eval()
        supported, reason = fp8_conv2d_capability(grouped, {})
        self.assertFalse(supported)
        self.assertIn("groups", reason)


if __name__ == "__main__":
    unittest.main()
