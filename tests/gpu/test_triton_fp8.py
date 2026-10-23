import unittest

import torch
from torch import nn

try:
    import triton
except ImportError:
    triton = None

from fast_compressai.backends.triton.fp8_conv2d import FP8TritonConv2d, fp8_triton_conv2d_capability


@unittest.skipUnless(torch.cuda.is_available() and triton is not None, "CUDA and Triton are required")
class TritonFP8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.cuda.get_device_capability() < (9, 0):
            raise unittest.SkipTest("Triton FP8 candidate requires sm90+")

    def test_non_aligned_conv2d(self):
        torch.manual_seed(11)
        source = nn.Conv2d(3, 5, 3, padding=1).cuda().eval()
        candidate = FP8TritonConv2d(3, 5, 3, padding=1, device="cuda").eval()
        candidate.load_state_dict(source.state_dict())
        supported, reason = fp8_triton_conv2d_capability(source, {})
        self.assertTrue(supported, reason)
        x = torch.randn(2, 3, 16, 16, device="cuda")
        with torch.inference_mode():
            reference = source(x)
            output = candidate(x)
        self.assertEqual(output.shape, reference.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(float((output - reference).abs().mean()), 0.2)

    def test_cpu_is_unavailable(self):
        supported, reason = fp8_triton_conv2d_capability(nn.Conv2d(3, 5, 3), {})
        self.assertFalse(supported)
        self.assertIn("CUDA", reason)


if __name__ == "__main__":
    unittest.main()
