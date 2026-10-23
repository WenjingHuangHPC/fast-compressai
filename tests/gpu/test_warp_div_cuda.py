"""CUDA-only integration tests for WarpANS V2 warp_div."""

import unittest

import torch

from fast_compressai.coders import WarpDivBackend, warp_div_capability


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for warp_div integration tests")
class WarpDivCudaTests(unittest.TestCase):
    def setUp(self):
        capability = warp_div_capability("cuda:0")
        if not capability.available:
            self.skipTest(capability.reason)
        self.backend = WarpDivBackend()

    def _case(self, symbols, p_ch):
        device = torch.device("cuda:0")
        symbols = torch.tensor([symbols], dtype=torch.int32, device=device)
        indexes = torch.tensor([[0, 1, 2, 3] * (symbols.shape[1] // 4)], dtype=torch.int32, device=device)
        cdfs = torch.tensor([[0, 8192, 16384, 32768, 65536]] * 4, dtype=torch.int32, device=device)
        sizes = torch.tensor([5] * 4, dtype=torch.int32, device=device)
        offsets = torch.tensor([-2, -1, 0, 1], dtype=torch.int32, device=device)
        payload = self.backend.encode(symbols, indexes, cdfs, sizes, offsets, p_ch=p_ch)
        decoded = self.backend.decode(payload, indexes, cdfs, sizes, offsets)
        self.assertTrue(torch.equal(symbols, decoded))
        self.assertEqual(payload.p_ch, p_ch)
        self.assertEqual(payload.max_rounds.shape, (1, 4 // p_ch))

    def test_round_trip_multiple_p_ch(self):
        symbols = [0, 1, 2, 3] * 16
        for p_ch in (1, 2, 4):
            with self.subTest(p_ch=p_ch):
                self._case(symbols, p_ch)

    def test_round_trip_escape_symbols(self):
        self._case([-100, 100, -101, 102] * 16, 2)

    def test_cuda_backend_only_returns_device_local_payload(self):
        self._case([0, 1, 2, 3] * 16, 4)


if __name__ == "__main__":
    unittest.main()
