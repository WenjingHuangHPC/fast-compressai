import unittest

import torch
from torch import nn

import fast_compressai


class CompressAIStyleMock(nn.Module):
    def forward(self, x):
        return {"x_hat": x, "likelihoods": {"y": torch.ones_like(x)}}

    def compress(self, x):
        return {"strings": [[b"mock"]], "shape": tuple(x.shape)}

    def decompress(self, strings, shape):
        return {"x_hat": torch.zeros(shape), "strings": strings}


class AccelerateApiTests(unittest.TestCase):
    def test_p_maps_to_p_ch_and_budget_is_independent(self):
        with self.assertRaises(fast_compressai.ConfigurationError):
            fast_compressai.accelerate(CompressAIStyleMock(), P=8, budget=0.1, device="cpu")
        accelerated = fast_compressai.accelerate(
            CompressAIStyleMock(), P=8, budget=0.1, device="cpu", coder="rans"
        )

        self.assertIsInstance(accelerated, fast_compressai.FastModel)
        self.assertEqual(accelerated.acceleration_config.p_ch, 8)
        self.assertEqual(accelerated.acceleration_config.budget, 0.1)
        self.assertEqual(accelerated.acceleration_report.p_ch, 8)
        self.assertEqual(accelerated.acceleration_report.quality_budget, 0.1)
        self.assertEqual(accelerated.acceleration_report.status, "reference_only")
        self.assertEqual(accelerated.acceleration_report.runtime_status, "not_applied")
        self.assertIsNone(accelerated.acceleration_report.speedup)
        self.assertFalse(accelerated.acceleration_report.accelerated)

    def test_wrapper_preserves_model_paths(self):
        accelerated = fast_compressai.accelerate(CompressAIStyleMock(), P=4, budget=0.2, coder="rans")
        value = torch.ones(1, 2)
        self.assertIn("x_hat", accelerated(value))
        strings = accelerated.compress(value)
        self.assertEqual(strings["shape"], (1, 2))
        self.assertEqual(accelerated.decompress(strings, (1, 2))["x_hat"].shape, (1, 2))
        self.assertEqual(set(accelerated.state_dict()), set())

    def test_config_alias_and_validation(self):
        config = fast_compressai.AccelerationConfig.from_value({"P": 16, "budget": 0.05})
        self.assertEqual(config.p_ch, 16)
        self.assertEqual(config.budget, 0.05)
        self.assertEqual(config.to_dict()["p_ch"], 16)
        with self.assertRaises(fast_compressai.ConfigurationError):
            fast_compressai.AccelerationConfig.from_value({"P": 4, "p_ch": 4})
        for value in (True, 0, -1, 1.5):
            with self.subTest(P=value), self.assertRaises(fast_compressai.ConfigurationError):
                fast_compressai.AccelerationConfig(p_ch=value)
        with self.assertRaises(fast_compressai.ConfigurationError):
            fast_compressai.AccelerationConfig(budget=-0.1)
        with self.assertRaises(fast_compressai.ConfigurationError):
            fast_compressai.AccelerationConfig(budget=float("inf"))

    def test_forward_profile_is_measured_without_sample_path_changes(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 1), nn.ReLU()).eval()
        sample = torch.rand(1, 3, 8, 8)
        accelerated = fast_compressai.accelerate(model, budget=100.0, sample=sample, coder="rans")
        report = accelerated.acceleration_report
        self.assertIn(report.profiling_status, {"measured", "failed", "unavailable"})
        self.assertEqual(report.quality_metric, "psnr_drop_db")
        self.assertEqual(report.coder_status, "rans")
        self.assertFalse(report.p_ch_applied)
        self.assertIsNone(report.p_ch)

    def test_default_coder_is_warp_div_and_requires_supported_cuda_model(self):
        with self.assertRaisesRegex(fast_compressai.ConfigurationError, "warp_div currently supports only"):
            fast_compressai.accelerate(CompressAIStyleMock(), P=4, device="cuda")
        with self.assertRaisesRegex(fast_compressai.ConfigurationError, "device='cuda'"):
            from compressai.models import FactorizedPrior
            model = FactorizedPrior(4, 8)
            model.update(force=True)
            fast_compressai.accelerate(model, P=4, device="cpu")

    def test_compile_remains_compatible(self):
        model = nn.Identity()
        self.assertIsInstance(fast_compressai.compile(model), fast_compressai.FastModel)


if __name__ == "__main__":
    unittest.main()
