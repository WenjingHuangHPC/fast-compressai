"""GPU candidate tests for Balle's operator-level BF16 catalog."""

import unittest

import torch

try:
    from compressai.models import FactorizedPrior
except ImportError:
    FactorizedPrior = None

import fast_compressai


@unittest.skipUnless(torch.cuda.is_available() and FactorizedPrior is not None,
                     "CUDA and CompressAI are required for Balle operator tests")
class FactorizedOperatorCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("the selected CUDA device does not support BF16")

    def setUp(self):
        torch.manual_seed(71)
        self.catalog = fast_compressai.build_balle_catalog()
        self.model = FactorizedPrior(N=8, M=12).cuda().eval()
        self.model.update(force=True)

    def test_factorized_prior_inventory_covers_transform_and_locked_entropy(self):
        inventory = self.catalog.inventory(self.model)
        counts = {}
        for item in inventory:
            counts[item.logical_op] = counts.get(item.logical_op, 0) + 1
        self.assertEqual(counts["conv2d"], 4)
        self.assertEqual(counts["conv_transpose2d"], 4)
        self.assertEqual(counts["gdn"], 3)
        self.assertEqual(counts["igdn"], 3)
        self.assertEqual(counts["entropy_bottleneck"], 1)
        self.assertTrue(self.catalog.get("entropy_bottleneck").precision_locked)

    def test_each_balle_bf16_candidate_executes_on_cuda_and_preserves_fp32_state(self):
        samples = {
            "conv2d": torch.randn(2, 3, 32, 32, device="cuda"),
            "conv_transpose2d": torch.randn(2, self.model.g_s[0].in_channels, 16, 16, device="cuda"),
        }
        for logical_op, module, sample in (
            ("conv2d", self.model.g_a[0], samples["conv2d"]),
            ("conv_transpose2d", self.model.g_s[0], samples["conv_transpose2d"]),
            ("gdn", self.model.g_a[1], torch.randn(2, 8, 32, 32, device="cuda")),
            ("igdn", self.model.g_s[1], torch.randn(2, 8, 16, 16, device="cuda")),
        ):
            with self.subTest(logical_op=logical_op):
                original_state = {key: value.clone() for key, value in module.state_dict().items()}
                candidate = self.catalog.build(module, logical_op, "bf16", context={"training": False})
                self.assertFalse(candidate.training)
                for parameter in candidate.parameters():
                    self.assertEqual(parameter.dtype, torch.float32)
                    self.assertEqual(parameter.device.type, "cuda")
                with torch.inference_mode():
                    reference_output = module(sample)
                    candidate_output = candidate(sample)
                self.assertEqual(candidate_output.shape, reference_output.shape)
                self.assertEqual(candidate_output.dtype, reference_output.dtype)
                self.assertEqual(candidate_output.device.type, "cuda")
                self.assertTrue(torch.isfinite(candidate_output).all())
                self.assertTrue(torch.allclose(candidate_output, reference_output, atol=0.2, rtol=0.2))
                self.assertTrue(all(torch.equal(value, module.state_dict()[key]) for key, value in original_state.items()))

    def test_per_instance_mixed_candidates_run_in_factorized_prior_copy(self):
        model = self.model
        inventory = self.catalog.inventory(model)
        by_path = {instance.module_path: instance for instance in inventory}
        assignments = {
            by_path["g_a.0"].instance_id: "bf16",
            by_path["g_a.1"].instance_id: "bf16",
            by_path["g_s.0"].instance_id: "fp32",
        }
        source_state = {key: value.clone() for key, value in model.state_dict().items()}
        converted, applications = self.catalog.apply(model, assignments)

        self.assertEqual({record.module_path for record in applications}, {"g_a.0", "g_a.1", "g_s.0"})
        self.assertIsInstance(converted.g_a[0], fast_compressai.BF16Conv2d)
        self.assertIsInstance(converted.g_a[1], fast_compressai.BF16GDN)
        self.assertNotIsInstance(converted.g_s[0], fast_compressai.BF16ConvTranspose2d)
        self.assertEqual(converted.entropy_bottleneck._quantized_cdf.dtype, torch.int32)
        self.assertTrue(torch.equal(converted.entropy_bottleneck._quantized_cdf,
                                    model.entropy_bottleneck._quantized_cdf))
        with torch.inference_mode():
            x = torch.rand(1, 3, 64, 64, device="cuda")
            output = converted(x)
        self.assertEqual(output["x_hat"].shape, x.shape)
        self.assertTrue(torch.isfinite(output["x_hat"]).all())
        self.assertEqual(set(converted.state_dict()), set(model.state_dict()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in source_state.items()))

    def test_entropy_bottleneck_has_no_bf16_coder_candidate(self):
        entry = self.catalog.get("entropy_bottleneck")
        self.assertTrue(entry.precision_locked)
        self.assertEqual(
            {(candidate.key.precision, candidate.key.backend) for candidate in entry.candidates},
            {("fp32", "pytorch-cudnn")},
        )
        with self.assertRaises(fast_compressai.RegistryError):
            self.catalog.candidate("entropy_bottleneck", "bf16")


if __name__ == "__main__":
    unittest.main()
