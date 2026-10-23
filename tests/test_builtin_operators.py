import unittest

import torch
from torch import nn

import fast_compressai


class BuiltinOperatorCatalogTests(unittest.TestCase):
    def setUp(self):
        self.catalog = fast_compressai.build_balle_catalog()

    def test_catalog_has_reference_and_bf16_gpu_candidates(self):
        for logical_op in ("conv2d", "conv_transpose2d", "gdn", "igdn"):
            with self.subTest(logical_op=logical_op):
                entry = self.catalog.get(logical_op)
                candidates = {(candidate.key.precision, candidate.key.backend) for candidate in entry.candidates}
                self.assertIn(("fp32", "pytorch-cudnn"), candidates)
                self.assertIn(("fp16", "pytorch-cudnn"), candidates)
                self.assertIn(("bf16", "pytorch-cudnn"), candidates)
                self.assertIn(("fp8", "pytorch-scaled-mm"), candidates)
                self.assertIn(("fp8", "triton"), candidates)
        entropy = self.catalog.get("entropy_bottleneck")
        self.assertTrue(entropy.precision_locked)
        self.assertEqual([candidate.key.precision for candidate in entropy.candidates], ["fp32"])

    def test_inventory_distinguishes_gdn_inverse(self):
        import compressai.layers

        class MiniBalle(nn.Module):
            def __init__(self):
                super().__init__()
                self.g_a = nn.Sequential(
                    nn.Conv2d(3, 4, 3, padding=1),
                    compressai.layers.GDN(4),
                    nn.Conv2d(4, 4, 3, padding=1),
                )
                self.g_s = nn.Sequential(
                    nn.ConvTranspose2d(4, 4, 3, padding=1),
                    compressai.layers.GDN(4, inverse=True),
                )

            def forward(self, x):
                return self.g_s(self.g_a(x))

        inventory = self.catalog.inventory(MiniBalle())
        logical_ops = [instance.logical_op for instance in inventory]
        self.assertEqual(logical_ops.count("conv2d"), 2)
        self.assertEqual(logical_ops.count("conv_transpose2d"), 1)
        self.assertEqual(logical_ops.count("gdn"), 1)
        self.assertEqual(logical_ops.count("igdn"), 1)

    @unittest.skip("GPU factory test is covered by tests/gpu/test_factorized_operators.py")
    def test_bf16_factory_keeps_fp32_weights_and_does_not_alias_source(self):
        source = nn.Conv2d(3, 4, 3, padding=1).cuda().eval()
        replacement = self.catalog.build(
            source,
            "conv2d",
            "bf16",
            context={"training": False},
        )
        self.assertIsNot(replacement, source)
        self.assertEqual(replacement.weight.dtype, torch.float32)
        self.assertEqual(replacement.weight.device.type, "cuda")
        self.assertTrue(torch.equal(replacement.weight, source.weight))
        x = torch.randn(2, 3, 16, 16, device="cuda")
        with torch.inference_mode():
            y = replacement(x)
        self.assertEqual(y.dtype, torch.float32)
        self.assertEqual(y.device.type, "cuda")
        self.assertEqual(tuple(y.shape), (2, 4, 16, 16))

    @unittest.skip("GPU factory test is covered by tests/gpu/test_factorized_operators.py")
    def test_per_instance_assignment_applies_only_selected_module(self):
        model = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1),
        ).cuda().eval()
        inventory = self.catalog.inventory(model)
        self.assertEqual(len(inventory), 2)
        source_state = {key: value.clone() for key, value in model.state_dict().items()}
        converted, applications = self.catalog.apply(
            model,
            {inventory[0].instance_id: "bf16"},
        )
        self.assertEqual(len(applications), 1)
        self.assertEqual(applications[0].module_path, "0")
        self.assertNotEqual(type(converted[0]), type(model[0]))
        self.assertEqual(type(converted[2]), type(model[2]))
        self.assertEqual(set(converted.state_dict()), set(model.state_dict()))
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in source_state.items()))
        self.assertEqual(converted[0].weight.dtype, torch.float32)

    def test_cuda_capability_rejects_cpu_candidate(self):
        module = nn.Conv2d(3, 4, 3, padding=1)
        candidate = self.catalog.candidate("conv2d", "bf16")
        supported, reason = candidate.capability(module, {})
        self.assertFalse(supported)
        self.assertIn("CUDA", reason)


if __name__ == "__main__":
    unittest.main()
