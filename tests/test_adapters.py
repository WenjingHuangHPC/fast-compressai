import json
import unittest

import torch
from torch import nn

import fast_compressai


class StyleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.register_buffer("marker", torch.zeros(1))
        self.entropy_bottleneck = nn.Identity()
        self.update_count = 0

    def forward(self, x, extra=None):
        return x * self.weight if extra is None else x * self.weight + extra

    def compress(self, x, *, mode="default"):
        return {"value": x, "mode": mode}

    def decompress(self, strings, shape=None, *, device=None):
        return {"value": strings["value"], "shape": shape, "device": device}

    def update(self, force=False):
        self.update_count += 1
        return force


class ForwardOnly(nn.Module):
    def forward(self, x):
        return x


class AdapterTests(unittest.TestCase):
    def test_generic_module_is_reported_without_compressai_requirement(self):
        info = fast_compressai.CompressAIAdapter(ForwardOnly()).info
        self.assertEqual(info.contract_family, "generic")
        self.assertEqual(info.validation_level, "forward_only")
        self.assertFalse(info.update_available)
        self.assertEqual(info.missing_paths, ("compress", "decompress"))

    def test_style_paths_and_update_are_transparent(self):
        original = StyleModel()
        compiled = fast_compressai.compile(original)
        self.assertEqual(compiled.report.contract_family, "compressai_style")
        self.assertEqual(compiled.report.signature_level, "exact")
        self.assertTrue(compiled.report.update_available)
        self.assertFalse(compiled.report.update_called)
        self.assertIn("entropy_bottleneck", compiled.report.entropy_models)

        value = torch.ones(2)
        result = compiled.compress(value, mode="fast")
        self.assertEqual(result["mode"], "fast")
        decoded = compiled.decompress(result, (2,), device="cpu")
        self.assertEqual(decoded["shape"], (2,))
        self.assertEqual(decoded["device"], "cpu")
        self.assertEqual(compiled.update(force=True), True)
        self.assertEqual(original.update_count, 1)
        self.assertTrue(compiled.report.update_called)

    def test_missing_update_is_explicit(self):
        compiled = fast_compressai.compile(ForwardOnly())
        with self.assertRaisesRegex(fast_compressai.UnsupportedOperationError, "update"):
            compiled.update()

    def test_state_dict_and_load_state_dict_are_delegated(self):
        original = StyleModel()
        compiled = fast_compressai.compile(original)
        state = compiled.state_dict()
        self.assertEqual(set(state), {"weight", "marker"})
        self.assertFalse(any(key.startswith("model.") for key in state))
        result = compiled.load_state_dict(state)
        self.assertTrue(hasattr(result, "missing_keys"))

    def test_report_is_json_serializable(self):
        report = fast_compressai.compile(StyleModel()).report.to_dict()
        encoded = json.dumps(report)
        self.assertIn("compressai_style", encoded)
        self.assertNotIn("<function", encoded)

    def test_device_moves_parameter_and_buffer(self):
        compiled = fast_compressai.compile(StyleModel(), {"device": "cpu"})
        self.assertEqual(next(compiled.parameters()).device.type, "cpu")
        self.assertEqual(compiled.marker.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
