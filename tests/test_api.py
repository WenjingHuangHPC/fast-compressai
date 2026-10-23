import unittest

import torch
from torch import nn

import fast_compressai
from fast_compressai import UnsupportedOperationError


class CompressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4)

    def compress(self, x):
        return {"value": x.detach()}

    def decompress(self, strings, shape=None):
        del shape
        return strings["value"]


class ReferenceModelTests(unittest.TestCase):
    def test_compile_preserves_forward_and_state_dict(self):
        model = nn.Linear(4, 4)
        compiled = fast_compressai.compile(model)
        value = torch.randn(2, 4)

        torch.testing.assert_close(compiled(value), model(value))
        self.assertEqual(set(compiled.state_dict()), set(model.state_dict()))
        self.assertFalse(compiled.report.accelerated)
        self.assertEqual(compiled.report.missing_paths, ("compress", "decompress"))

    def test_compression_paths_are_forwarded(self):
        model = CompressionModel()
        compiled = fast_compressai.compile(model)
        value = torch.randn(2, 4)

        strings = compiled.compress(value)
        torch.testing.assert_close(compiled.decompress(strings, (2, 4)), value)
        self.assertEqual(compiled.report.missing_paths, ())

    def test_missing_compression_path_fails_explicitly(self):
        compiled = fast_compressai.compile(nn.Identity())
        with self.assertRaisesRegex(UnsupportedOperationError, "does not define compress"):
            compiled.compress(torch.ones(1))
        with self.assertRaisesRegex(UnsupportedOperationError, "does not define decompress"):
            compiled.decompress(None, None)

    def test_device_config_is_applied(self):
        compiled = fast_compressai.compile(nn.Linear(2, 2), {"device": "cpu"})
        self.assertEqual(next(compiled.parameters()).device.type, "cpu")

    def test_optimize_uses_reference_path(self):
        model = nn.Identity()
        compiled = fast_compressai.optimize(model)
        self.assertIsInstance(compiled, fast_compressai.FastModel)
        self.assertEqual(compiled.report.backend, "reference")

    def test_invalid_model_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "torch.nn.Module"):
            fast_compressai.compile(object())

    def test_explicit_operator_selection_only_updates_report(self):
        registry = fast_compressai.OperatorRegistry()
        reference = lambda x: x + 1
        low_precision = lambda x: x + 2
        registry.register("increment", reference, {"fp16": low_precision})
        model = nn.Identity()
        module_names = tuple(name for name, _ in model.named_modules())
        state_keys = set(model.state_dict())

        compiled = fast_compressai.compile(
            model,
            operators=[{"key": "increment", "precision": "fp16"}],
            registry=registry,
        )

        self.assertFalse(compiled.report.accelerated)
        self.assertTrue(compiled.report.candidate_accelerated)
        self.assertFalse(compiled.report.runtime_applied)
        self.assertFalse(compiled.report.operator_fallback_used)
        self.assertEqual(compiled.report.operator_decisions[0].selected_precision, "fp16")
        self.assertEqual(tuple(name for name, _ in model.named_modules()), module_names)
        self.assertEqual(set(model.state_dict()), state_keys)
        self.assertEqual(compiled.report.to_dict()["operator_decisions"][0]["key"], "increment")

    def test_operator_fallback_is_reported_separately_from_missing_paths(self):
        registry = fast_compressai.OperatorRegistry()
        registry.register("increment", lambda x: x + 1, {"fp16": lambda x: x + 2}, capabilities={"fp16": lambda _: False})
        compiled = fast_compressai.compile(
            nn.Linear(2, 2),
            operators=[{"key": "increment", "precision": "fp16"}],
            registry=registry,
        )
        self.assertTrue(compiled.report.operator_fallback_used)
        self.assertTrue(compiled.report.fallback_used)
        self.assertEqual(compiled.report.missing_paths, ("compress", "decompress"))


if __name__ == "__main__":
    unittest.main()
