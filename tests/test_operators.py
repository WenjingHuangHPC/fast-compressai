import unittest

import torch

import fast_compressai
from fast_compressai.operators import OperatorContext


class OperatorRegistryTests(unittest.TestCase):
    def setUp(self):
        self.registry = fast_compressai.OperatorRegistry()

    def test_reference_only_operator(self):
        reference = lambda x: x + 1
        self.registry.register("increment", reference)
        decision, implementation = self.registry.select("increment", precision="fp32")

        self.assertIs(implementation, reference)
        self.assertEqual(decision.selected_precision, "fp32")
        self.assertFalse(decision.fallback_used)
        self.assertEqual(self.registry.dispatch("increment", 2), 3)

    def test_low_precision_capability_is_selected(self):
        calls = []

        def reference(x):
            calls.append("reference")
            return x + 1

        def fp16(x):
            calls.append("fp16")
            return x + 2

        self.registry.register(
            "increment",
            reference,
            {"fp16": fp16},
            capabilities={"fp16": lambda context: context.device == "cpu"},
        )
        decision, implementation = self.registry.select(
            "increment", precision="fp16", context={"device": "cpu"}
        )

        self.assertTrue(decision.accelerated)
        self.assertFalse(decision.fallback_used)
        self.assertIs(implementation, fp16)
        self.assertEqual(self.registry.dispatch("increment", 2, precision="fp16", context={"device": "cpu"}), 4)
        self.assertEqual(calls, ["fp16"])

    def test_capability_rejection_falls_back_to_reference(self):
        reference = lambda x: x + 1
        fp16 = lambda x: x + 2
        self.registry.register(
            "increment", reference, {"fp16": fp16},
            capabilities={"fp16": lambda context: False},
        )

        decision, implementation = self.registry.select("increment", precision="fp16")
        self.assertIs(implementation, reference)
        self.assertTrue(decision.fallback_used)
        self.assertIn("capability rejected", decision.reason)
        self.assertEqual(self.registry.dispatch("increment", 2, precision="fp16"), 3)

    def test_missing_precision_falls_back_or_fails(self):
        self.registry.register("increment", lambda x: x + 1)
        decision, implementation = self.registry.select("increment", precision="bf16")
        self.assertTrue(decision.fallback_used)
        self.assertIs(implementation(2), 3)

        with self.assertRaisesRegex(fast_compressai.OperatorSelectionError, "increment"):
            self.registry.select("increment", precision="bf16", allow_fallback=False)

    def test_unknown_operator_and_duplicate_registration_are_explicit(self):
        with self.assertRaises(fast_compressai.RegistryError):
            self.registry.get("unknown")
        self.registry.register("increment", lambda x: x)
        with self.assertRaisesRegex(fast_compressai.RegistryError, "already registered"):
            self.registry.register("increment", lambda x: x)

    def test_registries_are_isolated(self):
        self.registry.register("increment", lambda x: x + 1)
        other = fast_compressai.OperatorRegistry()
        with self.assertRaises(fast_compressai.RegistryError):
            other.get("increment")

    def test_context_can_be_inferred_from_tensor_inputs(self):
        self.registry.register("identity", lambda x: x)
        context = OperatorContext.from_inputs(torch.ones(2, 3))
        self.assertEqual(context.device, "cpu")
        self.assertEqual(context.shape, (2, 3))
        self.assertIn("float", context.dtype)

    def test_runtime_errors_are_not_hidden_by_fallback(self):
        def broken(x):
            raise RuntimeError("implementation bug")

        self.registry.register("broken", lambda x: x, {"fp16": broken})
        with self.assertRaisesRegex(RuntimeError, "implementation bug"):
            self.registry.dispatch("broken", 1, precision="fp16")


if __name__ == "__main__":
    unittest.main()
