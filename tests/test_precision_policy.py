import json
import unittest

import torch
from torch import nn

import fast_compressai


class PrecisionPolicyTests(unittest.TestCase):
    def test_builtin_policies_resolve(self):
        request = fast_compressai.PrecisionRequest(component="encoder", operator="op")
        self.assertEqual(fast_compressai.PrecisionPolicy.fp32().resolve(request).precision, "fp32")
        self.assertEqual(fast_compressai.PrecisionPolicy.uniform("fp16").resolve(request).precision, "fp16")
        self.assertEqual(
            fast_compressai.PrecisionPolicy.component({"encoder": "bf16"}).resolve(request).precision,
            "bf16",
        )
        self.assertEqual(
            fast_compressai.PrecisionPolicy.operator({"op": "fp16"}).resolve(request).precision,
            "fp16",
        )

    def test_policy_serialization_round_trip_is_stable(self):
        policy = fast_compressai.PrecisionPolicy.component(
            {"decoder": "bf16", "encoder": "fp16"}, default="fp32"
        )
        payload = policy.to_dict()
        self.assertEqual(list(payload["policy"]["assignments"]), ["decoder", "encoder"])
        restored = fast_compressai.PrecisionPolicy.from_dict(payload)
        self.assertEqual(restored, policy)
        json.dumps(payload)

    def test_fp8_policy_serializes_without_changing_default(self):
        policy = fast_compressai.PrecisionPolicy.uniform("fp8")
        restored = fast_compressai.PrecisionPolicy.from_dict(policy.to_dict())
        self.assertEqual(restored, policy)
        self.assertEqual(fast_compressai.PrecisionPolicy.fp32().resolve(fast_compressai.PrecisionRequest()).precision, "fp32")

    def test_invalid_precision_and_schema_are_rejected(self):
        with self.assertRaises(fast_compressai.PrecisionPolicyError):
            fast_compressai.PrecisionPolicy.uniform("int8")
        with self.assertRaises(fast_compressai.PrecisionPolicyError):
            fast_compressai.PrecisionPolicy.from_dict({"schema_version": 2})

    def test_custom_policy_is_runtime_only(self):
        policy = fast_compressai.PrecisionPolicy.custom(lambda request: "fp16", name="test")
        self.assertEqual(policy.resolve(fast_compressai.PrecisionRequest()).source, "custom")
        with self.assertRaisesRegex(fast_compressai.PrecisionPolicyError, "runtime-only"):
            policy.to_dict()

    def test_operator_overrides_policy_and_compile_is_reporting_only(self):
        calls = []
        registry = fast_compressai.OperatorRegistry()
        registry.register(
            "op",
            lambda x: x,
            {"fp16": lambda x: calls.append("fp16") or x},
        )
        model = nn.Linear(2, 2)
        dtype_before = tuple(parameter.dtype for parameter in model.parameters())
        compiled = fast_compressai.compile(
            model,
            operators=[{"key": "op", "precision": "fp16"}],
            registry=registry,
            precision_policy=fast_compressai.PrecisionPolicy.uniform("bf16"),
        )
        self.assertEqual(compiled.report.operator_decisions[0].requested_precision, "fp16")
        self.assertEqual(compiled.report.precision_sources["op"], "explicit_operator")
        self.assertEqual(compiled.report.selected_precisions["op"], "fp16")
        self.assertFalse(compiled.report.precision_runtime_applied)
        self.assertTrue(compiled.report.precision_evaluation_only)
        self.assertEqual(calls, [])
        self.assertEqual(tuple(parameter.dtype for parameter in model.parameters()), dtype_before)

    def test_policy_fallback_is_reported(self):
        registry = fast_compressai.OperatorRegistry()
        registry.register("op", lambda x: x, {"fp16": lambda x: x}, capabilities={"fp16": lambda _: False})
        compiled = fast_compressai.compile(
            nn.Identity(),
            operators=["op"],
            registry=registry,
            precision_policy=fast_compressai.PrecisionPolicy.uniform("fp16"),
        )
        self.assertTrue(compiled.report.precision_fallback_used)
        self.assertEqual(compiled.report.precision_map["op"], "fp32")


if __name__ == "__main__":
    unittest.main()
