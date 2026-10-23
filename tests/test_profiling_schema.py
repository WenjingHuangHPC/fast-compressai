import json
import unittest

import fast_compressai


class ProfileSchemaTests(unittest.TestCase):
    def test_profile_set_json_round_trip_and_stable_order(self):
        coder = fast_compressai.ProfileRecord(
            kind="coder",
            target="latent",
            candidate="ga-ans:p_ch=8",
            context={"model": "mock", "device": "cpu", "shape": [1, 8, 4, 4]},
            latency_ms=1.25,
            compression_ratio=1.7,
            provenance="measured",
            repetitions=5,
        )
        operator = fast_compressai.ProfileRecord(
            kind="operator",
            target="analysis",
            candidate="fp32",
            context={"model": "mock", "device": "cpu"},
            latency_ms=2.0,
            quality_delta=0.0,
        )
        profile_set = fast_compressai.ProfileSet((coder, operator))
        restored = fast_compressai.ProfileSet.from_json(profile_set.to_json())
        self.assertEqual(restored, profile_set)
        serialized = json.loads(profile_set.to_json())
        self.assertEqual([record["kind"] for record in serialized["records"]], ["coder", "operator"])

    def test_estimated_and_unavailable_profiles_are_distinct(self):
        estimated = fast_compressai.ProfileRecord(
            kind="operator", target="analysis", candidate="fp16",
            context={"device": "gpu-x"}, latency_ms=0.5,
            psnr_drop_db=0.02, provenance="estimated",
        )
        unavailable = fast_compressai.ProfileRecord(
            kind="operator", target="entropy", candidate="fp16",
            context={"device": "cpu"}, provenance="measured",
            unavailable_reason="no implementation registered",
        )
        self.assertEqual(estimated.provenance, "estimated")
        self.assertIn("no implementation", unavailable.unavailable_reason)

    def test_invalid_schema_measurements_and_unknown_fields_fail(self):
        with self.assertRaises(fast_compressai.ProfileError):
            fast_compressai.ProfileRecord(
                kind="operator", target="op", candidate="fp32", context={}, latency_ms=float("nan")
            )
        with self.assertRaises(fast_compressai.ProfileError):
            fast_compressai.ProfileRecord(
                kind="operator", target="op", candidate="fp32", context={}, latency_ms=-1
            )
        profile = fast_compressai.ProfileRecord(
            kind="operator", target="op", candidate="fp32", context={}, latency_ms=1
        ).to_dict()
        profile["unexpected"] = True
        with self.assertRaises(fast_compressai.ProfileError):
            fast_compressai.ProfileRecord.from_dict(profile)
        with self.assertRaises(fast_compressai.ProfileError):
            fast_compressai.ProfileSet((), schema_version=4)

    def test_context_requires_json_safe_identity(self):
        with self.assertRaises(fast_compressai.ProfileError):
            fast_compressai.ProfileRecord(
                kind="operator", target="op", candidate="fp32",
                context={"device": object()}, latency_ms=1,
            )


if __name__ == "__main__":
    unittest.main()
