import unittest

import fast_compressai
from fast_compressai.policies.caiec_precision import Sensitivity


class CAIECPrecisionTests(unittest.TestCase):
    def test_paper_rd_cost_and_thr_gain_formulas(self):
        cost = fast_compressai.rd_cost(0.2, 0.1, 29.0, 30.0)
        self.assertAlmostEqual(cost, 1.0 + __import__("math").log10(2.0))
        gain = fast_compressai.thr_gain(10.0, 9.0, 10)
        self.assertAlmostEqual(gain, 2.0)

    def test_strict_category_thresholds(self):
        self.assertEqual(Sensitivity("a", 1.01, 1.0, ("fp16",)).category, "high_cost")
        self.assertEqual(Sensitivity("b", 1.0, 1.51, ("fp8",)).category, "high_gain")
        self.assertEqual(Sensitivity("c", 1.0, 1.5, ("fp8",)).category, "low_impact")

    def test_initial_assignment_respects_available_precision(self):
        candidates = (
            Sensitivity("high", 2.0, 1.0, ("fp32", "fp16")),
            Sensitivity("gain", 0.5, 2.0, ("fp32", "fp16", "fp8")),
            Sensitivity("missing_fp8", 0.5, 2.0, ("fp32", "fp16"), ("fp8 kernel unavailable",)),
            Sensitivity("reference_only", 0.5, 1.0, ("fp32",)),
        )
        self.assertEqual(
            fast_compressai.initial_assignments(candidates),
            {"high": "fp16", "gain": "fp8", "missing_fp8": "fp16", "reference_only": "fp32"},
        )

    def test_recovery_remeasures_and_uses_stable_order(self):
        candidates = (
            Sensitivity("b", 0.4, 1.2, ("fp32", "fp16", "fp8")),
            Sensitivity("a", 0.4, 1.2, ("fp32", "fp16", "fp8")),
            Sensitivity("high", 2.0, 1.0, ("fp32", "fp16")),
        )
        seen = []

        def validate(assignment):
            seen.append(dict(assignment))
            if assignment["a"] == "fp16":
                return 0.5
            return 2.0

        result, cost, changed, steps, met = fast_compressai.recover_to_budget(
            {"a": "fp8", "b": "fp8", "high": "fp16"}, candidates, validate, 1.0
        )
        self.assertEqual(changed[0], "a")
        self.assertTrue(met)
        self.assertEqual(cost, 0.5)
        self.assertEqual(result["a"], "fp16")
        self.assertEqual(len(seen), 2)

    def test_infeasible_budget_is_explicit(self):
        candidates = (Sensitivity("a", 2.0, 1.0, ("fp32", "fp16")),)
        result, cost, changed, steps, met = fast_compressai.recover_to_budget(
            {"a": "fp16"}, candidates, lambda assignment: 2.0, 1.0
        )
        self.assertFalse(met)
        self.assertEqual(result["a"], "fp32")
        self.assertEqual(changed, ("a",))
        self.assertEqual(cost, 2.0)


if __name__ == "__main__":
    unittest.main()
