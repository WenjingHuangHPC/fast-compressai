import unittest

from fast_compressai import AccelerationConfig, CompileConfig, ConfigurationError


class AccelerationConfigTests(unittest.TestCase):
    def test_warp_div_is_the_default_and_rans_is_explicit(self):
        self.assertEqual(AccelerationConfig().coder, "warp_div")
        self.assertEqual(AccelerationConfig.from_value({"coder": "rans"}).coder, "rans")
        with self.assertRaisesRegex(ConfigurationError, "coder"):
            AccelerationConfig(coder="warp_smem")

    def test_duplicate_coder_sources_fail(self):
        with self.assertRaises(ConfigurationError):
            AccelerationConfig.from_value({"coder": "rans"}, coder="warp_div")


class CompileConfigTests(unittest.TestCase):
    def test_defaults_are_reference_first(self):
        config = CompileConfig()
        self.assertEqual(config.backend, "reference")
        self.assertTrue(config.enable_reference_fallback)
        self.assertTrue(config.enable_operator_fallback)
        self.assertEqual(config.default_precision, "fp32")
        self.assertEqual(config.to_dict(), {
            "device": None,
            "backend": "reference",
            "enable_reference_fallback": True,
            "enable_operator_fallback": True,
            "default_precision": "fp32",
        })

    def test_mapping_is_normalized(self):
        config = CompileConfig.from_value({"device": "cpu"})
        self.assertEqual(config.device, "cpu")

    def test_unknown_fields_are_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "unknown compile configuration"):
            CompileConfig.from_value({"precision": "fp16"})

    def test_non_reference_backend_is_rejected(self):
        with self.assertRaisesRegex(ConfigurationError, "unsupported backend"):
            CompileConfig(backend="cuda")


if __name__ == "__main__":
    unittest.main()
