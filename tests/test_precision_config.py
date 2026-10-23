import json
import tempfile
import unittest
from pathlib import Path

import fast_compressai
from fast_compressai.configs import PrecisionAssignment, PrecisionConfig, PrecisionConfigError


class PrecisionConfigTests(unittest.TestCase):
    def _config(self):
        return PrecisionConfig(
            model_name="FactorizedPrior",
            model_fingerprint="1" * 64,
            dataset_name="validation-set",
            dataset_fingerprint="2" * 64,
            dataset_sample_count=4,
            device_type="cuda",
            device_capability=(9, 0),
            coder="warp_div",
            p_ch=4,
            epsilon=1.0,
            rd_cost_formula="log10(bpp/bpp_ref)+(psnr_ref-psnr)",
            selector_version="caiec-operator-v1",
            catalog_fingerprint="3" * 64,
            assignments=(PrecisionAssignment(
                instance_id="model:g_a.0",
                module_path="g_a.0",
                logical_op="conv2d",
                candidate_key={
                    "logical_op": "conv2d", "precision": "fp16", "backend": "pytorch-cudnn",
                    "implementation": "fp16-module-boundary", "config": {"parameter_storage": "fp32"},
                },
                precision="fp16",
                backend="pytorch-cudnn",
                category="high_cost",
                probe_rd_cost=1.2,
                probe_thr_gain=1.4,
            ),),
            probe_records=({"kind": "operator", "target": "model:g_a.0", "candidate": "fp8/unavailable"},),
            validated_rd_cost=0.8,
            budget_met=True,
            recovery_order=("model:g_a.0",),
            profile_cost_ms=12.5,
            created_utc="20261019T120000.000000Z",
        )

    def test_round_trip_and_atomic_save_load(self):
        config = self._config()
        restored = PrecisionConfig.from_json(config.to_json())
        self.assertEqual(config, restored)
        with tempfile.TemporaryDirectory() as directory:
            path = config.save(Path(directory) / "config.json")
            self.assertEqual(PrecisionConfig.load(path), config)

    def test_compatibility_rejects_mismatched_model_or_data(self):
        config = self._config()
        with self.assertRaisesRegex(PrecisionConfigError, "model_fingerprint"):
            config.validate_compatibility(
                model_fingerprint="4" * 64,
                dataset_fingerprint="2" * 64,
                device_type="cuda",
                device_capability=(9, 0),
                coder="warp_div",
                p_ch=4,
                catalog_fingerprint="3" * 64,
            )

    def test_filename_uses_model_dataset_and_timestamp(self):
        path = PrecisionConfig.default_path("/tmp/fca-configs", "FactorizedPrior", "val-set")
        self.assertEqual(path.parent.name, "FactorizedPrior")
        self.assertTrue(path.name.startswith("val-set-"))
        self.assertTrue(path.name.endswith("Z.json"))

    def test_schema_rejects_unknown_fields(self):
        payload = self._config().to_dict()
        payload["surprise"] = True
        with self.assertRaises(PrecisionConfigError):
            PrecisionConfig.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
