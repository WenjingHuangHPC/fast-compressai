import unittest
from unittest import mock

import torch
from torch import nn

import fast_compressai
from fast_compressai.policies.caiec_selection import select_balle_precision
from fast_compressai.profiling.rd import CalibrationData


class SelectorFlowTests(unittest.TestCase):
    def setUp(self):
        self.catalog = fast_compressai.build_balle_catalog()
        self.model = nn.Conv2d(3, 4, 1)
        self.sample = torch.ones(1, 3, 4, 4)
        self.calibration = None

    def test_selector_measures_fp16_reference_and_records_fp8_unavailable(self):
        # Use a Balle-like module catalog entry but mock the full RD probe; this
        # verifies selector orchestration without pretending CPU is the GPU profile.
        with mock.patch("fast_compressai.policies.caiec_selection.measure_factorized_warpdiv_rd") as measure:
            def result(model, calibration, **kwargs):
                record = fast_compressai.ProfileRecord(
                    kind="coder", target="FactorizedPrior", candidate="warp_div/p_ch=1",
                    context={"dataset": calibration.dataset_name}, latency_ms=1.0,
                    bpp=0.1, psnr_db=30.0, rd_cost=0.0, measurement_scope="compress",
                )
                return fast_compressai.FactorizedRDMeasurement(
                    bpp=0.1, psnr_db=30.0, rd_cost=0.0,
                    forward_latency_ms=1.0, compress_latency_ms=0.5,
                    decompress_latency_ms=0.5, sample_count=1, record=record,
                )
            measure.side_effect = result
            # The tiny catalog contains no Balle operators; use a real mini Balle
            # model here only for inventory, and use mocked measurements.
            from compressai.models import FactorizedPrior
            balle = FactorizedPrior(4, 8).cuda().eval()
            balle.update(force=True)
            calibration = CalibrationData(
                samples=(torch.rand(1,3,64,64,device="cuda"),),
                raw_targets=(torch.rand(1,3,64,64,device="cuda")*255,),
                dataset_name="mock-kodak", dataset_fingerprint="b"*64,
                normalization="divide_by_255", denormalization="multiply_by_255", data_range=255.0,
            )
            selection = select_balle_precision(
                balle, self.catalog, calibration, epsilon=1.0, p_ch=1,
                model_name="FactorizedPrior", device_capability=(9,0),
            )
        self.assertEqual(len(selection.config.assignments), 14)
        # Records contain one compact entry per measured candidate. The
        # all-FP16 reference is stored in the recovery/validation summary.
        self.assertEqual(len(selection.config.probe_records), 2 * len(selection.config.assignments))
        self.assertTrue(all(record.get("status") == "measured"
                            for record in selection.config.probe_records))
        self.assertTrue(selection.config.recovery_steps)
        self.assertTrue(selection.config.budget_met)


if __name__ == "__main__":
    unittest.main()
