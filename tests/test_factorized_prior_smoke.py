import math
import unittest

import torch

try:
    import compressai
    from compressai.models import FactorizedPrior
except ImportError:
    compressai = None
    FactorizedPrior = None

import fast_compressai


@unittest.skipIf(compressai is None, "CompressAI is optional")
class FactorizedPriorSmokeTests(unittest.TestCase):
    def test_cpu_accelerate_forward_and_entropy_round_trip(self):
        torch.manual_seed(20261015)
        torch.set_num_threads(1)
        model = FactorizedPrior(N=8, M=12).cpu()
        self.assertTrue(model.update(force=True))
        model.eval()
        sample = torch.rand(1, 3, 64, 64)
        source_state = {key: value.clone() for key, value in model.state_dict().items()}

        reference = fast_compressai.compile(model, {"device": "cpu"})
        accelerated = fast_compressai.accelerate(
            model, P=8, budget=0.1, device="cpu", sample=sample, coder="rans"
        )

        with torch.inference_mode():
            reference_output = reference(sample)
            accelerated_output = accelerated(sample)
            reference_strings = reference.compress(sample)
            accelerated_strings = accelerated.compress(sample)
            reference_recon = reference.decompress(
                reference_strings["strings"], reference_strings["shape"]
            )["x_hat"]
            accelerated_recon = accelerated.decompress(
                accelerated_strings["strings"], accelerated_strings["shape"]
            )["x_hat"]

        report = accelerated.acceleration_report
        self.assertEqual(report.profiling_status, "measured")
        self.assertIn(report.precision_selection_status, {"selected", "fallback_fp32"})
        self.assertEqual(report.quality_metric, "psnr_drop_db")
        self.assertEqual(report.coder_status, "rans")
        self.assertFalse(report.p_ch_applied)
        self.assertTrue(torch.equal(reference_output["x_hat"], accelerated_output["x_hat"]))
        self.assertEqual(reference_strings["strings"], accelerated_strings["strings"])
        self.assertTrue(torch.equal(reference_recon, accelerated_recon))
        self.assertEqual(tuple(accelerated_recon.shape), tuple(sample.shape))
        self.assertTrue(
            all(torch.equal(value, model.state_dict()[key]) for key, value in source_state.items())
        )

        mse = float(((sample - accelerated_recon) ** 2).mean())
        psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
        bits = sum(len(stream) for group in accelerated_strings["strings"] for stream in group) * 8
        bpp = bits / (sample.shape[-2] * sample.shape[-1])
        self.assertTrue(math.isfinite(psnr))
        self.assertGreater(bpp, 0.0)


if __name__ == "__main__":
    unittest.main()
