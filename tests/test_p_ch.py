import unittest

from fast_compressai import choose_p_ch, legal_p_candidates


class PChSelectionTests(unittest.TestCase):
    def test_legal_candidates_are_sorted_and_bounded(self):
        self.assertEqual(legal_p_candidates(8, [8, 1, 3, 3, 99, 0]), (1, 3, 8))

    def test_selects_smallest_balanced_p(self):
        result = choose_p_ch([
            {"p_ch": 1, "inference_total_ms": 10.0, "codec_total_ms": 30.0},
            {"p_ch": 2, "inference_total_ms": 10.0, "codec_total_ms": 10.8},
            {"p_ch": 4, "inference_total_ms": 10.0, "codec_total_ms": 9.5},
        ])
        self.assertEqual(result.selected_p_ch, 4)
        self.assertEqual(result.status, "auto_selected")

    def test_falls_back_to_closest_balance(self):
        result = choose_p_ch([
            {"p_ch": 1, "inference_total_ms": 10.0, "codec_total_ms": 20.0},
            {"p_ch": 2, "inference_total_ms": 10.0, "codec_total_ms": 15.0},
        ])
        self.assertEqual(result.selected_p_ch, 2)
        self.assertEqual(result.status, "fallback_closest_balance")

    def test_detects_platform(self):
        result = choose_p_ch([
            {"p_ch": 1, "inference_total_ms": 10.0, "codec_total_ms": 30.0},
            {"p_ch": 2, "inference_total_ms": 10.0, "codec_total_ms": 20.0},
            {"p_ch": 4, "inference_total_ms": 10.0, "codec_total_ms": 19.5},
            {"p_ch": 8, "inference_total_ms": 10.0, "codec_total_ms": 19.3},
        ], plateau_patience=2)
        self.assertTrue(result.plateau["detected"])


if __name__ == "__main__":
    unittest.main()
