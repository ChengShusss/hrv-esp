import cmath
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from hrv_spectrum import _fft, _cubic_resample, _welch_density, _band_power, frequency_domain


class SpectrumTests(unittest.TestCase):
    def test_fft_matches_direct_dft(self):
        data = [2, -1, 4, 7, 3, -5, 0, 1]
        expected = [sum(v * cmath.exp(-2j * math.pi * k * i / len(data))
                        for i, v in enumerate(data)) for k in range(len(data))]
        self.assertTrue(all(abs(a - b) < 1e-10 for a, b in zip(_fft(data), expected)))

    def test_spline_preserves_linear_data_without_extrapolation(self):
        times = [0, .9, 1.7, 3.2]
        samples = _cubic_resample(times, [1 + 2 * t for t in times])
        self.assertEqual(len(samples), 13)
        for i, value in enumerate(samples):
            self.assertAlmostEqual(value, 1 + 2 * i / 4)

    def test_psd_has_correct_units_and_known_band_powers(self):
        values = [800 + 30 * math.sin(2 * math.pi * .09375 * i / 4)
                  + 20 * math.sin(2 * math.pi * .25 * i / 4) for i in range(2400)]
        f, psd, parameters = _welch_density(values)
        self.assertAlmostEqual(_band_power(f, psd, .04, .15), 30**2 / 2, delta=.2)
        self.assertAlmostEqual(_band_power(f, psd, .15, .4), 20**2 / 2, delta=.2)
        self.assertAlmostEqual(_band_power(f, psd, 0, 2), 650, delta=1)
        self.assertTrue(parameters["tail_window_added"])

    def test_exact_band_boundary_integration(self):
        self.assertAlmostEqual(_band_power([0, .1, .2], [0, 10, 0], .04, .15), .795)

    def test_short_and_suspect_series_are_not_reported_as_zero_power(self):
        short = frequency_domain([800] * 100)
        self.assertEqual(short["status"], "insufficient_duration")
        self.assertIsNone(short["lf_ms2"])
        data = [800] * 450
        data[200] = 1600
        flagged = frequency_domain(data)
        self.assertEqual(flagged["status"], "rr_review_required")
        self.assertIsNone(flagged["hf_ms2"])

    def test_constant_rr_has_zero_power_and_undefined_ratio(self):
        result = frequency_domain([1000] * 301)
        self.assertEqual(result["status"], "available_rr_estimate")
        self.assertEqual(result["lf_ms2"], 0)
        self.assertEqual(result["hf_ms2"], 0)
        self.assertIsNone(result["lf_hf_ratio"])
        self.assertIsNone(result["lf_nu_percent"])

    def test_irregular_rr_recovers_lf_and_hf_components(self):
        rr, stamp = [], 0.0
        while stamp < 360:
            interval = 800 + 30 * math.sin(2 * math.pi * .1 * stamp) + 20 * math.sin(2 * math.pi * .25 * stamp)
            rr.append(interval)
            stamp += interval / 1000
        result = frequency_domain(rr)
        self.assertEqual(result["status"], "available_rr_estimate")
        self.assertAlmostEqual(result["lf_ms2"], 450, delta=30)
        self.assertAlmostEqual(result["hf_ms2"], 200, delta=30)
        self.assertAlmostEqual(result["lf_nu_percent"] + result["hf_nu_percent"], 100)
        for lower, upper, target in ((.04, .15, .1), (.15, .4, .25)):
            peak = max((p, f) for p, f in zip(result["psd_ms2_per_hz"], result["frequencies_hz"])
                       if lower <= f <= upper)[1]
            self.assertAlmostEqual(peak, target, delta=.004)
