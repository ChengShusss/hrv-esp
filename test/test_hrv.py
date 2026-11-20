import csv
import math
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from hrv import analyze_rr_csv, summarize_rr


class HrvTests(unittest.TestCase):
    def test_known_numbers_and_strict_pnn50_threshold(self):
        result = summarize_rr([1000, 1050, 950, 1000])
        self.assertEqual(result["mean_rr_ms"], 1000)
        self.assertEqual(result["mean_hr_bpm"], 60)
        self.assertAlmostEqual(result["sdrr_ms"], math.sqrt(5000 / 3))
        self.assertAlmostEqual(result["rmssd_ms"], math.sqrt(5000))
        self.assertAlmostEqual(result["pnn50_percent"], 100 / 3)
        self.assertAlmostEqual(result["ln_rmssd"], math.log(math.sqrt(5000)))

    def test_empty_short_and_invalid_input(self):
        self.assertIsNone(summarize_rr([])["mean_rr_ms"])
        self.assertIsNone(summarize_rr([800])["rmssd_ms"])
        self.assertEqual(summarize_rr([800, 800])["rmssd_ms"], 0)
        self.assertIsNone(summarize_rr([800, 800])["ln_rmssd"])
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                summarize_rr([800, value])

    def write_csv(self, path, records):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("source", "session", "epoch", "analysis_segment", "r_sample_index",
                             "sample_rate_hz", "rr_ms", "polarity"))
            writer.writerows(("serial", "1234ABCD", 0, segment, index, 1000, rr, 1)
                             for segment, index, rr in records)

    def test_csv_never_merges_segments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rr.csv"
            self.write_csv(path, [(0, 0, ""), (0, 800, "800"), (0, 1610, "810"),
                                  (1, 5000, ""), (1, 5900, "900"), (1, 6810, "910")])
            report = analyze_rr_csv(path)
            self.assertEqual(report["segment_count"], 2)
            self.assertEqual([s["metrics"]["rr_count"] for s in report["segments"]], [2, 2])
            self.assertEqual([s["metrics"]["rmssd_ms"] for s in report["segments"]], [10, 10])

    def test_csv_rejects_inconsistent_rr_and_invalid_boundaries(self):
        cases = [[(0, 0, ""), (0, 800, "900")],
                 [(0, 0, "800")], [(0, 0, ""), (0, 0, "0")],
                 [(0, 0, ""), (0, 800, "")], [(0, 0, ""), (1, 900, "900")]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rr.csv"
            for records in cases:
                self.write_csv(path, records)
                with self.assertRaises(ValueError):
                    analyze_rr_csv(path)

    def test_two_short_segments_do_not_combine_into_frequency_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rr.csv"
            records = []
            for segment in range(2):
                for i in range(201):
                    records.append((segment, segment * 1000000 + i * 1000, "" if i == 0 else "1000"))
            self.write_csv(path, records)
            report = analyze_rr_csv(path)
            self.assertEqual(report["segment_count"], 2)
            self.assertTrue(all(s["frequency"]["status"] == "insufficient_duration" for s in report["segments"]))
