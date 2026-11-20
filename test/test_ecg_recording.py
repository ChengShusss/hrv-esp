import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from ecg_protocol import Frame
from ecg_processing import RPeak
from ecg_recording import RecordingSession, analyze_recording, save_report_new


class RecordingTests(unittest.TestCase):
    def test_boundaries_duplicates_and_first_rr(self):
        with tempfile.TemporaryDirectory() as directory:
            session = RecordingSession(Path(directory) / "capture.csv")
            frame = Frame(123, 0, 1000, 1000, 0, (2000,) * 32)
            before = RPeak(123, 0, 0, 990, 1000, 2000, 800, 1008)
            first = RPeak(123, 0, 0, 1005, 1000, 2000, 815, 1020)
            session.add_frame(frame, [before, first], 1)
            session.add_frame(frame, [first], 1)
            for start in range(1032, 1832, 32):
                peak = RPeak(123, 0, 0, 1805, 1000, 2000, 800, 1815)
                session.add_frame(Frame(123, 0, start, 1000, 0, (2000,) * 32),
                                  [peak] if start <= 1815 < start + 32 else [], 1)
            metadata = session.close()
            self.assertEqual(metadata["samples"], 832)
            self.assertEqual(metadata["r_peaks"], 2)
            self.assertEqual(metadata["rr_intervals"], 1)
            self.assertEqual(session.close(), metadata)
            with session.rr_path.open() as source:
                peaks = list(csv.DictReader(source))
            self.assertEqual([p["r_sample_index"] for p in peaks], ["1005", "1805"])
            self.assertEqual([p["rr_ms"] for p in peaks], ["", "800.000"])
            with self.assertRaises(RuntimeError):
                session.add_frame(frame, [], 1)
            result = analyze_recording(metadata)
            self.assertTrue(Path(result["saved_path"]).exists())
            self.assertEqual(result["report"]["segments"][0]["frequency"]["status"], "insufficient_duration")

    def test_quality_counts_and_empty_record(self):
        with tempfile.TemporaryDirectory() as directory:
            session = RecordingSession(Path(directory) / "bad.csv", "demo")
            session.add_frame(Frame(123, 0, 0, 1000, 1, (0, 4095, 2000)), [], 1)
            session.add_frame(Frame(123, 0, 10, 1000, 0, (2000,)), [], 1)
            session.add_frame(Frame(123, 1, 0, 1000, 0, (2000,)), [], 1)
            report = analyze_recording(session.close("disconnected"))["report"]
            self.assertEqual(report["segment_count"], 0)
            self.assertEqual(report["recording"]["invalid_samples"], 3)
            self.assertEqual(report["recording"]["near_rail_samples"], 2)
            self.assertEqual(report["recording"]["missing_samples"], 7)
            self.assertEqual(report["recording"]["device_restarts"], 1)
            self.assertEqual(report["recording"]["source"], "demo")
            empty = RecordingSession(Path(directory) / "empty.csv")
            self.assertEqual(analyze_recording(empty.close())["report"]["segment_count"], 0)

    def test_existing_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.csv"
            companion = path.with_stem("saved_rr")
            companion.write_text("keep")
            with self.assertRaises(FileExistsError):
                RecordingSession(path)
            self.assertFalse(path.exists())
            self.assertEqual(companion.read_text(), "keep")
            report = Path(directory) / "report.json"
            report.write_text("keep")
            with self.assertRaises(FileExistsError):
                save_report_new({}, report)
            self.assertEqual(report.read_text(), "keep")

    def test_open_failure_cleans_only_new_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture.csv"
            original_open = Path.open
            def failing_open(candidate, *args, **kwargs):
                if candidate.name.endswith("_rr.csv"):
                    raise PermissionError("denied")
                return original_open(candidate, *args, **kwargs)
            with patch.object(Path, "open", failing_open), self.assertRaises(PermissionError):
                RecordingSession(path)
            self.assertFalse(path.exists())

    def test_write_and_close_failure_do_not_publish_hrv_values(self):
        with tempfile.TemporaryDirectory() as directory:
            session = RecordingSession(Path(directory) / "write.csv")
            session.raw_writer = Mock()
            session.raw_writer.writerow.side_effect = OSError("disk full")
            with self.assertRaises(OSError):
                session.add_frame(Frame(123, 0, 0, 1000, 0, (2000,)), [], 1)
            report = analyze_recording(session.close("write_error"))["report"]
            self.assertIn("analysis_error", report)
            self.assertEqual(report["segments"], [])
            session = RecordingSession(Path(directory) / "close.csv")
            real_output = session.raw_file
            session.raw_file = Mock()
            session.raw_file.close.side_effect = OSError("flush failed")
            metadata = session.close()
            real_output.close()
            self.assertEqual(metadata["io_error"], "flush failed")
            self.assertEqual(analyze_recording(metadata)["report"]["segments"], [])

    def test_report_save_failure_retains_calculated_result(self):
        with tempfile.TemporaryDirectory() as directory:
            session = RecordingSession(Path(directory) / "capture.csv")
            metadata = session.close()
            session.report_path.write_text("another report")
            result = analyze_recording(metadata)
            self.assertIsNone(result["saved_path"])
            self.assertIsNotNone(result["save_error"])
            self.assertIn("recording", result["report"])
            self.assertEqual(session.report_path.read_text(), "another report")
