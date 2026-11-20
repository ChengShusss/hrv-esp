import json
from pathlib import Path
import sys
import tempfile
import tkinter as tk
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "host"), str(Path(__file__).resolve().parent)]
from ecg_protocol import Frame, encode_frame
from plot_ecg import EcgApp
from test_ecg_processing import synthetic


class GuiRecordingTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as error:
            self.skipTest(str(error))
        self.app = EcgApp(self.root, demo=True)
        self.root.after_cancel(self.app.tick_id)
        self.app.tick_id = None
        self.root.update()
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self):
        if not self.app.closed:
            self.app.close()
        self.app.analysis_executor.shutdown(wait=True)
        self.directory.cleanup()

    def feed(self):
        values = synthetic()
        for first in range(0, len(values), 32):
            self.app.ingest(encode_frame(Frame(123, 0, first, 1000, 0, tuple(values[first:first+32]))))

    def finish(self):
        self.app.analysis_future.result(timeout=5)
        self.app.poll_analysis()
        self.root.update()

    def test_start_stop_buttons_generate_and_display_report(self):
        path = Path(self.directory.name) / "capture.csv"
        with patch("plot_ecg.filedialog.asksaveasfilename", return_value=str(path)):
            self.app.record_button.invoke()
        self.assertIsNotNone(self.app.recording)
        self.assertEqual(str(self.app.record_button['state']), "disabled")
        self.feed()
        self.app.stop_record_button.invoke()
        future = self.app.analysis_future
        self.app.stop_recording()
        self.assertIs(self.app.analysis_future, future)
        self.finish()
        report = json.loads(path.with_stem("capture_hrv").with_suffix(".json").read_text(encoding="utf-8"))
        self.assertEqual(report["recording"]["source"], "demo")
        self.assertGreater(report["segments"][0]["metrics"]["rr_count"], 5)
        self.assertEqual(report["segments"][0]["frequency"]["status"], "insufficient_duration")
        self.assertEqual(self.app.result_window.frequency_values["lf_ms2"].get(), "—")
        self.assertTrue(self.app.result_window.rr_canvas.find_withtag("data_curve"))
        self.assertEqual(str(self.app.record_button['state']), "normal")
        self.app.start_recording(Path(self.directory.name) / "second.csv")
        self.app.stop_recording()
        self.finish()
        self.assertEqual(self.app.latest_result["report"]["segment_count"], 0)

    def test_disconnect_ends_recording_and_saves_partial_report(self):
        self.app.start_recording(Path(self.directory.name) / "disconnect.csv")
        self.feed()
        self.app.disconnect("test disconnect")
        self.finish()
        self.assertEqual(self.app.latest_result["report"]["recording"]["stop_reason"], "disconnected")
        self.assertIsNone(self.app.recording)

    def test_window_close_finishes_report_without_gui_callbacks(self):
        path = Path(self.directory.name) / "closed.csv"
        self.app.start_recording(path)
        self.feed()
        self.app.close()
        result = self.app.analysis_future.result(timeout=5)
        self.assertTrue(Path(result["saved_path"]).exists())
        self.assertEqual(result["report"]["recording"]["stop_reason"], "window_closed")
