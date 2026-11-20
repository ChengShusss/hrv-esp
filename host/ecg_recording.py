"""A recording owns its files and boundaries; analysis only reads closed files."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from ecg_processing import DETECTOR_VERSION
from hrv import analyze_rr_csv


class RecordingSession:
    def __init__(self, path: Path, source="serial"):
        self.raw_path = Path(path).resolve()
        self.rr_path = self.raw_path.with_stem(self.raw_path.stem + "_rr")
        self.report_path = self.raw_path.with_stem(self.raw_path.stem + "_hrv").with_suffix(".json")
        self.source = source
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_clock = time.monotonic()
        self.last_flush = self.started_clock
        self.raw_file = self.rr_file = None
        self.closed = False
        self.error = None
        self.samples = self.leads_off = self.near_rail = self.invalid_samples = 0
        self.missing = self.restarts = self.bad_packets = self.pauses = 0
        self.sample_duration_s = 0.0
        self.peak_count = self.rr_count = 0
        self.stream = None
        self.next_sample = None
        self.first_recorded = None
        self.last_peak = None
        self.peak_key = None
        self.second_r_index = None
        self.first_r_index = None
        self.summary = None
        paths = (self.raw_path, self.rr_path, self.report_path)
        if any(path.exists() for path in paths):
            raise FileExistsError("记录文件或配套 RR/HRV 文件已存在，请选择新文件名")
        created = []
        try:
            self.raw_file = self.raw_path.open("x", encoding="utf-8", newline="")
            created.append(self.raw_path)
            self.rr_file = self.rr_path.open("x", encoding="utf-8", newline="")
            created.append(self.rr_path)
            self.raw_writer = csv.writer(self.raw_file)
            self.rr_writer = csv.writer(self.rr_file)
            self.raw_writer.writerow(("source", "session", "epoch", "sample_index", "sample_rate_hz",
                                      "time_s", "raw_adc", "lo_plus", "lo_minus"))
            self.rr_writer.writerow(("source", "session", "epoch", "analysis_segment", "r_sample_index",
                                     "sample_rate_hz", "r_time_s", "raw_adc", "rr_ms",
                                     "reported_at_sample", "polarity", "detector_version"))
        except OSError:
            for output in (self.raw_file, self.rr_file):
                if output is not None:
                    try:
                        output.close()
                    except OSError:
                        pass
            for path in created:
                path.unlink(missing_ok=True)
            raise

    @property
    def frequency_support_s(self):
        if self.last_peak is None or self.second_r_index is None:
            return 0.0
        return (self.last_peak.sample_index - self.second_r_index) / self.last_peak.sample_rate

    def add_frame(self, frame, peaks, polarity):
        if self.closed:
            raise RuntimeError("Recording is already closed")
        stream = (frame.session, frame.epoch, frame.sample_rate)
        if stream == self.stream and frame.first_sample < self.next_sample:
            return  # A duplicate delivery is not another measurement.
        if self.first_recorded is None:
            self.first_recorded = (*stream, frame.first_sample)
        if self.stream is not None:
            if stream != self.stream:
                self.restarts += 1
            elif frame.first_sample > self.next_sample:
                self.missing += frame.first_sample - self.next_sample
        self.stream = stream
        self.next_sample = frame.first_sample + len(frame.samples)
        try:
            for offset, value in enumerate(frame.samples):
                index = frame.first_sample + offset
                self.raw_writer.writerow((self.source, f"{frame.session:08X}", frame.epoch,
                                          index, frame.sample_rate, f"{index / frame.sample_rate:.6f}",
                                          value, frame.leads_off & 1, (frame.leads_off >> 1) & 1))
                self.samples += 1
                self.leads_off += bool(frame.leads_off)
                rail = value <= 8 or value >= 4087
                self.near_rail += rail
                self.invalid_samples += bool(frame.leads_off or rail)
                self.sample_duration_s += 1 / frame.sample_rate
            for peak in peaks:
                # Delayed confirmations from before Start are outside this file.
                if (peak.session, peak.epoch, peak.sample_rate) == self.first_recorded[:3] and peak.sample_index < self.first_recorded[3]:
                    continue
                if (peak.session, peak.epoch, peak.sample_rate) != stream or peak.sample_index >= self.next_sample:
                    raise ValueError("R peak falls outside the recorded sample stream")
                key = (peak.session, peak.epoch, peak.segment, peak.sample_rate, polarity)
                same = self.peak_key == key
                if same and peak.sample_index <= self.last_peak.sample_index:
                    continue
                if same:
                    rr_ms = (peak.sample_index - self.last_peak.sample_index) * 1000 / peak.sample_rate
                    if peak.rr_ms is None or abs(peak.rr_ms - rr_ms) > 1e-6:
                        raise ValueError("Detector RR continuity is inconsistent")
                    if self.second_r_index is None:
                        self.second_r_index = peak.sample_index
                    self.rr_count += 1
                else:
                    rr_ms = None
                    self.first_r_index = peak.sample_index
                    self.second_r_index = None
                self.rr_writer.writerow((self.source, f"{peak.session:08X}", peak.epoch, peak.segment,
                                        peak.sample_index, peak.sample_rate, f"{peak.time_s:.6f}",
                                        peak.raw_adc, "" if rr_ms is None else f"{rr_ms:.3f}",
                                        peak.reported_at_sample, polarity, DETECTOR_VERSION))
                self.peak_key, self.last_peak = key, peak
                self.peak_count += 1
            now = time.monotonic()
            if now - self.last_flush >= 1:
                self.raw_file.flush()
                self.rr_file.flush()
                self.last_flush = now
        except (OSError, ValueError) as error:
            self.error = str(error)
            raise

    def close(self, reason="manual"):
        if self.closed:
            return self.summary
        self.closed = True
        for output in (self.raw_file, self.rr_file):
            try:
                output.close()
            except OSError as error:
                self.error = str(error)
        self.summary = {"source": self.source, "raw_csv": str(self.raw_path), "rr_csv": str(self.rr_path),
                        "report_json": str(self.report_path), "started_at_utc": self.started_at,
                        "ended_at_utc": datetime.now(timezone.utc).isoformat(),
                        "wall_duration_s": time.monotonic() - self.started_clock,
                        "captured_duration_s": self.sample_duration_s, "stop_reason": reason,
                        "samples": self.samples, "missing_samples": self.missing,
                        "bad_packets": self.bad_packets, "device_restarts": self.restarts,
                        "data_pauses": self.pauses, "leads_off_samples": self.leads_off,
                        "near_rail_samples": self.near_rail, "invalid_samples": self.invalid_samples,
                        "r_peaks": self.peak_count, "rr_intervals": self.rr_count,
                        "io_error": self.error, "detector_version": DETECTOR_VERSION,
                        "boundary_policy": "Accepted samples and confirmed peaks at Stop; no late peak append"}
        return self.summary


def save_report_new(report, path: Path):
    """Publish a new complete JSON; never replace another recording's report."""
    document = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    created = False
    try:
        with Path(path).open("x", encoding="utf-8") as output:
            created = True
            output.write(document)
    except OSError:
        if created:
            Path(path).unlink(missing_ok=True)
        raise


def analyze_recording(metadata):
    """Worker entry: files are closed; no GUI access or recording mutations."""
    if metadata["io_error"]:
        report = {"segment_count": 0, "segments": [], "analysis_error": "记录写入失败，未计算 HRV"}
    else:
        try:
            report = analyze_rr_csv(Path(metadata["rr_csv"]))
        except (OSError, ValueError) as error:
            report = {"segment_count": 0, "segments": [], "analysis_error": str(error)}
    report["recording"] = metadata
    try:
        with Path(metadata["raw_csv"]).open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(65536), b""):
                digest.update(chunk)
        report["raw_sha256"] = digest.hexdigest()
        save_report_new(report, Path(metadata["report_json"]))
        return {"report": report, "saved_path": metadata["report_json"], "save_error": None}
    except OSError as error:
        return {"report": report, "saved_path": None, "save_error": str(error)}
