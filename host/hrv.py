"""Time/frequency HRV estimates from RR CSV, keeping every continuous segment."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from hrv_spectrum import frequency_domain, rr_quality

HRV_VERSION = "hrv-r2-welch256s-cubic4hz"


def _finish_segment(segment, rr, intervals_enabled):
    segment["metrics"] = summarize_rr(rr)
    segment["quality"] = rr_quality(rr)
    segment["frequency"] = frequency_domain(rr, segment["quality"])
    if intervals_enabled:
        segment["rr_intervals_ms"] = list(rr)
    return segment


def summarize_rr(intervals):
    """One continuous RR series, milliseconds; no deletion or interpolation."""
    rr = [float(value) for value in intervals]
    if any(not math.isfinite(value) or value <= 0 for value in rr):
        raise ValueError("RR values must be finite and positive")
    differences = [right - left for left, right in zip(rr, rr[1:])]
    rmssd = math.sqrt(statistics.mean(d * d for d in differences)) if differences else None
    return {
        "rr_count": len(rr), "adjacent_difference_count": len(differences),
        "rr_duration_s": sum(rr) / 1000,
        "mean_rr_ms": statistics.mean(rr) if rr else None,
        "mean_hr_bpm": 60000 / statistics.mean(rr) if rr else None,
        # SDRR is deliberately used: the input has not been classified as NN.
        "sdrr_ms": statistics.stdev(rr) if len(rr) >= 2 else None,
        "rmssd_ms": rmssd,
        "ln_rmssd": math.log(rmssd) if rmssd is not None and rmssd > 0 else None,
        "pnn50_percent": (100 * sum(abs(d) > 50 for d in differences) / len(differences)
                          if differences else None),
        "min_rr_ms": min(rr) if rr else None, "max_rr_ms": max(rr) if rr else None,
    }


def analyze_rr_csv(path: Path, *, include_intervals=True):
    """Validate exported intervals and calculate each continuous segment alone."""
    segments = []
    key = None
    previous_index = None
    rr = []
    segment = None
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"source", "session", "epoch", "analysis_segment", "r_sample_index",
                    "sample_rate_hz", "rr_ms", "polarity"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Not a supported RR CSV: missing required columns")
        for line, row in enumerate(reader, start=2):
            try:
                rate = int(row["sample_rate_hz"])
                index = int(row["r_sample_index"])
                if rate <= 0 or index < 0:
                    raise ValueError("invalid sample index or rate")
                new_key = (row["source"], row["session"], int(row["epoch"]),
                           int(row["analysis_segment"]), rate, int(row["polarity"]),
                           row.get("detector_version", "legacy-unspecified"))
                text = row["rr_ms"].strip()
                if new_key != key:
                    if text:
                        raise ValueError("first peak in a segment must have empty RR")
                    if segment is not None:
                        segments.append(_finish_segment(segment, rr, include_intervals))
                    key, rr = new_key, []
                    segment = {"source": key[0], "session": key[1], "epoch": key[2],
                               "analysis_segment": key[3], "sample_rate_hz": rate,
                               "polarity": key[5], "detector_version": key[6],
                               "first_r_sample": index, "last_r_sample": index, "peak_count": 1}
                else:
                    if index <= previous_index or not text:
                        raise ValueError("non-increasing peak or missing RR within a segment")
                    interval = float(text)
                    expected = (index - previous_index) * 1000 / rate
                    if not math.isfinite(interval) or abs(interval - expected) > 0.00051:
                        raise ValueError("RR does not match original sample indices")
                    rr.append(interval)
                    segment["peak_count"] += 1
                    segment["last_r_sample"] = index
                previous_index = index
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"RR CSV line {line}: {error}") from error
    if segment is not None:
        segments.append(_finish_segment(segment, rr, include_intervals))
    return {"rr_csv": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "hrv_version": HRV_VERSION,
            "basis": "RR estimates; normal-to-normal (NN) classification not performed",
            "conventions": {"sdrr": "sample standard deviation, denominator N-1",
                            "rmssd": "sqrt(mean(square(adjacent RR differences)))",
                            "pnn50": "100 * count(abs(adjacent difference) > 50 ms) / (N-1)",
                            "mean_hr": "60000 / mean RR in ms"},
            "correction": "no beat deletion, imputation, or joining across gaps; uniform resampling for PSD only",
            "segment_count": len(segments), "segments": segments,
            "notes": ["Short segments are duration-specific estimates, not equivalent to a 5-minute recording.",
                      "SDRR is not validated SDNN without confirming normal beats.",
                      "LF/HF requires at least 300 seconds of continuous RR interpolation support and no review flags.",
                      "LF/HF is not a stress score or a direct measure of autonomic balance.",
                      "VLF and clinical interpretation are not included in this short-term report."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rr_csv", type=Path)
    parser.add_argument("--output", type=Path, help="Save JSON to a new file; existing files are preserved")
    args = parser.parse_args()
    try:
        report = analyze_rr_csv(args.rr_csv)
        document = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            with args.output.open("x", encoding="utf-8") as target:
                target.write(document + "\n")
        print(document)
    except (OSError, ValueError) as error:
        parser.exit(1, f"HRV analysis failed: {error}\n")


if __name__ == "__main__":
    main()
