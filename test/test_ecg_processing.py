"""Behavioral checks with known beat times, noise, gaps and polarity changes."""

import math
from pathlib import Path
import random
import statistics
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from ecg_processing import Biquad, EcgProcessor
from ecg_protocol import Frame


BEATS = (0.6, 1.5, 2.35, 3.15, 4.05, 5.02, 5.85, 6.75, 7.65, 8.5, 9.4)


def synthetic(rate=1000, duration=10.0, polarity=1, beats=BEATS, noise=True):
    generator = random.Random(7)
    result = []
    for index in range(round(duration * rate)):
        stamp = index / rate
        value = 2048.0
        if noise:
            value += 160 * math.sin(2 * math.pi * 0.25 * stamp)
            value += 70 * math.sin(2 * math.pi * 50 * stamp) + generator.gauss(0, 4)
        for beat in beats:
            delta = stamp - beat
            if -0.3 < delta < 0.5:
                # S is larger than R: an absolute-amplitude picker would mark S.
                value += polarity * (700 * math.exp(-(delta / 0.012) ** 2)
                    - 950 * math.exp(-((delta - 0.035) / 0.014) ** 2)
                    - 80 * math.exp(-((delta + 0.025) / 0.01) ** 2)
                    + 330 * math.exp(-((delta - 0.25) / 0.08) ** 2))
        result.append(round(value))
    return result


def feed(processor, values, first=0, rate=1000, size=32, leads=0, epoch=0, session=123):
    peaks = []
    for offset in range(0, len(values), size):
        peaks.extend(processor.process_frame(Frame(session, epoch, first + offset, rate,
                     leads, tuple(values[offset:offset + size]))))
    return peaks


class ProcessingTests(unittest.TestCase):
    def test_r_times_and_rr_with_mains_noise_drift_and_large_s(self):
        peaks = feed(EcgProcessor(), synthetic())
        expected = [round(t * 1000) for t in BEATS if t > 2.1]
        self.assertEqual(len(peaks), len(expected))
        for peak, index in zip(peaks, expected):
            self.assertLessEqual(abs(peak.sample_index - index), 3)
            self.assertGreater(peak.reported_at_sample, peak.sample_index)
        self.assertIsNone(peaks[0].rr_ms)
        for left, right, previous, current in zip(expected, expected[1:], peaks, peaks[1:]):
            self.assertEqual(current.rr_ms, current.sample_index - previous.sample_index)
            self.assertLessEqual(abs(current.rr_ms - (right - left)), 4)

    def test_usb_chunk_sizes_do_not_change_detection(self):
        values = synthetic()
        reference = feed(EcgProcessor(), values, size=32)
        for size in (1, 7, 17):
            self.assertEqual(feed(EcgProcessor(), values, size=size), reference)

    def test_inverted_ecg_and_other_sample_rate(self):
        for rate in (500, 1000):
            peaks = feed(EcgProcessor(polarity=-1), synthetic(rate=rate, polarity=-1), rate=rate)
            expected = [round(t * rate) for t in BEATS if t > 2.1]
            self.assertEqual(len(peaks), len(expected))
            self.assertTrue(all(abs(p.sample_index - t) <= round(0.004 * rate)
                                for p, t in zip(peaks, expected)))

    def test_flatline_and_mains_alone_do_not_create_beats(self):
        for values in ([2048] * 8000, synthetic(duration=8, beats=())):
            self.assertEqual(feed(EcgProcessor(), values), [])

    def test_strong_mains_with_frequency_drift_and_phase_changes(self):
        # Known R times, independent of the filter: 300-count mains is close to
        # the interference seen in the real capture, including near-50 Hz drift.
        for rate in (500, 1000):
            for polarity in (-1, 1):
                clean = synthetic(rate=rate, polarity=polarity, noise=False)
                expected = [round(t * rate) for t in BEATS if t > 2.1]
                for frequency in (49.5, 50, 50.5):
                    for phase in (0, .5 * math.pi, math.pi, 1.5 * math.pi):
                        with self.subTest(rate=rate, polarity=polarity, frequency=frequency, phase=phase):
                            values = [round(value + 300 * math.sin(2 * math.pi * frequency * i / rate + phase)
                                            + 160 * math.sin(2 * math.pi * .25 * i / rate))
                                      for i, value in enumerate(clean)]
                            peaks = feed(EcgProcessor(polarity), values, rate=rate)
                            self.assertEqual(len(peaks), len(expected))
                            for peak, truth in zip(peaks, expected):
                                self.assertLessEqual(abs(peak.sample_index - truth) * 1000 / rate, 2)
                            for old, current in zip(peaks, peaks[1:]):
                                self.assertEqual(current.rr_ms,
                                                 1000 * (current.sample_index - old.sample_index) / rate)

    def test_gap_lead_off_clipping_and_reboot_break_rr(self):
        for failure in ("gap", "leads", "clipping", "epoch", "session"):
            with self.subTest(failure=failure):
                processor = EcgProcessor()
                before = feed(processor, synthetic(duration=5))
                self.assertTrue(before)
                first, epoch, session = 5000, 0, 123
                if failure == "gap":
                    first += 100
                elif failure == "leads":
                    feed(processor, [2000] * 64, first=first, leads=1)
                    first += 64
                elif failure == "clipping":
                    feed(processor, [4095] * 64, first=first)
                    first += 64
                elif failure == "epoch":
                    first, epoch = 0, 1
                else:
                    first, session = 0, 456
                after = feed(processor, synthetic(duration=5), first=first, epoch=epoch, session=session)
                self.assertTrue(after)
                self.assertIsNone(after[0].rr_ms)
                self.assertNotEqual(before[-1].segment, after[0].segment)

    def test_duplicate_packet_is_ignored(self):
        processor = EcgProcessor()
        values = synthetic(duration=5)
        before = feed(processor, values)
        self.assertEqual(feed(processor, values), [])
        self.assertEqual(processor.detected_count, len(before))

    def test_polarity_change_breaks_rr(self):
        processor = EcgProcessor()
        feed(processor, synthetic(duration=5))
        processor.set_polarity(-1)
        self.assertIsNone(processor.last_peak)
        after = feed(processor, synthetic(duration=5, polarity=-1), first=5000)
        self.assertTrue(after)
        self.assertIsNone(after[0].rr_ms)

    def test_transport_pause_breaks_rr_even_with_contiguous_indices(self):
        processor = EcgProcessor()
        feed(processor, synthetic(duration=5))
        processor.pause()
        after = feed(processor, synthetic(duration=5), first=5000)
        self.assertTrue(after)
        self.assertIsNone(after[0].rr_ms)

    def test_filter_removes_dc_and_rejects_50_hz(self):
        highpass = Biquad("highpass", 0.5, 1000)
        self.assertLess(max(abs(highpass.step(2200)) for _ in range(3000)), 1e-5)
        rms = {}
        for frequency in (10, 50):
            notch = Biquad("notch", 50, 1000, q=30)
            values = [notch.step(math.sin(2 * math.pi * frequency * i / 1000)) for i in range(6000)]
            rms[frequency] = math.sqrt(statistics.mean(v * v for v in values[3000:]))
        self.assertGreater(rms[10], 0.69)
        self.assertLess(rms[50], 0.001)

    def test_unsupported_rate_disables_detection(self):
        processor = EcgProcessor()
        self.assertEqual(feed(processor, [2048] * 1000, rate=100), [])
        self.assertFalse(processor.active)


if __name__ == "__main__":
    unittest.main()
