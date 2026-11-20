"""Behavior checks for serial corruption, gaps and device restarts."""

from pathlib import Path
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "host"))
from ecg_protocol import Frame, LineDecoder, Waveform, decode_line, encode_frame


def make_frame(first=0, epoch=0, session=0x1234ABCD, samples=(1000, 2000, 4095)):
    return Frame(session, epoch, first, 1000, 0, samples)


class ProtocolTests(unittest.TestCase):
    def test_fragmentation_and_multiple_packets(self):
        expected = [make_frame(), make_frame(3)]
        wire = b"ESP-ROM boot log\r\n" + b"".join(map(encode_frame, expected))
        for chunk_size in (1, 7, 64, len(wire)):
            decoder = LineDecoder()
            actual = []
            for offset in range(0, len(wire), chunk_size):
                actual.extend(decoder.feed(wire[offset:offset + chunk_size]))
            self.assertEqual(actual, expected)
            self.assertEqual(decoder.invalid_packets, 0)

    def test_corruption_and_incomplete_write_recover(self):
        valid = encode_frame(make_frame())
        corrupted = valid.replace(b",2000,", b",2001,")
        decoder = LineDecoder()
        self.assertEqual(decoder.feed(corrupted + valid[:25] + valid), [make_frame()])
        self.assertEqual(decoder.invalid_packets, 2)

    def test_field_validation_even_with_correct_crc(self):
        invalid_bodies = (
            b"ECG1,1234ABCD,0,0,1000,0,1,4096",
            b"ECG1,1234ABCD,0,0,1000,0,2,100",
            b"ECG1,1234ABCD,0,-1,1000,0,1,100",
            b"ECG1,1234ABCD,0,0,0,0,1,100",
            b"ECG1,1234ABCD,0,0,1000,4,1,100",
            b"ECG1,1234ABCD,-1,0,1000,0,1,100",
        )
        for body in invalid_bodies:
            with self.subTest(body=body), self.assertRaises(ValueError):
                decode_line(body + f",{zlib.crc32(body):08X}".encode())

    def test_unterminated_garbage_has_bounded_memory(self):
        decoder = LineDecoder()
        decoder.feed(b"x" * 100000)
        self.assertLessEqual(len(decoder.buffer), 384)
        self.assertEqual(decoder.feed(encode_frame(make_frame())), [make_frame()])

    def test_all_adc_codes_and_largest_packet(self):
        for first in range(0, 4096, 32):
            frame = make_frame(first=first, samples=tuple(range(first, first + 32)))
            self.assertEqual(decode_line(encode_frame(frame).strip()), frame)
        frame = Frame(0xFFFFFFFF, 0xFFFFFFFF, 2**64 - 33, 1000, 3, (4095,) * 32)
        self.assertLess(len(encode_frame(frame)), 384)
        self.assertEqual(decode_line(encode_frame(frame).strip()), frame)

    def test_gap_is_visible_and_time_uses_sample_clock(self):
        waveform = Waveform()
        waveform.accept(make_frame(first=100))
        waveform.accept(make_frame(first=108))
        self.assertEqual(waveform.missing_samples, 5)
        self.assertEqual(list(waveform.points)[3], (0.108, None))
        self.assertEqual(waveform.points[-1], (0.110, 4095))
        self.assertFalse(waveform.accept(make_frame(first=108)))
        self.assertEqual(waveform.received_samples, 6)

    def test_device_reboot_and_adc_restart_start_new_timeline(self):
        waveform = Waveform()
        waveform.accept(make_frame(first=10000))
        waveform.accept(make_frame(epoch=1))
        self.assertEqual(waveform.restarts, 1)
        self.assertEqual(len(waveform.points), 3)
        self.assertEqual(waveform.points[0][0], 0)
        waveform.accept(make_frame(session=0x5678))
        self.assertEqual(waveform.restarts, 2)
        self.assertEqual(waveform.missing_samples, 0)

    def test_waveform_memory_is_bounded(self):
        waveform = Waveform(max_samples=5)
        for first in range(0, 30, 3):
            waveform.accept(make_frame(first))
        self.assertEqual(len(waveform.points), 5)
        self.assertEqual(waveform.received_samples, 30)


if __name__ == "__main__":
    unittest.main()
