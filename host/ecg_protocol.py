"""Validated ECG1 serial packets and a sample-clock-based waveform buffer."""

from collections import deque
from dataclasses import dataclass
import zlib

MAX_LINE_BYTES = 384
MAX_FRAME_SAMPLES = 32


@dataclass(frozen=True)
class Frame:
    session: int
    epoch: int
    first_sample: int
    sample_rate: int
    leads_off: int
    samples: tuple[int, ...]


def encode_frame(frame: Frame) -> bytes:
    """Used by the demo and protocol checks; mirrors the firmware wire format."""
    fields = ["ECG1", f"{frame.session:08X}", str(frame.epoch),
              str(frame.first_sample), str(frame.sample_rate), str(frame.leads_off),
              str(len(frame.samples)), *(str(value) for value in frame.samples)]
    body = ",".join(fields).encode("ascii")
    return b"\n" + body + f",{zlib.crc32(body):08X}\n".encode("ascii")


def decode_line(line: bytes) -> Frame:
    line = line.rstrip(b"\r\n")
    if len(line) > MAX_LINE_BYTES:
        raise ValueError("packet too long")
    body, separator, checksum = line.rpartition(b",")
    if not separator or len(checksum) != 8 or zlib.crc32(body) != int(checksum, 16):
        raise ValueError("invalid CRC")
    fields = body.decode("ascii").split(",")
    if len(fields) < 8 or fields[0] != "ECG1" or len(fields[1]) != 8:
        raise ValueError("invalid header")
    session = int(fields[1], 16)
    epoch, first, rate, leads, count = map(int, fields[2:7])
    samples = tuple(map(int, fields[7:]))
    if not (0 <= session <= 0xFFFFFFFF and 0 <= epoch <= 0xFFFFFFFF
            and 0 <= first <= 0xFFFFFFFFFFFFFFFF - count
            and 1 <= rate <= 10000 and 0 <= leads <= 3
            and 1 <= count <= MAX_FRAME_SAMPLES and count == len(samples)
            and all(0 <= sample <= 4095 for sample in samples)):
        raise ValueError("invalid field range or sample count")
    return Frame(session, epoch, first, rate, leads, samples)


class LineDecoder:
    """Handles arbitrary USB chunk boundaries, boot logs and truncated packets."""

    def __init__(self):
        self.buffer = bytearray()
        self.discard_until_newline = False
        self.invalid_packets = 0

    def feed(self, data: bytes) -> list[Frame]:
        self.buffer.extend(data)
        lines = self.buffer.split(b"\n")
        self.buffer = bytearray(lines.pop())
        frames = []
        for line in lines:
            if self.discard_until_newline:
                self.discard_until_newline = False
                continue
            if not line.startswith(b"ECG1,"):
                continue
            try:
                frames.append(decode_line(line))
            except (ValueError, UnicodeError):
                self.invalid_packets += 1
        if len(self.buffer) > MAX_LINE_BYTES:
            self.buffer.clear()
            self.discard_until_newline = True
            self.invalid_packets += 1
        return frames


class Waveform:
    def __init__(self, max_samples: int = 10000):
        self.points = deque(maxlen=max_samples)
        self.stream = None
        self.next_sample = 0
        self.missing_samples = 0
        self.restarts = 0
        self.received_samples = 0
        self.last_frame = None

    def accept(self, frame: Frame) -> bool:
        stream = (frame.session, frame.epoch, frame.sample_rate)
        if stream != self.stream:
            if self.stream is not None:
                self.restarts += 1
            self.points.clear()
            self.stream = stream
            self.next_sample = frame.first_sample
        if frame.first_sample < self.next_sample:
            return False  # Duplicate/stale delivery is not another measurement.
        if frame.first_sample > self.next_sample:
            self.missing_samples += frame.first_sample - self.next_sample
            self.points.append((frame.first_sample / frame.sample_rate, None))
        self.points.extend(((frame.first_sample + index) / frame.sample_rate, value)
                           for index, value in enumerate(frame.samples))
        self.next_sample = frame.first_sample + len(frame.samples)
        self.received_samples += len(frame.samples)
        self.last_frame = frame
        return True
