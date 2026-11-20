"""Streaming ECG filters and provisional R-peak/RR detection.

The QRS candidate detector borrows the bandpass / slope-energy / adaptive
threshold approach of Pan and Tompkins; this is not a full implementation of
their algorithm. R timestamps are refined on buffered samples with centered
50 Hz rejection and smoothing, not on the causal filter or USB arrival time.
"""

from collections import deque
from dataclasses import dataclass
import math
import statistics

from ecg_protocol import Frame

DETECTOR_VERSION = "ecg1-r2-centered50"


class Biquad:
    """RBJ high-pass, low-pass and notch sections, transposed direct form II."""

    def __init__(self, kind: str, frequency: float, rate: int, q: float = 2**-0.5):
        omega = 2 * math.pi * frequency / rate
        cosine = math.cos(omega)
        alpha = math.sin(omega) / (2 * q)
        if kind == "highpass":
            b0, b1, b2 = (1 + cosine) / 2, -(1 + cosine), (1 + cosine) / 2
        elif kind == "lowpass":
            b0, b1, b2 = (1 - cosine) / 2, 1 - cosine, (1 - cosine) / 2
        elif kind == "notch":
            b0, b1, b2 = 1, -2 * cosine, 1
        else:
            raise ValueError("unknown filter kind")
        a0 = 1 + alpha
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = -2 * cosine / a0, (1 - alpha) / a0
        self.z1 = self.z2 = 0.0
        self.initialized = False

    def step(self, value: float) -> float:
        if not self.initialized:
            # Initialize at DC equilibrium; a 2000-count ADC offset must not
            # look like a giant QRS complex when the stream is first connected.
            steady = value * (self.b0 + self.b1 + self.b2) / (1 + self.a1 + self.a2)
            self.z1 = steady - self.b0 * value
            self.z2 = self.b2 * value - self.a2 * steady
            self.initialized = True
        result = self.b0 * value + self.z1
        self.z1 = self.b1 * value - self.a1 * result + self.z2
        self.z2 = self.b2 * value - self.a2 * result
        return result


@dataclass(frozen=True)
class RPeak:
    session: int
    epoch: int
    segment: int
    sample_index: int
    sample_rate: int
    raw_adc: int
    rr_ms: float | None
    reported_at_sample: int

    @property
    def time_s(self) -> float:
        return self.sample_index / self.sample_rate


class EcgProcessor:
    """One ordered processing state per stream; gaps always break RR continuity."""

    def __init__(self, polarity: int = 1):
        if polarity not in (-1, 1):
            raise ValueError("polarity must be +1 or -1")
        self.polarity = polarity
        self.stream = None
        self.next_sample = None
        self.segment = 0
        self.filtered = deque(maxlen=10000)
        self.peaks = deque(maxlen=500)
        self.detected_count = 0
        self.last_peak = None
        self.status = "等待采样"
        self.active = False
        self.rate = 1000

    def set_polarity(self, polarity: int):
        if polarity not in (-1, 1):
            raise ValueError("polarity must be +1 or -1")
        if polarity != self.polarity:
            self.polarity = polarity
            self._break("R 峰方向变化，重新预热")
            self.peaks.clear()

    def pause(self, reason: str = "数据暂停，重新预热"):
        self._break(reason)

    def _break(self, reason: str):
        if self.active:
            self.segment += 1
            if self.filtered and self.filtered[-1][1] is not None:
                self.filtered.append((self.filtered[-1][0], None))
        self.active = False
        self.last_peak = None
        self.status = reason

    def _start(self):
        rate = self.rate
        self.display_filters = [Biquad("highpass", 0.5, rate),
                                Biquad("notch", 50, rate, q=30),
                                Biquad("lowpass", 40, rate)]
        self.qrs_filters = [Biquad("highpass", 5, rate), Biquad("lowpass", 20, rate)]
        self.derivatives = deque(maxlen=max(1, round(0.005 * rate)))
        self.energy_window = deque()
        self.energy_length = max(1, round(0.1 * rate))
        self.energy_sum = 0.0
        self.raw_buffer = deque(maxlen=round(0.8 * rate))
        self.bootstrap = []
        self.valid_samples = 0
        self.warmup_samples = round(2 * rate)
        self.signal_level = self.noise_level = 0.0
        self.pending = None
        self.below_samples = 0
        self.last_qrs_slope = 0.0
        self.active = True
        self.status = "滤波预热（2 秒）"

    def _threshold(self) -> float:
        return max(1.0, self.noise_level + 0.25 * (self.signal_level - self.noise_level))

    def _refine(self, center: int):
        raw = list(self.raw_buffer)
        half_width = max(1, round(0.004 * self.rate))
        # Symmetric FIR, centered on each original sample. At 1000 Hz this is
        # 0.25*x[n-10] + 0.5*x[n] + 0.25*x[n+10], with an exact zero at 50 Hz.
        # Candidate confirmation already buffers the required future samples;
        # the timestamp remains n, with no causal IIR delay added to RR.
        delay = max(1, round(self.rate / 100))
        side = 1 / (2 - 2 * math.cos(2 * math.pi * 50 * delay / self.rate))
        middle = 1 - 2 * side
        margin = half_width + delay
        start = center - round(0.16 * self.rate)
        end = center + round(0.03 * self.rate)
        candidates = []
        for position in range(margin, len(raw) - margin):
            index, value = raw[position]
            if start <= index <= end:
                smoothed = sum(side * (raw[j - delay][1] + raw[j + delay][1])
                               + middle * raw[j][1]
                               for j in range(position - half_width,
                                              position + half_width + 1)) / (2 * half_width + 1)
                candidates.append((self.polarity * smoothed, index, value))
        if not candidates:
            return None
        score, index, value = max(candidates)
        baseline = statistics.median(item[0] for item in candidates)
        # A fixed ADC-domain floor rejects flat input/quantization noise. It is
        # deliberately documented and is not a physiological validity test.
        if score - baseline < 40:
            return None
        return index, value

    def _finish_candidate(self, frame: Frame, reported_at: int):
        candidate = self.pending
        self.pending = None
        self.below_samples = 0
        if candidate is None:
            return None
        refined = self._refine(candidate["index"])
        if refined is None:
            self.noise_level = 0.125 * candidate["energy"] + 0.875 * self.noise_level
            return None
        index, value = refined
        rr_samples = index - self.last_peak.sample_index if self.last_peak else None
        if rr_samples is not None:
            if rr_samples < round(0.25 * self.rate):
                return None
            if rr_samples < round(0.36 * self.rate) and candidate["slope"] < self.last_qrs_slope * 0.5:
                self.noise_level = 0.125 * candidate["energy"] + 0.875 * self.noise_level
                return None
        peak = RPeak(frame.session, frame.epoch, self.segment, index, self.rate,
                     value, 1000 * rr_samples / self.rate if rr_samples is not None else None,
                     reported_at)
        self.last_peak = peak
        self.peaks.append(peak)
        self.detected_count += 1
        self.last_qrs_slope = candidate["slope"]
        self.signal_level = 0.125 * candidate["energy"] + 0.875 * self.signal_level
        return peak

    def process_frame(self, frame: Frame) -> list[RPeak]:
        stream = (frame.session, frame.epoch, frame.sample_rate)
        if stream != self.stream:
            self._break("新采样时间段")
            self.stream = stream
            self.rate = frame.sample_rate
            self.next_sample = frame.first_sample
            self.filtered = deque(maxlen=max(1000, 10 * self.rate))
            self.peaks.clear()
        if frame.first_sample < self.next_sample:
            return []
        if frame.first_sample != self.next_sample:
            self._break("采样缺口，重新预热")
        self.next_sample = frame.first_sample + len(frame.samples)
        if not 200 <= self.rate <= 5000:
            self._break("检测支持 200～5000 Hz 采样率")
            return []
        if frame.leads_off:
            self._break("电极脱落，暂停检测")
            return []

        found = []
        for offset, raw in enumerate(frame.samples):
            index = frame.first_sample + offset
            if raw <= 8 or raw >= 4087:
                self._break("信号触及量程边缘，暂停检测")
                continue
            if not self.active:
                self._start()
            self.raw_buffer.append((index, raw))
            filtered = float(raw)
            for section in self.display_filters:
                filtered = section.step(filtered)
            self.filtered.append((index / self.rate, filtered))
            qrs = filtered
            for section in self.qrs_filters:
                qrs = section.step(qrs)
            derivative = ((qrs - self.derivatives[0]) / len(self.derivatives)
                          if self.derivatives else 0.0)
            self.derivatives.append(qrs)
            energy = derivative * derivative
            self.energy_window.append(energy)
            self.energy_sum += energy
            if len(self.energy_window) > self.energy_length:
                self.energy_sum -= self.energy_window.popleft()
            envelope = self.energy_sum / len(self.energy_window)
            self.valid_samples += 1
            if self.valid_samples <= self.warmup_samples:
                self.bootstrap.append(envelope)
                if self.valid_samples == self.warmup_samples:
                    self.noise_level = statistics.median(self.bootstrap)
                    self.signal_level = max(max(self.bootstrap) * 0.6, self.noise_level * 2)
                    self.bootstrap.clear()
                    self.status = "检测中（RR 待核验）"
                continue

            threshold = self._threshold()
            if envelope >= threshold:
                self.below_samples = 0
                if self.pending is None:
                    self.pending = {"start": index, "index": index, "energy": envelope,
                                    "slope": abs(derivative)}
                else:
                    if envelope > self.pending["energy"]:
                        self.pending["energy"] = envelope
                        self.pending["index"] = index
                    self.pending["slope"] = max(self.pending["slope"], abs(derivative))
            elif self.pending is not None:
                self.below_samples += 1
            else:
                self.noise_level = 0.001 * envelope + 0.999 * self.noise_level

            if self.pending is not None and (
                    self.below_samples >= round(0.02 * self.rate)
                    or index - self.pending["start"] >= round(0.2 * self.rate)):
                peak = self._finish_candidate(frame, index)
                if peak is not None:
                    found.append(peak)
        return found
