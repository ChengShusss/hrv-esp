"""RR spectral estimates: natural cubic resampling and normalized Welch PSD.

No third-party numerical runtime is required. Band powers are integrals of
one-sided density in ms^2/Hz, not FFT amplitudes or frequencies of the raw ECG.
"""

import bisect
import cmath
import math
import statistics

MIN_FREQUENCY_DURATION_S = 300.0
RESAMPLE_HZ = 4.0
WELCH_SAMPLES = 1024  # 256 s, >= 10 cycles at the 0.04 Hz LF band edge.


def _fft(values):
    """Radix-2 forward DFT, unnormalized."""
    result = [complex(value) for value in values]
    n = len(result)
    if not n or n & (n - 1):
        raise ValueError("FFT length must be a positive power of two")
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            result[i], result[j] = result[j], result[i]
    width = 2
    while width <= n:
        rotation = cmath.exp(-2j * math.pi / width)
        half = width // 2
        for first in range(0, n, width):
            factor = 1 + 0j
            for offset in range(half):
                left = result[first + offset]
                right = factor * result[first + offset + half]
                result[first + offset] = left + right
                result[first + offset + half] = left - right
                factor *= rotation
        width *= 2
    return result


def _cubic_resample(times, values, rate=RESAMPLE_HZ):
    """Natural cubic spline on observed support only; no extrapolation."""
    n = len(values)
    if n < 2 or len(times) != n or rate <= 0:
        raise ValueError("At least two matching samples and a positive rate are required")
    h = [b - a for a, b in zip(times, times[1:])]
    if any(step <= 0 for step in h):
        raise ValueError("Sample times must strictly increase")
    upper, rhs, second = [0.0] * n, [0.0] * n, [0.0] * n
    for i in range(1, n - 1):
        diagonal = 2 * (h[i - 1] + h[i]) - h[i - 1] * upper[i - 1]
        upper[i] = h[i] / diagonal
        change = 6 * ((values[i + 1] - values[i]) / h[i]
                      - (values[i] - values[i - 1]) / h[i - 1])
        rhs[i] = (change - h[i - 1] * rhs[i - 1]) / diagonal
    for i in range(n - 2, 0, -1):
        second[i] = rhs[i] - upper[i] * second[i + 1]
    output = []
    count = math.floor((times[-1] - times[0]) * rate + 1e-9) + 1
    interval = 0
    for k in range(count):
        stamp = min(times[-1], times[0] + k / rate)
        while interval < n - 2 and stamp > times[interval + 1]:
            interval += 1
        width = h[interval]
        left = (times[interval + 1] - stamp) / width
        right = 1 - left
        output.append(left * values[interval] + right * values[interval + 1]
                      + ((left**3 - left) * second[interval]
                         + (right**3 - right) * second[interval + 1]) * width**2 / 6)
    return output


def _welch_density(values, rate=RESAMPLE_HZ, window_samples=WELCH_SAMPLES):
    n = min(window_samples, len(values))
    if n < 4:
        raise ValueError("Too few samples for spectral estimation")
    nfft = 1 << (n - 1).bit_length()
    step = max(1, n // 2)
    starts = list(range(0, len(values) - n + 1, step))
    tail_added = starts[-1] != len(values) - n
    if tail_added:
        starts.append(len(values) - n)  # Include the observed tail, never pad missing time.
    window = [0.5 - 0.5 * math.cos(2 * math.pi * i / n) for i in range(n)]
    scale = rate * sum(weight * weight for weight in window)
    center = (n - 1) / 2
    x_variance = sum((i - center)**2 for i in range(n))
    density = [0.0] * (nfft // 2 + 1)
    for first in starts:
        segment = values[first:first + n]
        mean = statistics.mean(segment)
        slope = sum((i - center) * (value - mean) for i, value in enumerate(segment)) / x_variance
        transformed = _fft([(value - mean - slope * (i - center)) * window[i]
                            for i, value in enumerate(segment)] + [0.0] * (nfft - n))
        for k in range(len(density)):
            factor = 1 if k in (0, nfft // 2) else 2
            density[k] += factor * abs(transformed[k])**2 / scale / len(starts)
    frequencies = [k * rate / nfft for k in range(len(density))]
    return frequencies, density, {"window_samples": n, "window_s": n / rate,
                                   "nfft": nfft, "window_count": len(starts),
                                   "nominal_overlap_percent": 50, "tail_window_added": tail_added,
                                   "detrend": "linear per window", "window": "periodic Hann"}


def _band_power(frequencies, density, low, high):
    """Integrate piecewise-linear PSD with interpolated exact band boundaries."""
    power = 0.0
    first = max(0, bisect.bisect_right(frequencies, low) - 1)
    for i in range(first, len(frequencies) - 1):
        f0, f1 = frequencies[i:i + 2]
        left, right = max(low, f0), min(high, f1)
        if left >= high:
            break
        if right <= left:
            continue
        slope = (density[i + 1] - density[i]) / (f1 - f0)
        a = density[i] + slope * (left - f0)
        b = density[i] + slope * (right - f0)
        power += (a + b) * (right - left) / 2
    return max(0.0, power)


def rr_quality(intervals):
    """Review flags only. No physiological diagnosis or automatic correction."""
    flagged = []
    for i, value in enumerate(intervals):
        local = statistics.median(intervals[max(0, i - 5):i + 6])
        if value < 250 or value > 2000 or abs(value - local) > 0.30 * local:
            flagged.append(i)
    return {"suspect_rr_count": len(flagged), "suspect_rr_indices": flagged,
            "rule": "outside 250..2000 ms or >30% from local 11-interval median; review only",
            "normal_beats_classified": False}


def frequency_domain(intervals, quality=None):
    rr = [float(value) for value in intervals]
    if any(not math.isfinite(value) or value <= 0 for value in rr):
        raise ValueError("RR values must be finite and positive")
    support = sum(rr[1:]) / 1000 if len(rr) >= 2 else 0.0
    result = {"status": "insufficient_duration", "supported_duration_s": support,
              "required_duration_s": MIN_FREQUENCY_DURATION_S,
              "lf_ms2": None, "hf_ms2": None, "lf_hf_ratio": None,
              "lf_nu_percent": None, "hf_nu_percent": None,
              "lf_hf_sum_ms2": None, "frequencies_hz": [], "psd_ms2_per_hz": [],
              "bands_hz": {"lf": [0.04, 0.15], "hf": [0.15, 0.4]},
              "method": "natural cubic spline at 4 Hz; one-sided Welch density",
              "resample_hz": RESAMPLE_HZ}
    quality = rr_quality(rr) if quality is None else quality
    if quality["suspect_rr_count"]:
        result["status"] = "rr_review_required"
        return result
    if support + 1e-9 < MIN_FREQUENCY_DURATION_S:
        return result
    times, stamp = [], 0.0
    for interval in rr:
        stamp += interval / 1000
        times.append(stamp)
    uniform = _cubic_resample(times, rr)
    frequencies, density, parameters = _welch_density(uniform)
    lf = _band_power(frequencies, density, 0.04, 0.15)
    hf = _band_power(frequencies, density, 0.15, 0.4)
    total = lf + hf
    result.update(status="available_rr_estimate", lf_ms2=lf, hf_ms2=hf,
                  lf_hf_ratio=lf / hf if hf > 1e-9 else None,
                  lf_nu_percent=100 * lf / total if total > 1e-9 else None,
                  hf_nu_percent=100 * hf / total if total > 1e-9 else None,
                  lf_hf_sum_ms2=total, frequencies_hz=frequencies,
                  psd_ms2_per_hz=density, parameters=parameters,
                  uniform_samples=len(uniform))
    return result
