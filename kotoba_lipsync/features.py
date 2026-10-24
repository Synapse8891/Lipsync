"""25 ms analysis windows, 10 ms hop; optional bounded lookahead.

LPC-derived formants are acoustic estimates, not an anatomical nasal sensor.
"""
from collections import deque
from dataclasses import dataclass, asdict
from math import gcd
import numpy as np
import soundfile as sf
from scipy import signal
from scipy.fft import dct
from scipy.linalg import solve_toeplitz

@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    frame_size: int = 400
    hop_size: int = 160
    n_fft: int = 512
    mel_bins: int = 40
    lpc_order: int = 12
    context: tuple = (0, 1, 2, 4, 8, 12)
    lookahead_frames: int = 0

    def __post_init__(self):
        if not self.context or min(self.context)<0 or not 0<=self.lookahead_frames<=max(self.context):
            raise ValueError("Lookahead must be between 0 and the maximum nonnegative context lag")

    def to_dict(self):
        return asdict(self)

def read_audio(path, target_rate=16000):
    audio, rate = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != target_rate:
        div = gcd(int(rate), int(target_rate))
        audio = signal.resample_poly(audio, target_rate // div, rate // div).astype(np.float32)
    return np.nan_to_num(audio).astype(np.float32)

def mel_filterbank(config):
    hz_to_mel = lambda x: 2595 * np.log10(1 + x / 700)
    edges = 700 * (10 ** (np.linspace(hz_to_mel(60), hz_to_mel(7600), config.mel_bins + 2) / 2595) - 1)
    freq = np.fft.rfftfreq(config.n_fft, 1 / config.sample_rate)
    out = np.zeros((config.mel_bins, len(freq)), dtype=np.float32)
    for i in range(config.mel_bins):
        out[i] = np.maximum(0, np.minimum((freq - edges[i]) / (edges[i+1] - edges[i]), (edges[i+2] - freq) / (edges[i+2] - edges[i+1])))
        out[i] /= max(float(out[i].sum()), 1e-8)
    return out

class FeatureExtractor:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        self.window = np.hanning(self.config.frame_size).astype(np.float32)
        self.mel = mel_filterbank(self.config)
        self.freq = np.fft.rfftfreq(self.config.n_fft, 1 / self.config.sample_rate)
        self.reset()

    def reset(self):
        self.previous_spectrum = np.zeros(self.config.n_fft // 2 + 1, dtype=np.float32)
        self.previous_rms = 0.0

    def frame(self, samples):
        c = self.config
        raw = np.asarray(samples, dtype=np.float32)
        if raw.shape != (c.frame_size,):
            raise ValueError(f"Expected {c.frame_size} mono samples")
        rms = float(np.sqrt(np.mean(raw * raw) + 1e-12))
        db = 20 * np.log10(rms + 1e-8)
        centered = raw - raw.mean()
        emphasized = signal.lfilter([1, -.97], [1], centered).astype(np.float32)
        win = emphasized * self.window
        power = (np.abs(np.fft.rfft(win, c.n_fft)) ** 2).astype(np.float32) + 1e-12
        logs = np.log(self.mel @ power + 1e-10)
        cepstra = dct(logs, norm="ortho")[1:14]
        normalized = power / power.sum()
        flux = float(np.maximum(normalized - self.previous_spectrum, 0).sum())
        self.previous_spectrum = normalized
        zcr = float(np.mean(raw[1:] * raw[:-1] < 0))
        flatness = float(np.exp(np.mean(np.log(power))) / np.mean(power))
        centroid = float(normalized @ self.freq) / 8000
        low = float(power[self.freq < 450].sum() / power.sum())
        high = float(power[self.freq > 3000].sum() / power.sum())
        # Autocorrelation voicing, independent of the pre-emphasized LPC signal.
        ac = signal.correlate(centered * self.window, centered * self.window, mode="full", method="fft")[c.frame_size-1:]
        lag = 40 + int(np.argmax(ac[40:267]))
        periodicity = float(np.clip(ac[lag] / (ac[0] + 1e-10), 0, 1))
        pitch = c.sample_rate / lag / 500 if periodicity > .25 else 0.0
        ac_lpc = signal.correlate(win, win, mode="full", method="fft")[c.frame_size-1:c.frame_size+c.lpc_order]
        ac_lpc[0] += max(float(ac_lpc[0]) * 1e-5, 1e-8)
        try:
            coeff = solve_toeplitz(ac_lpc[:-1], -ac_lpc[1:], check_finite=False).astype(np.float32)
            roots = np.roots(np.r_[1, coeff])
            frequencies = np.angle(roots) * c.sample_rate / (2 * np.pi)
            bandwidth = -np.log(np.maximum(np.abs(roots), 1e-10)) * c.sample_rate / np.pi
            valid = np.sort(frequencies[(frequencies > 180) & (frequencies < 5000) & (bandwidth > 0) & (bandwidth < 900)])
            formants = np.zeros(3, dtype=np.float32)
            formants[:min(3, len(valid))] = valid[:3]
        except (np.linalg.LinAlgError, ValueError):
            coeff = np.zeros(c.lpc_order, dtype=np.float32)
            formants = np.zeros(3, dtype=np.float32)
        dynamics = np.clip(db - 20 * np.log10(self.previous_rms + 1e-8), -40, 40) / 40
        self.previous_rms = rms
        extra = np.array([db / 80, zcr, flatness, centroid, flux, low, high, periodicity, pitch, dynamics], dtype=np.float32)
        feat = np.r_[logs, cepstra, np.clip(coeff, -8, 8), formants / 5000, extra].astype(np.float32)
        diag = {"rms_db": float(db), "formants_hz": formants.tolist(), "periodicity": periodicity, "spectral_flatness": flatness, "spectral_flux": flux}
        return np.nan_to_num(feat), diag

class StreamingFeatures:
    def __init__(self, config=None):
        self.config = config or FeatureConfig()
        self.extractor = FeatureExtractor(self.config)
        self.pending = np.zeros(self.config.frame_size - self.config.hop_size, dtype=np.float32)
        self.history = deque(maxlen=max(self.config.context) + 1)
        self.diagnostics = deque(maxlen=self.config.lookahead_frames+1)
        self.frames = 0

    def push(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("Audio must be a finite mono float array")
        self.pending = np.concatenate((self.pending, samples))
        result = []
        c = self.config
        pos = 0
        while len(self.pending) - pos >= c.frame_size:
            feat, diag = self.extractor.frame(self.pending[pos:pos+c.frame_size])
            self.diagnostics.append(diag)
            if not self.history:
                self.history.extend([feat] * (max(c.context) + 1))
            else:
                self.history.append(feat)
            stacked = np.concatenate([self.history[-1-delay] for delay in c.context])
            # Frame timestamp is the center of its causal analysis window.
            center = ((self.frames + 1-c.lookahead_frames) * c.hop_size - c.frame_size / 2) / c.sample_rate
            if self.frames >= c.lookahead_frames:
                result.append((stacked, max(0.0, center), self.diagnostics[0]))
            pos += c.hop_size
            self.frames += 1
        self.pending = self.pending[pos:]
        return result

def extract_audio(samples, config=None):
    stream = StreamingFeatures(config)
    frames = stream.push(samples)
    if not frames:
        raise ValueError("Audio must be at least 10 milliseconds long")
    return (np.stack([f[0] for f in frames]), np.array([f[1] for f in frames]), [f[2] for f in frames])
