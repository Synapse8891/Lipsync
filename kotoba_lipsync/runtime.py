"""Bounded-memory streaming inference with attack/release and a silence gate."""
from dataclasses import dataclass, asdict
import time
import numpy as np
from .features import StreamingFeatures
from .model import load_model
from .visemes import LABELS, VRM_MAP

@dataclass
class FrameResult:
    timestamp: float
    viseme: str
    viseme_id: int
    confidence: float
    weights: list
    vrm: list
    breath: float
    nasal: float
    rms_db: float
    formants_hz: list
    inference_ms: float
    sequence: int

    def to_dict(self):
        return asdict(self)

class LipSyncEngine:
    def __init__(self, model_path, silence_db=-55.0, smoothing_ms=None):
        self.model = load_model(model_path)
        self.features = StreamingFeatures(self.model.config, self.model.predict_proba if self.model.stateful else None)
        self.silence_db = float(silence_db)
        self.breath_silence_db = self.silence_db + self.model.metadata.get("breath_gate_offset_db",0.)
        if smoothing_ms is None:
            smoothing_ms = self.model.metadata.get("recommended_smoothing_ms", 25.0)
        self.smoothing_ms = max(0.0, float(smoothing_ms))
        self.state = np.zeros(16, dtype=np.float32)
        self.state[0] = 1
        self.sequence = 0
        self.breath_state = 0.0
        self.labels = self.model.metadata.get("labels", LABELS)

    def push(self, mono_samples):
        results = []
        for feature, timestamp, diag in self.features.push(mono_samples):
            started = time.perf_counter()
            probability = self.model.predict_proba(feature)
            if diag["rms_db"] < self.silence_db:
                target = np.zeros(16, dtype=np.float32)
                target[0] = 1
            else:
                target = probability
            # 10 ms frame updates. This is additional response smoothing, not lookahead.
            tau = self.smoothing_ms * (0.6 if target[1] > self.state[1] else 1.0)
            alpha = 1.0 if tau <= 0 else 1 - np.exp(-10 / tau)
            self.state += alpha * (target - self.state)
            self.state = np.clip(self.state, 0, 1)
            self.state /= max(float(self.state.sum()), 1e-9)
            idx = int(self.state.argmax())
            vrm = np.clip(self.state @ VRM_MAP, 0, 1)
            # /m/ also closes lips; this auxiliary channel is an articulation cue.
            nasal = float(self.state[8])
            if self.model.has_breath_head:
                breath_target = self.model.last_breath if diag["rms_db"] >= self.breath_silence_db else 0.0
                self.breath_state += alpha * (breath_target - self.breath_state)
                breath = self.breath_state
            else:
                breath = float(self.state[15])
            results.append(FrameResult(timestamp, self.labels[idx], min(idx, 14) if idx != 15 else 0,
                                       float(probability.max()), self.state[:15].tolist(), vrm.tolist(),
                                       breath, nasal, diag["rms_db"], diag["formants_hz"],
                                       (time.perf_counter()-started)*1000, self.sequence))
            self.sequence += 1
        return results

def neutral_result(sequence=0):
    return FrameResult(0,"sil",0,1,[1.]+[0.]*14,[0.]*5,0,0,-100,[0.]*3,0,sequence)
