"""Int8 weight storage, float32 CPU GEMV; no Torch/ONNX needed at runtime."""
import json
from pathlib import Path
import numpy as np
from .features import FeatureConfig

def quantize_weight(weight):
    scale = np.maximum(np.max(np.abs(weight), axis=0) / 127, 1e-8).astype(np.float32)
    q = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
    return q, scale

def export_model(path, weights, biases, mean, scale, config, metadata):
    arrays = {"mean": np.asarray(mean, dtype=np.float32), "scale": np.asarray(scale, dtype=np.float32),
              "metadata": np.array(json.dumps({**metadata, "feature_config": config.to_dict(), "layers": len(weights), "format": "kotoba-int8-v1"}))}
    for i, (w, b) in enumerate(zip(weights, biases)):
        arrays[f"w{i}"], arrays[f"s{i}"] = quantize_weight(w)
        arrays[f"b{i}"] = np.asarray(b, dtype=np.float32)
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(path, **arrays)

class TinyModel:
    stateful = False
    has_breath_head = False
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.metadata = json.loads(str(data["metadata"]))
            if self.metadata.get("format") != "kotoba-int8-v1":
                raise ValueError("Unsupported model format")
            cfg = self.metadata["feature_config"]
            cfg["context"] = tuple(cfg["context"])
            self.config = FeatureConfig(**cfg)
            self.mean = data["mean"].copy()
            self.scale = data["scale"].copy()
            self.weights = [data[f"w{i}"].astype(np.float32) * data[f"s{i}"] for i in range(self.metadata["layers"])]
            self.biases = [data[f"b{i}"].copy() for i in range(self.metadata["layers"])]
        self.classes = np.array(self.metadata["classes"], dtype=np.int32)
        self.parameters = sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        self.float_weights_bytes = sum(w.nbytes + b.nbytes for w, b in zip(self.weights, self.biases))

    def predict_proba(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.shape[-1] != len(self.mean):
            raise ValueError("Feature width does not match model")
        x = (x - self.mean) / self.scale
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            if i + 1 < len(self.weights):
                x = np.maximum(x, 0)
        x = x - np.max(x, axis=-1, keepdims=True)
        p = np.exp(x)
        p /= p.sum(axis=-1, keepdims=True)
        full = np.zeros((*p.shape[:-1], 16), dtype=np.float32)
        full[..., self.classes] = p
        return full


def load_model(path):
    with np.load(path, allow_pickle=False) as data:
        model_format = json.loads(str(data["metadata"])).get("format")
    if model_format == "kotoba-int8-v1":
        return TinyModel(path)
    if model_format == "kotoba-tcn-int8-v1":
        from .temporal import TemporalModel
        return TemporalModel(path)
    raise ValueError(f"Unsupported model format: {model_format}")
