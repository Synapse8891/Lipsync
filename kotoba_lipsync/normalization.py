"""Causal microphone/speaker spectral normalization; no future frames used."""
import numpy as np
from .temporal import expand_features


class CausalSpectralMean:
    def __init__(self,alpha=.03,mode="cmn-v1"):self.alpha=alpha;self.mean=None;self.count=0;self.mode=mode
    def push(self,base):
        mel=np.asarray(base[:40],dtype=np.float32)
        if base[68]*80>-70 and base[75]>.4:
            self.count+=1
            if self.mean is None:self.mean=mel.copy()
            else:
                alpha=1/min(self.count,300) if self.mode=="cmn-v2" else self.alpha
                self.mean+=alpha*(mel-self.mean)
        return np.zeros(40,dtype=np.float32) if self.mean is None else mel-self.mean


def spectral_residuals(base,mode="cmn-v1"):
    normalizer=CausalSpectralMean(mode=mode)
    return np.asarray([normalizer.push(row) for row in base],dtype=np.float32)


def sequence_features(base,mode="base"):
    expanded=expand_features(base)
    return np.c_[expanded,spectral_residuals(base,mode)] if mode in ("cmn-v1","cmn-v2") else expanded
