"""Causal microphone/speaker spectral normalization; no future frames used."""
import numpy as np
from .temporal import expand_features


class CausalSpectralMean:
    def __init__(self,alpha=.03):self.alpha=alpha;self.mean=None
    def push(self,base):
        mel=np.asarray(base[:40],dtype=np.float32)
        if base[68]*80>-70 and base[75]>.4:
            if self.mean is None:self.mean=mel.copy()
            else:self.mean+=self.alpha*(mel-self.mean)
        return np.zeros(40,dtype=np.float32) if self.mean is None else mel-self.mean


def spectral_residuals(base):
    normalizer=CausalSpectralMean()
    return np.asarray([normalizer.push(row) for row in base],dtype=np.float32)


def sequence_features(base,mode="base"):
    expanded=expand_features(base)
    return np.c_[expanded,spectral_residuals(base)] if mode=="cmn-v1" else expanded
