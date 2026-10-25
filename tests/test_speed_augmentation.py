import numpy as np
from scripts.augment_speed_training import stretch_labels
from kotoba_lipsync.features import FeatureConfig


def test_waveform_speed_labels_keep_timing_and_quiet_gate():
    cfg=FeatureConfig();original=np.zeros(100,dtype=np.int16);original[20:70]=10
    times=np.maximum(0,(np.arange(120)+1)*.01-.0125)
    for speed in (.9,1.1):
        labels=stretch_labels(original,times,speed,np.full(len(times),-20),cfg)
        active=times[labels==10]
        assert abs(active[0]-.1975/speed)<.012
        assert abs(active[-1]-.6875/speed)<.012
        quiet=stretch_labels(original,times,speed,np.full(len(times),-60),cfg)
        assert not quiet.any()
