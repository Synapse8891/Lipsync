import numpy as np
import pytest
from scripts.prepare_training import prepare_teacher_input,cache_is_current,PIPELINE_VERSION
from scripts.train import validate_pipeline

def test_japanese_teacher_preserves_raw_waveform_amplitude():
    audio=np.array([-.1,.03,.07,.2],dtype=np.float32)
    actual=prepare_teacher_input(audio,{"do_normalize":False,"sampling_rate":16000})
    np.testing.assert_array_equal(actual,audio)

def test_english_teacher_normalizes_when_required():
    audio=np.array([-.1,.03,.07,.2],dtype=np.float32)
    actual=prepare_teacher_input(audio,{"do_normalize":True,"sampling_rate":16000})
    assert abs(float(actual.mean()))<1e-6
    assert abs(float(actual.var())-1)<1e-4

def test_missing_preprocessor_config_is_rejected():
    with pytest.raises(ValueError):prepare_teacher_input(np.zeros(10),{})

def test_legacy_cache_is_not_reused(tmp_path):
    path=tmp_path/"old.npz"
    np.savez(path,base=np.zeros((2,78)),y=[0,0])
    assert not cache_is_current(path,"expected")
    np.savez(path,pipeline_version=PIPELINE_VERSION,annotation_signature="expected")
    assert cache_is_current(path,"expected")
    assert not cache_is_current(path,"different")

def test_training_rejects_legacy_annotation_index():
    with pytest.raises(RuntimeError,match="v2"):
        validate_pipeline({"records":[]})

def test_training_rejects_wrong_japanese_normalization():
    with pytest.raises(RuntimeError,match="正規化"):
        validate_pipeline({"pipeline_version":PIPELINE_VERSION,"teacher_preprocessing":{"ja":{"do_normalize":True},"en":{"do_normalize":True}}})
