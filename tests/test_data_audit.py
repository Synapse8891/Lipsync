import pytest


def test_event_audio_requires_individual_cc0_license():
    from scripts.temporal_data import breath_features
    with pytest.raises(ValueError,match="individually declare CC0"):
        breath_features({"path":"data/nonexistent.wav","audio_license":"CC-BY-4.0"})


def test_gain_augmentation_matches_scaled_waveform():
    import numpy as np
    from kotoba_lipsync.features import extract_audio,FeatureConfig
    from scripts.temporal_data import apply_gain
    audio=np.random.default_rng(17).normal(0,.03,1600).astype(np.float32)
    original=extract_audio(audio,FeatureConfig())[0][:,:78]
    expected=extract_audio(audio*10**(6/20),FeatureConfig())[0][:,:78]
    actual=apply_gain(original,6)
    np.testing.assert_allclose(actual[:,:53],expected[:,:53],atol=1e-5)
    np.testing.assert_allclose(actual[:,68],expected[:,68],atol=1e-5)


def test_breath_tail_preserves_original_labels_and_unmasks_delayed_event(tmp_path,monkeypatch):
    import hashlib
    import numpy as np
    import soundfile as sf
    import scripts.temporal_data as data
    monkeypatch.setattr(data,"ROOT",tmp_path);(tmp_path/"cache").mkdir()
    audio=np.zeros(6400,dtype=np.float32);audio[-1600:]=np.random.default_rng(5).normal(0,.03,1600)
    path=tmp_path/"breath.wav";sf.write(path,audio,16000)
    record={"path":"breath.wav","id":"test","kind":"breath","audio_license":"CC0-1.0","sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    _,_,original=data.breath_features(record)
    _,_,continued=data.breath_features({**record,"event_pipeline":"tail-v2"})
    np.testing.assert_array_equal(continued[:len(original)],original)
    assert (continued[len(original):]==-1).all()
    assert not (original[:-10]>0).any()
    assert (continued[:-10]>0).any()
from scripts.train import audit

def test_speaker_leakage_rejected():
    index={"records":[{"id":"a","speaker_hash":"same","transcript_hash":"x","split":"train"},{"id":"b","speaker_hash":"same","transcript_hash":"y","split":"test"}]}
    with pytest.raises(RuntimeError,match="speaker_hash"):
        audit(index)

def test_transcript_leakage_rejected():
    index={"records":[{"id":"a","speaker_hash":"one","transcript_hash":"same","split":"train"},{"id":"b","speaker_hash":"two","transcript_hash":"same","split":"validation"}]}
    with pytest.raises(RuntimeError,match="transcript_hash"):
        audit(index)
