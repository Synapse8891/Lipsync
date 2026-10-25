import pytest


def test_event_audio_requires_individual_cc0_license():
    from scripts.temporal_data import breath_features
    with pytest.raises(ValueError,match="individually declare CC0"):
        breath_features({"path":"data/nonexistent.wav","audio_license":"CC-BY-4.0"})
from scripts.train import audit

def test_speaker_leakage_rejected():
    index={"records":[{"id":"a","speaker_hash":"same","transcript_hash":"x","split":"train"},{"id":"b","speaker_hash":"same","transcript_hash":"y","split":"test"}]}
    with pytest.raises(RuntimeError,match="speaker_hash"):
        audit(index)

def test_transcript_leakage_rejected():
    index={"records":[{"id":"a","speaker_hash":"one","transcript_hash":"same","split":"train"},{"id":"b","speaker_hash":"two","transcript_hash":"same","split":"validation"}]}
    with pytest.raises(RuntimeError,match="transcript_hash"):
        audit(index)
