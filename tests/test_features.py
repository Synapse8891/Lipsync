import numpy as np
from scipy.signal import resample_poly
from kotoba_lipsync.features import FeatureConfig, FeatureExtractor, StreamingFeatures, extract_audio
from kotoba_lipsync.audio import StreamingResampler

def signal_fixture(rate=16000):
    t=np.arange(rate)/rate
    return (.12*np.sin(2*np.pi*180*t)+.04*np.sin(2*np.pi*900*t)+.01*np.random.default_rng(3).normal(size=len(t))).astype(np.float32)

def test_chunk_boundaries_do_not_change_features():
    audio=signal_fixture()
    whole,times,_=extract_audio(audio)
    stream=StreamingFeatures()
    random=np.random.default_rng(4)
    frames=[]
    pos=0
    while pos<len(audio):
        count=int(random.integers(1,900))
        frames.extend(stream.push(audio[pos:pos+count]))
        pos+=count
    np.testing.assert_allclose(np.stack([f[0] for f in frames]),whole,atol=1e-6)
    np.testing.assert_allclose([f[1] for f in frames],times)
    assert len(whole)==100

def test_silence_features_finite():
    features,_,diag=extract_audio(np.zeros(1600,dtype=np.float32))
    assert np.isfinite(features).all()
    assert diag[-1]["rms_db"] < -100

def test_features_never_look_ahead():
    a=signal_fixture()
    b=a.copy()
    b[8000:]=0
    x,_,_=extract_audio(a)
    y,_,_=extract_audio(b)
    np.testing.assert_array_equal(x[:50],y[:50])

def test_bounded_lookahead_matches_training_alignment():
    audio=signal_fixture()
    causal,original_times,original_diag=extract_audio(audio)
    delayed,times,diag=extract_audio(audio,FeatureConfig(lookahead_frames=3))
    np.testing.assert_array_equal(delayed,causal[3:])
    np.testing.assert_array_equal(times,original_times[:-3])
    assert diag[0]==original_diag[0]
    stream=StreamingFeatures(FeatureConfig(lookahead_frames=3))
    assert stream.push(audio[:480])==[]
    output=stream.push(audio[480:640])
    assert len(output)==1
    np.testing.assert_array_equal(output[0][0],causal[3])

def test_resampler_continuity_at_48khz_and_44100():
    for rate in (48000,44100):
        audio=signal_fixture(rate)
        rs=StreamingResampler(rate)
        chunks=[rs.push(audio[i:i+rate//100]) for i in range(0,len(audio),rate//100)]
        actual=np.concatenate(chunks)
        expected=resample_poly(audio,16000//np.gcd(rate,16000),rate//np.gcd(rate,16000))
        # Last 10 ms waits for future anti-alias filter samples.
        assert len(actual)==len(expected)-160
        np.testing.assert_allclose(actual,expected[:len(actual)],atol=2e-6)
