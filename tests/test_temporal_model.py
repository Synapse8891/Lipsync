import numpy as np
import pytest
from kotoba_lipsync.features import FeatureConfig
from kotoba_lipsync.temporal import expand_features,export_temporal,TemporalModel
from kotoba_lipsync.features import extract_audio
from kotoba_lipsync.runtime import LipSyncEngine
from kotoba_lipsync.model import load_model
import json

def make_network(tmp_path,quantized=False,skip_lags=()):
    torch=pytest.importorskip("torch")
    from scripts.tcn_network import TemporalNetwork
    torch.manual_seed(3)
    network=TemporalNetwork(channels=8,dilations=(1,2),skip_lags=skip_lags,skip_hidden=8).eval()
    if skip_lags:
        with torch.no_grad():
            network.skip_head.weight.uniform_(-.2,.2)
            network.skip_breath.weight.uniform_(-.2,.2)
    path=tmp_path/"temporal.npz"
    export_temporal(path,{k:v.detach().numpy() for k,v in network.state_dict().items()},np.zeros(118),np.ones(118),FeatureConfig(lookahead_frames=3),{"channels":8,"dilations":[1,2],"skip_lags":list(skip_lags),"has_breath_head":True},quantized=quantized)
    return torch,network,TemporalModel(path)

@pytest.mark.parametrize("skip_lags",[(),(0,2,5)])
def test_temporal_numpy_matches_pytorch_and_frame_stream(tmp_path,skip_lags):
    torch,network,model=make_network(tmp_path,skip_lags=skip_lags)
    base=np.random.default_rng(4).normal(size=(120,78)).astype(np.float32)
    with torch.inference_mode():
        logits,breath=network(torch.from_numpy(expand_features(base))[None])
        expected=logits[0].softmax(-1).numpy();expected_breath=breath[0].sigmoid().numpy()
    actual,actual_breath=model.predict_sequence(base)
    np.testing.assert_allclose(actual,expected,atol=2e-6)
    np.testing.assert_allclose(actual_breath,expected_breath,atol=2e-6)
    online=np.stack([model.predict_proba(frame) for frame in base])
    np.testing.assert_allclose(online,actual,atol=2e-6)

def test_temporal_quantization_error_is_bounded(tmp_path):
    torch,network,model=make_network(tmp_path,quantized=True)
    base=np.random.default_rng(5).normal(size=(120,78)).astype(np.float32)
    with torch.inference_mode():expected=network(torch.from_numpy(expand_features(base))[None])[0][0].softmax(-1).numpy()
    actual,_=model.predict_sequence(base)
    assert np.abs(actual-expected).mean()<.002
    assert np.abs(actual-expected).max()<.025

def test_temporal_network_is_causal_and_resettable(tmp_path):
    _,_,model=make_network(tmp_path)
    a=np.random.default_rng(2).normal(size=(80,78)).astype(np.float32)
    b=a.copy();b[40:]*=-3
    pa,_=model.predict_sequence(a);pb,_=model.predict_sequence(b)
    np.testing.assert_array_equal(pa[:40],pb[:40])
    first=np.stack([model.predict_proba(frame) for frame in a])
    model.reset()
    second=np.stack([model.predict_proba(frame) for frame in a])
    np.testing.assert_array_equal(first,second)

def test_gain_invariant_mel_features():
    a=np.zeros((4,78),dtype=np.float32);a[:,:40]=np.arange(40)
    b=a.copy();b[:,:40]+=2
    np.testing.assert_allclose(expand_features(a)[:,78:],expand_features(b)[:,78:],atol=1e-6)


def test_runtime_temporal_warmup_and_audio_chunks_match_offline(tmp_path):
    _,_,model=make_network(tmp_path)
    audio=np.random.default_rng(7).normal(0,.02,16000).astype(np.float32)
    base=extract_audio(audio,FeatureConfig())[0][:,:78]
    expected,breath=model.predict_sequence(base)
    whole=LipSyncEngine(tmp_path/"temporal.npz",silence_db=-200,smoothing_ms=0).push(audio)
    stream=LipSyncEngine(tmp_path/"temporal.npz",silence_db=-200,smoothing_ms=0)
    chunks=[r for i in range(0,len(audio),137) for r in stream.push(audio[i:i+137])]
    np.testing.assert_allclose([r.weights for r in whole],expected[3:,:15],atol=2e-6)
    np.testing.assert_allclose([r.breath for r in whole],breath[3:],atol=2e-6)
    np.testing.assert_allclose([r.weights for r in chunks],[r.weights for r in whole],atol=2e-6)
    np.testing.assert_allclose([r.breath for r in chunks],[r.breath for r in whole],atol=2e-6)


def test_breath_head_is_independent_of_h_phoneme(tmp_path):
    _,network,_=make_network(tmp_path)
    state={k:v.detach().numpy().copy() for k,v in network.state_dict().items()}
    state["head.weight"][:]=0;state["head.bias"][:]=0;state["head.bias"][15]=20
    state["breath.weight"][:]=0;state["breath.bias"][:]=-20
    path=tmp_path/"h_only.npz"
    export_temporal(path,state,np.zeros(118),np.ones(118),FeatureConfig(lookahead_frames=3),{"channels":8,"dilations":[1,2],"has_breath_head":True,"labels":[str(i) for i in range(15)]+["HH"]})
    engine=LipSyncEngine(path,silence_db=-200,smoothing_ms=0)
    result=engine.push(np.random.default_rng(8).normal(0,.02,1600).astype(np.float32))[-1]
    assert result.viseme=="HH"
    assert result.breath<1e-6


def test_padded_future_has_no_training_target():
    pytest.importorskip("torch")
    from scripts.train_temporal import SequenceDataset
    class ShortCorpus:
        rows=[{"split":"train","frames":3}]
        mean=np.zeros(118,dtype=np.float32)
        scale=np.ones(118,dtype=np.float32)
        def arrays(self,row):return np.zeros((3,78),dtype=np.float32),np.array([10,11,12]),np.zeros(3)
    _,target,_,valid=SequenceDataset(ShortCorpus(),lookahead=1,context=2,frames=4)[0]
    assert target.tolist()==[-100,-100,-100,10,11,-100]
    assert not valid[-1]


@pytest.mark.parametrize("architecture",["tcn","gru"])
def test_decoder_stream_matches_batch_and_has_no_future_input(tmp_path,architecture):
    if architecture=="tcn":make_network(tmp_path,skip_lags=(0,2,5))
    else:
        pytest.importorskip("torch")
        from scripts.gru_network import RecurrentNetwork
        from kotoba_lipsync.recurrent import export_recurrent
        network=RecurrentNetwork(8,2).eval()
        export_recurrent(tmp_path/"temporal.npz",{k:v.detach().numpy() for k,v in network.state_dict().items()},np.zeros(118),np.ones(118),FeatureConfig(lookahead_frames=3),{"channels":8,"gru_layers":2,"has_breath_head":True},quantized=False)
    path=tmp_path/"temporal.npz"
    with np.load(path) as data:arrays={key:data[key].copy() for key in data.files}
    metadata=json.loads(str(arrays["metadata"]))
    metadata["feature_config"]["lookahead_frames"]=4
    metadata["decoder"]={"offsets":[-1,0,1],"base_lookahead_frames":3,"classes":list(range(16))}
    rng=np.random.default_rng(12)
    arrays.update(metadata=np.array(json.dumps(metadata)),decoder_weight=rng.normal(0,.1,(16,49)).astype(np.float32),decoder_bias=np.zeros(16,dtype=np.float32),decoder_mean=np.zeros(49,dtype=np.float32),decoder_scale=np.ones(49,dtype=np.float32))
    np.savez_compressed(path,**arrays)
    model=load_model(path)
    base=rng.normal(size=(100,78)).astype(np.float32)
    expected,expected_breath=model.predict_sequence(base)
    actual=[];breath=[]
    for row in base:actual.append(model.predict_proba(row));breath.append(model.last_breath)
    np.testing.assert_allclose(actual,expected,atol=2e-6)
    np.testing.assert_allclose(breath,expected_breath,atol=2e-6)
    changed=base.copy();changed[50:]*=10
    other,_=model.predict_sequence(changed)
    np.testing.assert_allclose(other[:50],expected[:50],atol=2e-6)
    model.reset()
    np.testing.assert_allclose(model.predict_proba(base[0]),actual[0],atol=2e-6)
