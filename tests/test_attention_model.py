import numpy as np
import pytest


def test_attention_stream_matches_torch_and_has_bounded_state(tmp_path):
    torch=pytest.importorskip("torch")
    from scripts.attention_network import AttentionNetwork
    from kotoba_lipsync.attention import export_attention
    from kotoba_lipsync.model import load_model
    from kotoba_lipsync.features import FeatureConfig
    from kotoba_lipsync.normalization import sequence_features
    torch.manual_seed(46);network=AttentionNetwork(16,2,4,12).eval()
    base=np.random.default_rng(46).normal(size=(80,78)).astype(np.float32);base[:,68]=-.4;base[:,75]=.8
    with torch.inference_mode():
        logits,breath=network(torch.from_numpy(sequence_features(base,"cmn-v2"))[None]);expected=logits[0].softmax(-1).numpy()
    path=tmp_path/"attention.npz"
    export_attention(path,{k:v.detach().numpy() for k,v in network.state_dict().items()},np.zeros(158),np.ones(158),FeatureConfig(lookahead_frames=10),{"channels":16,"attention_layers":2,"attention_heads":4,"attention_context":12,"feature_mode":"cmn-v2","spectral_frontend":True,"has_breath_head":True},quantized=False)
    model=load_model(path);actual,bb=model.predict_sequence(base)
    np.testing.assert_allclose(actual,expected,atol=2e-6);np.testing.assert_allclose(bb,breath[0].sigmoid().numpy(),atol=2e-6)
    np.testing.assert_allclose(np.stack([model.predict_proba(row) for row in base]),expected,atol=2e-6)
    assert model.last_embedding.dtype==np.float32
    assert all(k.dtype==v.dtype==np.float32 for keys,values in model.cache for k,v in zip(keys,values))
    assert all(len(keys)==12 and len(values)==12 for keys,values in model.cache)
    model.reset();np.testing.assert_allclose(model.predict_proba(base[0]),expected[0],atol=2e-6)


def test_attention_padding_is_causal_and_does_not_poison_valid_frames():
    torch=pytest.importorskip("torch")
    from scripts.attention_network import AttentionNetwork
    torch.manual_seed(47);network=AttentionNetwork(16,2,4,12).eval()
    x=torch.randn(1,40,158);padded=torch.cat((torch.zeros(1,20,158),x),dim=1)
    valid=torch.cat((torch.zeros(1,20,dtype=torch.bool),torch.ones(1,40,dtype=torch.bool)),dim=1)
    with torch.inference_mode():
        expected=network(x);actual=network(padded,valid)
        changed=x.clone();changed[:,20:]+=5;future=network(changed)
    for a,b,c in zip(actual,expected,future):
        np.testing.assert_allclose(a[:,20:].numpy(),b.numpy(),atol=2e-6)
        np.testing.assert_allclose(c[:,:20].numpy(),b[:,:20].numpy(),atol=2e-6)


def test_chunked_attention_does_not_change_record_predictions():
    torch=pytest.importorskip("torch")
    from scripts.attention_network import AttentionNetwork
    from scripts.network_tools import infer_sequence
    torch.manual_seed(48);network=AttentionNetwork(16,2,4,12).eval()
    x=np.random.default_rng(48).normal(size=(103,158)).astype(np.float32)
    with torch.inference_mode():expected=network(torch.from_numpy(x)[None]);actual=infer_sequence(network,x,chunk_frames=17)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a.numpy(),b.numpy(),atol=2e-6)


def test_attention_widening_preserves_class_and_breath_logits():
    torch=pytest.importorskip("torch")
    from scripts.attention_network import AttentionNetwork
    from scripts.widen_attention import widen_attention_state
    torch.manual_seed(49);old=AttentionNetwork(16,2,4,12).eval();new=AttentionNetwork(32,2,4,12).eval()
    new.load_state_dict(widen_attention_state(old.state_dict(),heads=4,context=12))
    x=torch.randn(2,60,158)
    with torch.inference_mode():expected=old(x);actual=new(x)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a.numpy(),b.numpy(),atol=3e-6)


def test_widening_and_adding_identity_layers_preserves_outputs():
    torch=pytest.importorskip("torch")
    from scripts.attention_network import AttentionNetwork
    from scripts.widen_attention import widen_attention_state
    torch.manual_seed(53);old=AttentionNetwork(16,2,4,12).eval();new=AttentionNetwork(32,4,4,12).eval()
    new.load_state_dict(widen_attention_state(old.state_dict(),heads=4,context=12,layers=4))
    x=torch.randn(2,60,158)
    with torch.inference_mode():expected=old(x);actual=new(x)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a.numpy(),b.numpy(),atol=3e-6)
