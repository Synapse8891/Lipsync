import json
import numpy as np
import pytest


def test_embedding_and_acoustic_decoder_match_stream_and_bounded_chunks(tmp_path):
    torch=pytest.importorskip("torch")
    from scripts.attention_network import AttentionNetwork
    from scripts.network_tools import infer_sequence
    from kotoba_lipsync.normalization import sequence_features
    from kotoba_lipsync.attention import export_attention
    from kotoba_lipsync.features import FeatureConfig
    from kotoba_lipsync.model import load_model
    torch.manual_seed(51);rng=np.random.default_rng(51)
    network=AttentionNetwork(16,2,4,12).eval()
    base=rng.normal(size=(85,78)).astype(np.float32);base[:,68]=-.4;base[:,75]=.8
    with torch.inference_mode():
        expected=infer_sequence(network,sequence_features(base,"cmn-v2"),capture_embedding=True,chunk_frames=17)
    path=tmp_path/"body.npz"
    metadata={"channels":16,"attention_layers":2,"attention_heads":4,"attention_context":12,"feature_mode":"cmn-v2","spectral_frontend":True,"has_breath_head":True,"architecture":"attention"}
    export_attention(path,{k:v.detach().numpy() for k,v in network.state_dict().items()},np.zeros(158),np.ones(158),FeatureConfig(lookahead_frames=2),metadata,quantized=False)
    body=load_model(path);p,b,h=body.predict_sequence(base,capture_embedding=True)
    np.testing.assert_allclose(h,expected[2][0].numpy(),atol=2e-6)
    body.predict_proba(base[0]);saved=body.last_embedding.copy()
    body.predict_sequence(base,capture_embedding=True)
    np.testing.assert_array_equal(saved,body.last_embedding)
    with np.load(path) as data:arrays={k:data[k].copy() for k in data.files}
    meta=json.loads(str(arrays["metadata"]));meta["feature_config"]["lookahead_frames"]=3
    meta["decoder"]={"offsets":[-2,0,1],"base_lookahead_frames":2,"extra_lookahead_frames":1,"classes":list(range(16)),"embedding_features":True,"acoustic_features":True}
    width=3*16+1+16+118
    arrays.update(metadata=np.array(json.dumps(meta)),decoder_weight=rng.normal(0,.02,(16,width)).astype(np.float32),decoder_bias=np.zeros(16,np.float32),decoder_mean=np.zeros(width,np.float32),decoder_scale=np.ones(width,np.float32))
    path=tmp_path/"calibrated.npz";np.savez_compressed(path,**arrays)
    model=load_model(path);expected,b=model.predict_sequence(base)
    actual=np.stack([model.predict_proba(row) for row in base])
    np.testing.assert_allclose(actual,expected,atol=2e-6)
    assert len(model.acoustic_history)==4 and len(model.embedding_history)==2
    changed=base.copy();changed[45:]+=4
    future,_=model.predict_sequence(changed)
    np.testing.assert_array_equal(future[:45],expected[:45])
    model.reset();np.testing.assert_allclose(model.predict_proba(base[0]),expected[0],atol=2e-6)


def test_float32_decoder_solver_matches_weighted_multinomial_objective():
    pytest.importorskip("torch")
    from scripts.fit_decoder import fit_linear
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(52);x=rng.normal(size=(1200,8)).astype(np.float32)
    y=np.asarray([0,4,7])[(x@rng.normal(size=(8,3))+rng.normal(0,.5,(1200,3))).argmax(1)]
    weights=rng.uniform(.3,1.,len(y)).astype(np.float32)
    expected=LogisticRegression(C=.03,max_iter=300,tol=1e-7).fit(x,y,sample_weight=weights)
    actual=fit_linear(x,y,.03,sample_weight=weights,max_iter=200)
    np.testing.assert_allclose(actual.coef_,expected.coef_,atol=2e-3)
    assert np.mean(actual.predict(x)==expected.predict(x))>.995


def test_decoder_extends_long_lookahead_context_without_mutating_body():
    pytest.importorskip("torch")
    from scripts.train_temporal import model_feature_config
    from scripts.calibrate_temporal import decoder_feature_config
    from kotoba_lipsync.features import FeatureConfig
    original=model_feature_config(20);actual=FeatureConfig(**decoder_feature_config(original,2))
    assert actual.lookahead_frames==22 and max(actual.context)==22
    assert original.lookahead_frames==20 and max(original.context)==20
