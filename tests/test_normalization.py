import numpy as np
import pytest
from kotoba_lipsync.normalization import CausalSpectralMean,spectral_residuals,sequence_features


def test_spectral_mean_is_causal_and_removes_constant_channel_response():
    base=np.random.default_rng(5).normal(size=(100,78)).astype(np.float32)
    base[:,68]=-.4;base[:,75]=.8
    changed=base.copy();changed[50:,:40]+=10
    np.testing.assert_array_equal(spectral_residuals(base)[:50],spectral_residuals(changed)[:50])
    equalized=base.copy();equalized[:,:40]+=np.linspace(-2,2,40)
    np.testing.assert_allclose(spectral_residuals(base),spectral_residuals(equalized),atol=2e-6)
    normalizer=CausalSpectralMean()
    np.testing.assert_array_equal(np.stack([normalizer.push(row) for row in base]),spectral_residuals(base))


def test_cmn_recurrent_stream_matches_pytorch(tmp_path):
    torch=pytest.importorskip("torch")
    from scripts.gru_network import RecurrentNetwork
    from kotoba_lipsync.recurrent import export_recurrent,RecurrentModel
    from kotoba_lipsync.features import FeatureConfig
    torch.manual_seed(3);network=RecurrentNetwork(8,2,features=158).eval()
    path=tmp_path/"cmn.npz"
    export_recurrent(path,{k:v.detach().numpy() for k,v in network.state_dict().items()},np.zeros(158),np.ones(158),FeatureConfig(lookahead_frames=10),{"channels":8,"gru_layers":2,"feature_mode":"cmn-v1","has_breath_head":True},quantized=False)
    base=np.random.default_rng(6).normal(size=(100,78)).astype(np.float32);base[:,68]=-.4;base[:,75]=.8
    with torch.inference_mode():expected=network(torch.from_numpy(sequence_features(base,"cmn-v1"))[None])[0][0].softmax(-1).numpy()
    model=RecurrentModel(path);actual,_=model.predict_sequence(base)
    np.testing.assert_allclose(actual,expected,atol=2e-6)
    np.testing.assert_allclose(np.stack([model.predict_proba(row) for row in base]),actual,atol=2e-6)
