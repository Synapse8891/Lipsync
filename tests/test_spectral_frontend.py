import numpy as np
import pytest


def test_spectral_cnn_stream_matches_training_graph(tmp_path):
    torch=pytest.importorskip("torch")
    from scripts.gru_network import RecurrentNetwork
    from kotoba_lipsync.recurrent import export_recurrent,RecurrentModel
    from kotoba_lipsync.features import FeatureConfig
    from kotoba_lipsync.normalization import sequence_features
    torch.manual_seed(32)
    network=RecurrentNetwork(12,2,features=158,skip_lags=(0,2,4),skip_hidden=8,spectral_frontend=True).eval()
    torch.nn.init.normal_(network.spectral_project.weight,std=.1)
    path=tmp_path/"cnn.npz"
    export_recurrent(path,{k:v.detach().numpy() for k,v in network.state_dict().items()},np.zeros(158),np.ones(158),FeatureConfig(lookahead_frames=10),{"channels":12,"gru_layers":2,"feature_mode":"cmn-v1","spectral_frontend":True,"skip_lags":[0,2,4],"has_breath_head":True},quantized=False)
    base=np.random.default_rng(15).normal(size=(100,78)).astype(np.float32);base[:,68]=-.4;base[:,75]=.8
    with torch.inference_mode():
        logits,br=network(torch.from_numpy(sequence_features(base,"cmn-v1"))[None])
        expected=logits[0].softmax(-1).numpy();expected_br=br[0].sigmoid().numpy()
    model=RecurrentModel(path);actual,actual_br=model.predict_sequence(base)
    np.testing.assert_allclose(actual,expected,atol=2e-6);np.testing.assert_allclose(actual_br,expected_br,atol=2e-6)
    np.testing.assert_allclose(np.stack([model.predict_proba(row) for row in base]),actual,atol=2e-6)


def test_spectral_cnn_padding_does_not_create_fake_history():
    torch=pytest.importorskip("torch")
    from scripts.gru_network import RecurrentNetwork
    torch.manual_seed(33);network=RecurrentNetwork(8,2,spectral_frontend=True).eval()
    torch.nn.init.normal_(network.spectral_project.weight,std=.1)
    x=torch.randn(1,50,118);padded=torch.cat((torch.zeros(1,20,118),x),dim=1)
    valid=torch.cat((torch.zeros(1,20,dtype=torch.bool),torch.ones(1,50,dtype=torch.bool)),dim=1)
    with torch.inference_mode():
        expected=network(x);actual=network(padded,valid)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a[:,20:].numpy(),b.numpy(),atol=3e-7)
