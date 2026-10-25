import json
import numpy as np
import pytest
from kotoba_lipsync.features import FeatureConfig
from kotoba_lipsync.recurrent import RecurrentModel,export_recurrent
from kotoba_lipsync.temporal import expand_features


@pytest.mark.parametrize("quantized",[False,True])
def test_recurrent_numpy_matches_pytorch_and_stream(tmp_path,quantized):
    torch=pytest.importorskip("torch")
    from scripts.gru_network import RecurrentNetwork
    torch.manual_seed(11)
    network=RecurrentNetwork(8,2).eval()
    path=tmp_path/"gru.npz"
    export_recurrent(path,{k:v.detach().numpy() for k,v in network.state_dict().items()},np.zeros(118),np.ones(118),FeatureConfig(lookahead_frames=10),{"channels":8,"gru_layers":2,"has_breath_head":True},quantized=quantized)
    model=RecurrentModel(path);base=np.random.default_rng(3).normal(size=(200,78)).astype(np.float32)
    with torch.inference_mode():
        logits,breath=network(torch.from_numpy(expand_features(base))[None])
        expected=logits[0].softmax(-1).numpy();expected_breath=breath[0].sigmoid().numpy()
    actual,b=model.predict_sequence(base)
    np.testing.assert_allclose(actual,expected,atol=.01 if quantized else 2e-6)
    np.testing.assert_allclose(b,expected_breath,atol=.01 if quantized else 2e-6)
    stream=np.stack([model.predict_proba(row) for row in base])
    np.testing.assert_allclose(stream,actual,atol=2e-6)
    model.reset();np.testing.assert_allclose(model.predict_proba(base[0]),actual[0],atol=2e-6)


def test_recurrent_padding_does_not_change_initial_state():
    torch=pytest.importorskip("torch")
    from scripts.gru_network import RecurrentNetwork
    torch.manual_seed(4);network=RecurrentNetwork(8,2).eval()
    x=torch.randn(2,20,118);valid=torch.zeros(2,20,dtype=torch.bool)
    valid[0,5:16]=True;valid[1,8:20]=True
    with torch.inference_mode():
        actual,breath=network(x,valid)
        for row,start,end in ((0,5,16),(1,8,20)):
            expected,expected_breath=network(x[row:row+1,start:end])
            torch.testing.assert_close(actual[row,start:end],expected[0],atol=2e-6,rtol=1e-5)
            torch.testing.assert_close(breath[row,start:end],expected_breath[0],atol=2e-6,rtol=1e-5)
