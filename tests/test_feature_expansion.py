import numpy as np
import pytest


def test_expanded_input_preserves_gru_and_time_branch():
    torch=pytest.importorskip("torch")
    from scripts.gru_network import RecurrentNetwork
    from scripts.expand_recurrent_features import expand_state
    torch.manual_seed(14);lags=(0,2,4)
    old=RecurrentNetwork(12,2,features=118,skip_lags=lags,skip_hidden=8).eval()
    new=RecurrentNetwork(12,2,features=158,skip_lags=lags,skip_hidden=8).eval()
    new.load_state_dict(expand_state(old.state_dict(),158,lags))
    x=torch.randn(2,60,158)
    with torch.inference_mode():
        expected=old(x[:,:,:118]);actual=new(x)
    for a,b in zip(actual,expected):np.testing.assert_allclose(a.numpy(),b.numpy(),atol=3e-7)
