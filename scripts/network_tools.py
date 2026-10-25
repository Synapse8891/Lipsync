"""Training-side model construction and bounded-memory full-record inference."""
import torch
from .attention_network import AttentionNetwork
from .gru_network import RecurrentNetwork
from .tcn_network import TemporalNetwork


def build_network(metadata,features):
    architecture=metadata.get("architecture","tcn")
    if architecture=="attention":
        return AttentionNetwork(metadata["channels"],metadata["attention_layers"],metadata["attention_heads"],metadata["attention_context"],features=features,spectral_frontend=metadata.get("spectral_frontend",False))
    if architecture=="gru":
        return RecurrentNetwork(metadata["channels"],metadata["gru_layers"],features=features,skip_lags=metadata.get("skip_lags",()),skip_hidden=metadata.get("skip_hidden",32),language_head=metadata.get("language_head",False),spectral_frontend=metadata.get("spectral_frontend",False))
    if architecture=="tcn":
        return TemporalNetwork(metadata["channels"],metadata["dilations"],features=features,skip_lags=metadata.get("skip_lags",()),skip_hidden=metadata.get("skip_hidden",64))
    raise ValueError("Unsupported training architecture: "+str(architecture))


def infer_sequence(network,values,device=None,chunk_frames=512):
    device=device or next(network.parameters()).device
    if not isinstance(network,AttentionNetwork):return network(torch.from_numpy(values)[None].to(device))
    outputs=[];breaths=[];warmup=network.receptive_frames-1
    for start in range(0,len(values),chunk_frames):
        left=max(0,start-warmup);end=min(start+chunk_frames,len(values))
        logits,br=network(torch.from_numpy(values[left:end])[None].to(device))
        outputs.append(logits[:,start-left:]);breaths.append(br[:,start-left:])
    return torch.cat(outputs,dim=1),torch.cat(breaths,dim=1)
