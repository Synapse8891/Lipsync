"""NumPy GRU using PyTorch's r,z,n gate ordering; no framework at inference."""
import json
from pathlib import Path
import numpy as np
from .features import FeatureConfig
from .temporal import expand_features

FORMAT="kotoba-gru-int8-v1"


def export_recurrent(path,state,mean,scale,config,metadata,*,quantized=True):
    arrays={"mean":np.asarray(mean,dtype=np.float32),"scale":np.asarray(scale,dtype=np.float32)}
    for name,value in state.items():
        value=np.asarray(value,dtype=np.float32)
        if quantized and value.ndim==2:
            factor=np.maximum(np.abs(value).max(axis=1)/127,1e-8)
            arrays[name+".q"]=np.clip(np.rint(value/factor[:,None]),-127,127).astype(np.int8)
            arrays[name+".scale"]=factor
        else:arrays[name]=value
    arrays["metadata"]=np.array(json.dumps({**metadata,"format":FORMAT,"feature_config":config.to_dict(),"state_names":list(state),"quantized":quantized}))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp.npz");np.savez_compressed(temporary,**arrays);temporary.replace(path)


class RecurrentModel:
    stateful=True

    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.metadata=json.loads(str(data["metadata"]))
            if self.metadata.get("format")!=FORMAT:raise ValueError("Unsupported recurrent model")
            cfg=dict(self.metadata["feature_config"]);cfg["context"]=tuple(cfg["context"]);self.config=FeatureConfig(**cfg)
            self.mean=data["mean"].copy();self.scale=data["scale"].copy()
            self.state={name:(data[name+".q"].astype(np.float32)*data[name+".scale"][:,None] if name+".q" in data else data[name].copy()) for name in self.metadata["state_names"]}
        self.parameters=sum(a.size for a in self.state.values());self.float_weights_bytes=sum(a.nbytes for a in self.state.values())
        self.has_breath_head=self.metadata.get("has_breath_head",False)
        self.reset()

    def reset(self):
        self.hidden=np.zeros((self.metadata["gru_layers"],self.metadata["channels"]),dtype=np.float32);self.last_breath=0.

    @staticmethod
    def _sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))

    def _step(self,features,hidden):
        state=self.state;x=(expand_features(np.asarray(features)[:78])-self.mean)/self.scale
        x=np.maximum(state["project.weight"]@x+state["project.bias"],0)
        for layer in range(len(hidden)):
            gi=state[f"rnn.weight_ih_l{layer}"]@x+state[f"rnn.bias_ih_l{layer}"]
            gh=state[f"rnn.weight_hh_l{layer}"]@hidden[layer]+state[f"rnn.bias_hh_l{layer}"]
            ir,iz,inn=np.split(gi,3);hr,hz,hn=np.split(gh,3)
            reset=self._sigmoid(ir+hr);update=self._sigmoid(iz+hz)
            candidate=np.tanh(inn+reset*hn)
            hidden[layer]=(1-update)*candidate+update*hidden[layer];x=hidden[layer]
        logits=state["head.weight"]@x+state["head.bias"];p=np.exp(logits-logits.max());p/=p.sum()
        breath=float(self._sigmoid((state["breath.weight"]@x+state["breath.bias"])[0]))
        return p.astype(np.float32),breath

    def predict_proba(self,features):
        p,self.last_breath=self._step(features,self.hidden);return p

    def predict_sequence(self,base):
        hidden=np.zeros_like(self.hidden);probabilities=[];breaths=[]
        for frame in base:
            p,b=self._step(frame,hidden);probabilities.append(p);breaths.append(b)
        return np.asarray(probabilities,dtype=np.float32),np.asarray(breaths,dtype=np.float32)
