"""NumPy GRU using PyTorch's r,z,n gate ordering; no framework at inference."""
import json
from pathlib import Path
import numpy as np
from collections import deque
from .features import FeatureConfig
from .temporal import expand_features
from .normalization import CausalSpectralMean

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
        self.skip_lags=tuple(self.metadata.get("skip_lags",()))
        self.feature_mode=self.metadata.get("feature_mode","cmn-v1" if len(self.mean)==158 else "base")
        self.reset()

    def reset(self):
        self.hidden=np.zeros((self.metadata["gru_layers"],self.metadata["channels"]),dtype=np.float32);self.last_breath=0.
        self.feature_history=self._new_history()
        self.spectral_mean=CausalSpectralMean()

    def _new_history(self):
        return deque([np.zeros(len(self.mean),dtype=np.float32) for _ in range(max(self.skip_lags,default=0)+1)],maxlen=max(self.skip_lags,default=0)+1)

    @staticmethod
    def _sigmoid(x):return 1/(1+np.exp(-np.clip(x,-40,40)))

    def _step(self,features,hidden,history):
        state=self.state;x=(np.asarray(features,dtype=np.float32)-self.mean)/self.scale
        history.append(x)
        x=np.maximum(state["project.weight"]@x+state["project.bias"],0)
        for layer in range(len(hidden)):
            gi=state[f"rnn.weight_ih_l{layer}"]@x+state[f"rnn.bias_ih_l{layer}"]
            gh=state[f"rnn.weight_hh_l{layer}"]@hidden[layer]+state[f"rnn.bias_hh_l{layer}"]
            ir,iz,inn=np.split(gi,3);hr,hz,hn=np.split(gh,3)
            reset=self._sigmoid(ir+hr);update=self._sigmoid(iz+hz)
            candidate=np.tanh(inn+reset*hn)
            hidden[layer]=(1-update)*candidate+update*hidden[layer];x=hidden[layer]
        logits=state["head.weight"]@x+state["head.bias"]
        breath_logit=(state["breath.weight"]@x+state["breath.bias"])[0]
        if self.skip_lags:
            aligned=np.concatenate([history[-1-lag] for lag in self.skip_lags])
            direct=np.maximum(state["skip_project.weight"]@aligned+state["skip_project.bias"],0)
            logits+=state["skip_head.weight"]@direct+state["skip_head.bias"]
            breath_logit+=(state["skip_breath.weight"]@direct+state["skip_breath.bias"])[0]
        if "language.weight" in state:
            language=state["language.weight"]@x+state["language.bias"];language=np.exp(language-language.max());language/=language.sum()
            mixed=np.zeros(16,dtype=np.float32)
            for index,key in enumerate(("delta_ja","delta_en")):
                values=logits+state[key+".weight"]@x+state[key+".bias"]
                values=np.exp(values-values.max());mixed+=language[index]*values/values.sum()
            p=mixed
        else:
            p=np.exp(logits-logits.max());p/=p.sum()
        breath=float(self._sigmoid(breath_logit+self.metadata.get("breath_logit_bias",0.)))
        return p.astype(np.float32),breath

    def predict_proba(self,features):
        base=np.asarray(features)[:78];expanded=expand_features(base)
        if self.feature_mode=="cmn-v1":expanded=np.r_[expanded,self.spectral_mean.push(base)]
        p,self.last_breath=self._step(expanded,self.hidden,self.feature_history);return p

    def predict_sequence(self,base):
        hidden=np.zeros_like(self.hidden);probabilities=[];breaths=[];history=self._new_history();normalizer=CausalSpectralMean()
        for frame in base:
            expanded=expand_features(frame)
            if self.feature_mode=="cmn-v1":expanded=np.r_[expanded,normalizer.push(frame)]
            p,b=self._step(expanded,hidden,history);probabilities.append(p);breaths.append(b)
        return np.asarray(probabilities,dtype=np.float32),np.asarray(breaths,dtype=np.float32)
