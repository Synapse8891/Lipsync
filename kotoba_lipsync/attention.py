"""Bounded-memory NumPy attention; PyTorch is not required at runtime."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from .features import FeatureConfig
from .normalization import CausalSpectralMean
from .temporal import expand_features
from .recurrent import RecurrentModel,export_recurrent

FORMAT="kotoba-attention-int8-v1"


def export_attention(path,state,mean,scale,config,metadata,*,quantized=True):
    export_recurrent(path,state,mean,scale,config,metadata,quantized=quantized,model_format=FORMAT)


class AttentionModel:
    stateful=True

    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.metadata=json.loads(str(data["metadata"]));cfg=dict(self.metadata["feature_config"]);cfg["context"]=tuple(cfg["context"]);self.config=FeatureConfig(**cfg)
            if self.metadata["format"]!=FORMAT:raise ValueError("Unsupported attention model")
            self.mean=data["mean"].copy();self.scale=data["scale"].copy()
            self.state={name:(data[name+".q"].astype(np.float32)*data[name+".scale"][:,None] if name+".q" in data else data[name].copy()) for name in self.metadata["state_names"]}
        self.parameters=sum(a.size for a in self.state.values());self.float_weights_bytes=sum(a.nbytes for a in self.state.values())
        self.has_breath_head=self.metadata.get("has_breath_head",False);self.feature_mode=self.metadata.get("feature_mode","cmn-v1")
        self.spectral_frontend=self.metadata.get("spectral_frontend",True);self.heads=self.metadata["attention_heads"]
        self.layers=self.metadata["attention_layers"];self.context=self.metadata["attention_context"]
        self.slopes=2.**(-np.arange(1,self.heads+1,dtype=np.float32));self.reset()

    def _new_cache(self):return [(deque(maxlen=self.context),deque(maxlen=self.context)) for _ in range(self.layers)]

    def reset(self):
        self.cache=self._new_cache();self.frontend_history=RecurrentModel._new_frontend_history()
        self.spectral_mean=CausalSpectralMean(mode=self.feature_mode);self.last_breath=0.
        self.last_embedding=np.zeros(self.metadata["channels"],dtype=np.float32)

    def _linear(self,x,name):return self.state[name+".weight"]@x+self.state[name+".bias"]

    def _norm(self,x,name):
        return (x-x.mean())/np.sqrt(x.var()+1e-5)*self.state[name+".weight"]+self.state[name+".bias"]

    def _step(self,features,cache,frontend_history):
        x=(features-self.mean)/self.scale;projected=self._linear(x,"project")
        if self.spectral_frontend:projected+=RecurrentModel._spectral_projection(self,x,frontend_history)
        x=np.maximum(projected,0)
        for layer,(keys,values) in enumerate(cache):
            prefix=f"blocks.{layer}.";q,k,v=self._linear(self._norm(x,prefix+"norm1"),prefix+"qkv").reshape(3,self.heads,-1)
            keys.append(k);values.append(v);kk=np.stack(keys,axis=1);vv=np.stack(values,axis=1)
            scores=np.einsum("hd,htd->ht",q,kk)*np.float32(q.shape[1]**-.5)
            scores-=self.slopes[:,None]*np.arange(len(keys)-1,-1,-1,dtype=np.float32)[None]
            probability=np.exp(scores-scores.max(axis=1,keepdims=True));probability/=probability.sum(axis=1,keepdims=True)
            attended=np.einsum("ht,htd->hd",probability,vv).reshape(-1)
            x=x+self._linear(attended,prefix+"out")
            x=x+self._linear(np.maximum(self._linear(self._norm(x,prefix+"norm2"),prefix+"ff1"),0),prefix+"ff2")
        x=self._norm(x,"norm");logits=self._linear(x,"head");p=np.exp(logits-logits.max());p/=p.sum()
        breath=float(RecurrentModel._sigmoid(self._linear(x,"breath")[0]+self.metadata.get("breath_logit_bias",0.)))
        return p.astype(np.float32),breath,x

    def _features(self,base,normalizer):
        x=expand_features(base)
        return np.r_[x,normalizer.push(base)] if self.feature_mode in ("cmn-v1","cmn-v2") else x

    def predict_proba(self,features):
        x=self._features(np.asarray(features)[:78],self.spectral_mean)
        p,self.last_breath,self.last_embedding=self._step(x,self.cache,self.frontend_history);return p

    def predict_sequence(self,base,*,capture_embedding=False):
        cache=self._new_cache();frontend=RecurrentModel._new_frontend_history();normalizer=CausalSpectralMean(mode=self.feature_mode)
        outputs=[];breaths=[];embeddings=[]
        for frame in base:
            p,b,h=self._step(self._features(frame,normalizer),cache,frontend)
            outputs.append(p);breaths.append(b)
            if capture_embedding:embeddings.append(h)
        result=(np.asarray(outputs,dtype=np.float32),np.asarray(breaths,dtype=np.float32))
        return (*result,np.asarray(embeddings,dtype=np.float32)) if capture_embedding else result
