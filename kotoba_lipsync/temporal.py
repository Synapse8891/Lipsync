"""Small causal TCN with int8 storage and NumPy-only streaming inference."""
from collections import deque
import json
from pathlib import Path
import numpy as np
from .features import FeatureConfig

FORMAT="kotoba-tcn-int8-v1"

def expand_features(base):
    base=np.asarray(base,dtype=np.float32)
    if base.shape[-1]!=78:raise ValueError("Expected 78 base acoustic features")
    mel=base[...,:40]
    return np.concatenate([base,mel-mel.mean(axis=-1,keepdims=True)],axis=-1)

def export_temporal(path,state,mean,scale,config,metadata,*,quantized=True):
    """state is a mapping of NumPy arrays; PyTorch is never imported here."""
    arrays={"mean":np.asarray(mean,dtype=np.float32),"scale":np.asarray(scale,dtype=np.float32)}
    names=[]
    for name,value in state.items():
        value=np.asarray(value,dtype=np.float32)
        names.append(name)
        if quantized and value.ndim==3:
            factor=np.maximum(np.max(np.abs(value),axis=(1,2))/127,1e-8)
            arrays[name+".q"]=np.clip(np.rint(value/factor[:,None,None]),-127,127).astype(np.int8)
            arrays[name+".scale"]=factor.astype(np.float32)
        else:arrays[name]=value
    arrays["metadata"]=np.array(json.dumps({**metadata,"format":FORMAT,"feature_config":config.to_dict(),"state_names":names,"quantized":quantized}))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,**arrays)
    temporary.replace(path)

class TemporalModel:
    stateful = True
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.metadata=json.loads(str(data["metadata"]))
            if self.metadata.get("format")!=FORMAT:raise ValueError("Invalid temporal model")
            cfg=dict(self.metadata["feature_config"]);cfg["context"]=tuple(cfg["context"])
            self.config=FeatureConfig(**cfg)
            self.mean=data["mean"].copy();self.scale=data["scale"].copy()
            self.state={}
            for name in self.metadata["state_names"]:
                self.state[name]=(data[name+".q"].astype(np.float32)*data[name+".scale"][:,None,None]) if name+".q" in data else data[name].copy()
        self.dilations=tuple(self.metadata["dilations"])
        self.skip_lags=tuple(self.metadata.get("skip_lags",()))
        self.parameters=sum(x.size for x in self.state.values())
        self.float_weights_bytes=sum(x.nbytes for x in self.state.values())
        self.has_breath_head=bool(self.metadata.get("has_breath_head",False))
        self.reset()

    def reset(self):
        self.history={}
        self.last_breath=0.0

    def _conv_step(self,key,x,dilation=1):
        w=self.state[key+".weight"];bias=self.state[key+".bias"]
        kernel=w.shape[-1]
        if kernel==1:return w[:,:,0]@x+bias
        if key not in self.history:
            self.history[key]=deque([np.zeros_like(x) for _ in range((kernel-1)*dilation+1)],maxlen=(kernel-1)*dilation+1)
        history=self.history[key];history.append(x)
        result=bias.copy()
        for k in range(kernel):result+=w[:,:,k]@history[-1-(kernel-1-k)*dilation]
        return result

    def _norm(self,key,x):
        return (x-x.mean(axis=-1,keepdims=True))/np.sqrt(x.var(axis=-1,keepdims=True)+1e-5)*self.state[key+".weight"]+self.state[key+".bias"]

    @staticmethod
    def _softmax(x):
        z=np.exp(x-np.max(x,axis=-1,keepdims=True));return z/z.sum(axis=-1,keepdims=True)

    def predict_proba(self,features):
        if np.asarray(features).ndim!=1:raise ValueError("Use predict_sequence for a complete temporal sequence")
        x=(expand_features(np.asarray(features)[:78])-self.mean)/self.scale
        h=np.maximum(self._conv_step("project",x),0)
        for i,d in enumerate(self.dilations):
            z=np.maximum(self._conv_step(f"blocks.{i}.conv1",h,d),0)
            z=self._conv_step(f"blocks.{i}.conv2",z,d)
            h=np.maximum(self._norm(f"blocks.{i}.norm",h+z),0)
        logit=float(self._conv_step("breath",h)[0])
        logits=self._conv_step("head",h)
        if self.skip_lags:
            if "skip" not in self.history:
                self.history["skip"]=deque([np.zeros_like(x) for _ in range(max(self.skip_lags)+1)],maxlen=max(self.skip_lags)+1)
            history=self.history["skip"];history.append(x)
            direct=np.maximum(self._conv_step("skip_project",np.concatenate([history[-1-lag] for lag in self.skip_lags])),0)
            logits+=self._conv_step("skip_head",direct)
            logit+=float(self._conv_step("skip_breath",direct)[0])
        self.last_breath=float(1/(1+np.exp(-np.clip(logit+self.metadata.get("breath_logit_bias",0.),-40,40))))
        return self._softmax(logits).astype(np.float32)

    def _conv_sequence(self,key,x,dilation=1):
        w=self.state[key+".weight"];bias=self.state[key+".bias"]
        result=np.broadcast_to(bias,(len(x),len(bias))).copy()
        for k in range(w.shape[-1]):
            delay=(w.shape[-1]-1-k)*dilation
            if delay<len(x):result[delay:]+=x[:len(x)-delay]@w[:,:,k].T
        return result

    def predict_sequence(self,base):
        x=(expand_features(base)-self.mean)/self.scale
        h=np.maximum(self._conv_sequence("project",x),0)
        for i,d in enumerate(self.dilations):
            z=np.maximum(self._conv_sequence(f"blocks.{i}.conv1",h,d),0)
            z=self._conv_sequence(f"blocks.{i}.conv2",z,d)
            h=np.maximum(self._norm(f"blocks.{i}.norm",h+z),0)
        breath=self._conv_sequence("breath",h)[:,0]
        logits=self._conv_sequence("head",h)
        if self.skip_lags:
            shifted=[]
            for lag in self.skip_lags:
                part=np.zeros_like(x)
                if lag<len(x):part[lag:]=x[:len(x)-lag]
                shifted.append(part)
            direct=np.maximum(self._conv_sequence("skip_project",np.concatenate(shifted,axis=-1)),0)
            logits+=self._conv_sequence("skip_head",direct)
            breath+=self._conv_sequence("skip_breath",direct)[:,0]
        return self._softmax(logits).astype(np.float32),(1/(1+np.exp(-np.clip(breath+self.metadata.get("breath_logit_bias",0.),-40,40)))).astype(np.float32)
