"""Small calibrated posterior decoder with explicitly bounded extra lookahead."""
from collections import deque
import numpy as np
from .temporal import expand_features


class CalibratedModel:
    stateful=True

    def __init__(self,base,path):
        self.base=base;self.metadata=base.metadata;self.config=base.config
        config=self.metadata["decoder"]
        self.offsets=tuple(config["offsets"])
        self.extra=max(self.offsets);self.base_lookahead=config["base_lookahead_frames"]
        self.depth=self.extra-min(self.offsets)+1
        self.embedding_features=config.get("embedding_features",False)
        self.acoustic_features=config.get("acoustic_features",False)
        with np.load(path,allow_pickle=False) as data:
            self.weight=data["decoder_weight"].copy();self.bias=data["decoder_bias"].copy()
            self.mean=data["decoder_mean"].copy();self.scale=data["decoder_scale"].copy()
            self.residual=[data[key].copy() for key in ("decoder_residual_w0","decoder_residual_b0","decoder_residual_w1","decoder_residual_b1")] if "decoder_residual_w0" in data else None
        self.classes=np.asarray(config["classes"],dtype=np.int32)
        self.parameters=base.parameters+self.weight.size+self.bias.size
        self.float_weights_bytes=base.float_weights_bytes+self.weight.nbytes+self.bias.nbytes
        if self.residual:
            self.parameters+=sum(array.size for array in self.residual)
            self.float_weights_bytes+=sum(array.nbytes for array in self.residual)
        self.has_breath_head=base.has_breath_head
        self.reset()

    def reset(self):
        self.base.reset();self.steps=0;self.posterior=None;self.breath_history=None
        self.energy=deque(maxlen=self.config.lookahead_frames+1)
        self.acoustic_history=deque(maxlen=self.config.lookahead_frames+1)
        self.embedding_history=None
        self.last_breath=0.

    def _classify(self,features):
        scaled=(features-self.mean)/self.scale
        logits=scaled@self.weight.T+self.bias
        if self.residual:
            w0,b0,w1,b1=self.residual
            logits+=np.maximum(scaled@w0.T+b0,0)@w1.T+b1
        logits-=logits.max(axis=-1,keepdims=True)
        p=np.exp(logits);p/=p.sum(axis=-1,keepdims=True)
        full=np.zeros((*p.shape[:-1],16),dtype=np.float32);full[...,self.classes]=p
        return full

    def predict_proba(self,features):
        raw=self.base.predict_proba(features);breath=self.base.last_breath
        self.energy.append(float(features[68]));step=self.steps;self.steps+=1
        if self.acoustic_features:self.acoustic_history.append(np.asarray(features[:78],dtype=np.float32).copy())
        if step<self.base_lookahead:
            self.last_breath=breath
            return raw
        if self.posterior is None:
            self.posterior=deque([raw]*self.depth,maxlen=self.depth)
            self.breath_history=deque([breath]*(self.extra+1),maxlen=self.extra+1)
            if self.embedding_features:self.embedding_history=deque([self.base.last_embedding]*(self.extra+1),maxlen=self.extra+1)
        else:
            self.posterior.append(raw);self.breath_history.append(breath)
            if self.embedding_features:self.embedding_history.append(self.base.last_embedding)
        context=np.concatenate([self.posterior[-1-(self.extra-offset)] for offset in self.offsets])
        parts=[np.log(np.maximum(context,1e-6)),np.asarray([self.energy[0]])]
        if self.embedding_features:parts.append(self.embedding_history[0])
        if self.acoustic_features:parts.append(expand_features(self.acoustic_history[0]))
        x=np.concatenate(parts).astype(np.float32)
        self.last_breath=float(self.breath_history[0])
        return self._classify(x)

    def predict_sequence(self,base_features):
        if self.embedding_features:
            raw,breath,embedding=self.base.predict_sequence(base_features,capture_embedding=True)
        else:raw,breath=self.base.predict_sequence(base_features)
        output=raw.copy();breath=breath.copy()
        rows=np.arange(self.base_lookahead,len(raw))
        if len(rows):
            target=rows-self.base_lookahead-self.extra
            indices=np.maximum(target[:,None]+np.asarray(self.offsets)[None,:],0)+self.base_lookahead
            logp=np.log(np.maximum(raw[indices],1e-6)).reshape(len(rows),-1)
            db=np.asarray(base_features)[np.maximum(target,0),68:69]
            parts=[logp,db]
            if self.embedding_features:parts.append(embedding[np.maximum(rows-self.extra,self.base_lookahead)])
            if self.acoustic_features:parts.append(expand_features(np.asarray(base_features)[np.maximum(target,0)]))
            output[rows]=self._classify(np.concatenate(parts,axis=1))
            breath[rows]=breath[np.maximum(rows-self.extra,self.base_lookahead)]
        return output,breath
