"""Training-only compact recurrent model; padding never evolves hidden state."""
import torch
from torch import nn
from torch.nn import functional as F


class RecurrentNetwork(nn.Module):
    def __init__(self,channels=96,layers=2,features=118,skip_lags=(),skip_hidden=32,language_head=False):
        super().__init__();self.channels=channels;self.layers=layers;self.dilations=()
        self.project=nn.Linear(features,channels)
        self.rnn=nn.GRU(channels,channels,layers,batch_first=True,dropout=.1 if layers>1 else 0)
        self.head=nn.Linear(channels,16);self.breath=nn.Linear(channels,1)
        self.skip_lags=tuple(skip_lags)
        self.language_head=language_head
        if language_head:
            self.language=nn.Linear(channels,2)
            self.delta_ja=nn.Linear(channels,16);self.delta_en=nn.Linear(channels,16)
            for module in (self.delta_ja,self.delta_en):nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
        if self.skip_lags:
            self.skip_project=nn.Linear(features*len(skip_lags),skip_hidden)
            self.skip_head=nn.Linear(skip_hidden,16);self.skip_breath=nn.Linear(skip_hidden,1)
            self.skip_dropout=nn.Dropout(.1)
            nn.init.zeros_(self.skip_head.weight);nn.init.zeros_(self.skip_head.bias)
            nn.init.zeros_(self.skip_breath.weight);nn.init.zeros_(self.skip_breath.bias)

    def forward(self,x,valid=None):
        z=F.relu(self.project(x))
        if valid is None:h,_=self.rnn(z)
        else:
            starts=(~valid).to(torch.int64).cumprod(1).sum(1)
            index=torch.arange(x.shape[1],device=x.device)[None,:]
            gather=(index+starts[:,None]).clamp(max=x.shape[1]-1)
            shifted=z.gather(1,gather[:,:,None].expand(-1,-1,z.shape[-1]))
            # Shift away invalid left padding. Right padding cannot influence
            # earlier GRU outputs and is masked below, so a packed sequence and
            # its CPU length synchronization are unnecessary.
            h,_=self.rnn(shifted)
            undo=(index-starts[:,None]).clamp(min=0)
            h=h.gather(1,undo[:,:,None].expand(-1,-1,h.shape[-1]))*valid[:,:,None]
        logits=self.head(h);breath=self.breath(h)
        if self.skip_lags:
            aligned=torch.cat([F.pad(x,(0,0,lag,0))[:,:x.shape[1]] for lag in self.skip_lags],dim=-1)
            direct=self.skip_dropout(F.relu(self.skip_project(aligned)))
            logits=logits+self.skip_head(direct);breath=breath+self.skip_breath(direct)
        if self.language_head:
            self.last_language_logits=self.language(h)
            language=F.log_softmax(self.last_language_logits,dim=-1)
            ja=F.log_softmax(logits+self.delta_ja(h),dim=-1)+language[...,:1]
            en=F.log_softmax(logits+self.delta_en(h),dim=-1)+language[...,1:]
            logits=torch.logaddexp(ja,en)
        return logits,breath.squeeze(-1)

    @property
    def receptive_frames(self):return 101  # BPTT warmup; streaming state itself is unbounded
