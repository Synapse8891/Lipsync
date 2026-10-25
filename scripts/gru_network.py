"""Training-only compact recurrent model; padding never evolves hidden state."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class RecurrentNetwork(nn.Module):
    def __init__(self,channels=96,layers=2,features=118):
        super().__init__();self.channels=channels;self.layers=layers;self.dilations=()
        self.project=nn.Linear(features,channels)
        self.rnn=nn.GRU(channels,channels,layers,batch_first=True,dropout=.1 if layers>1 else 0)
        self.head=nn.Linear(channels,16);self.breath=nn.Linear(channels,1)

    def forward(self,x,valid=None):
        z=F.relu(self.project(x))
        if valid is None:h,_=self.rnn(z)
        else:
            lengths=valid.sum(1)
            starts=(~valid).to(torch.int64).cumprod(1).sum(1)
            index=torch.arange(x.shape[1],device=x.device)[None,:]
            gather=(index+starts[:,None]).clamp(max=x.shape[1]-1)
            shifted=z.gather(1,gather[:,:,None].expand(-1,-1,z.shape[-1]))
            packed=pack_padded_sequence(shifted,lengths.cpu(),batch_first=True,enforce_sorted=False)
            packed,_=self.rnn(packed)
            h,_=pad_packed_sequence(packed,batch_first=True,total_length=x.shape[1])
            undo=(index-starts[:,None]).clamp(min=0)
            h=h.gather(1,undo[:,:,None].expand(-1,-1,h.shape[-1]))*valid[:,:,None]
        return self.head(h),self.breath(h).squeeze(-1)

    @property
    def receptive_frames(self):return 101  # BPTT warmup; streaming state itself is unbounded
