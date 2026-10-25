"""Training-only PyTorch network. The deployed implementation uses NumPy."""
import torch
from torch import nn
from torch.nn import functional as F

class TemporalBlock(nn.Module):
    def __init__(self,channels,dilation,dropout=.1):
        super().__init__()
        self.dilation=dilation
        self.conv1=nn.Conv1d(channels,channels,3,dilation=dilation)
        self.conv2=nn.Conv1d(channels,channels,3,dilation=dilation)
        self.norm=nn.LayerNorm(channels)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x,valid=None):
        z=F.relu(self.conv1(F.pad(x,(2*self.dilation,0))))
        if valid is not None:z=z*valid
        z=self.dropout(z)
        z=self.conv2(F.pad(z,(2*self.dilation,0)))
        result=F.relu(self.norm((x+z).transpose(1,2)).transpose(1,2))
        return result if valid is None else result*valid

class TemporalNetwork(nn.Module):
    def __init__(self,channels=64,dilations=(1,2,4,8),features=118,skip_lags=(),skip_hidden=64):
        super().__init__()
        self.channels=channels
        self.dilations=tuple(dilations)
        self.project=nn.Conv1d(features,channels,1)
        self.blocks=nn.ModuleList([TemporalBlock(channels,d) for d in dilations])
        self.head=nn.Conv1d(channels,16,1)
        self.breath=nn.Conv1d(channels,1,1)
        self.skip_lags=tuple(skip_lags)
        if self.skip_lags:
            self.skip_project=nn.Conv1d(features*len(self.skip_lags),skip_hidden,1)
            self.skip_head=nn.Conv1d(skip_hidden,16,1)
            self.skip_breath=nn.Conv1d(skip_hidden,1,1)
            self.skip_dropout=nn.Dropout(.1)
            nn.init.zeros_(self.skip_head.weight);nn.init.zeros_(self.skip_head.bias)
            nn.init.zeros_(self.skip_breath.weight);nn.init.zeros_(self.skip_breath.bias)

    def forward(self,x,valid=None):
        if valid is not None:valid=valid[:,None,:].to(x.dtype)
        h=F.relu(self.project(x.transpose(1,2)))
        if valid is not None:h=h*valid
        for block in self.blocks:h=block(h,valid)
        logits=self.head(h);breath=self.breath(h)
        if self.skip_lags:
            raw=x.transpose(1,2)
            aligned=torch.cat([F.pad(raw,(lag,0))[:,:,:raw.shape[-1]] for lag in self.skip_lags],dim=1)
            direct=self.skip_dropout(F.relu(self.skip_project(aligned)))
            logits=logits+self.skip_head(direct);breath=breath+self.skip_breath(direct)
        return logits.transpose(1,2),breath.squeeze(1)

    @property
    def receptive_frames(self):return max(1+4*sum(self.dilations),1+max(self.skip_lags,default=0))
