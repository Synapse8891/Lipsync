"""Compact causal attention with bounded context and relative distance bias."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AttentionBlock(nn.Module):
    def __init__(self,channels,heads,dropout=.1):
        super().__init__();self.heads=heads;self.width=channels//heads;self.dropout=dropout
        self.norm1=nn.LayerNorm(channels);self.qkv=nn.Linear(channels,channels*3);self.out=nn.Linear(channels,channels)
        self.norm2=nn.LayerNorm(channels);self.ff1=nn.Linear(channels,channels*2);self.ff2=nn.Linear(channels*2,channels)

    def forward(self,x,mask):
        batch,length,channels=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(batch,length,3,self.heads,self.width).permute(2,0,3,1,4).unbind(0)
        attended=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=self.dropout if self.training else 0.)
        attended=attended.transpose(1,2).reshape(batch,length,channels)
        x=x+F.dropout(self.out(attended),self.dropout,self.training)
        return x+F.dropout(self.ff2(F.relu(self.ff1(self.norm2(x)))),self.dropout,self.training)


class AttentionNetwork(nn.Module):
    def __init__(self,channels=128,layers=3,heads=4,context=32,features=158,spectral_frontend=True):
        super().__init__()
        if channels%heads:raise ValueError("Attention channels must be divisible by heads")
        self.layers=layers;self.heads=heads;self.context=context;self.spectral_frontend=spectral_frontend;self.dilations=()
        self.project=nn.Linear(features,channels)
        if spectral_frontend:
            self.conv1=nn.Conv2d(2,8,(5,5),stride=(1,2));self.conv2=nn.Conv2d(8,16,(3,3),stride=(1,2))
            self.spectral_project=nn.Linear(160,channels)
        self.blocks=nn.ModuleList(AttentionBlock(channels,heads) for _ in range(layers))
        self.norm=nn.LayerNorm(channels);self.head=nn.Linear(channels,16);self.breath=nn.Linear(channels,1)

    def forward(self,x,valid=None):
        projected=self.project(x)
        if self.spectral_frontend:
            spectral=F.relu(self.conv1(F.pad(torch.stack((x[:,:,:40],x[:,:,-40:]),dim=1),(2,2,4,0))))
            if valid is not None:spectral=spectral*valid[:,None,:,None]
            spectral=F.relu(self.conv2(F.pad(spectral,(1,1,2,0))))
            projected=projected+self.spectral_project(spectral.permute(0,2,1,3).flatten(2))
        hidden=F.relu(projected)
        position=torch.arange(x.shape[1],device=x.device);distance=position[:,None]-position[None,:]
        slopes=2.**(-torch.arange(1,self.heads+1,device=x.device,dtype=x.dtype))
        mask=-distance.clamp_min(0)[None,None].to(x.dtype)*slopes[None,:,None,None]
        mask=mask.masked_fill(~((distance>=0)&(distance<self.context))[None,None],float("-inf"))
        if valid is not None:mask=mask.masked_fill(~valid[:,None,None,:],float("-inf"))
        for block in self.blocks:hidden=block(hidden,mask)
        hidden=self.norm(hidden)
        return self.head(hidden),self.breath(hidden).squeeze(-1)

    @property
    def receptive_frames(self):return (7 if self.spectral_frontend else 1)+self.layers*(self.context-1)
