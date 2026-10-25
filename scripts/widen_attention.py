"""Duplicate attention units; exact for integer width multiples, warm start otherwise.

Noninteger expansion changes LayerNorm statistics and must be evaluated again.
"""
import argparse
from pathlib import Path
import torch
from scripts.attention_network import AttentionNetwork


def _mapping(previous,current,seed):
    if current<previous:raise ValueError("Cannot remove existing units")
    count,remainder=divmod(current,previous)
    order=torch.arange(previous).repeat(count)
    if remainder:order=torch.cat((order,torch.randperm(previous,generator=torch.Generator().manual_seed(seed))[:remainder]))
    return order,torch.bincount(order,minlength=previous).to(torch.float32)


def widen_attention_state(old,heads=4,context=32,channels=None):
    previous=old["project.weight"].shape[0];channels=channels or previous*2
    if channels%heads:raise ValueError("Channels must be divisible by heads")
    old_width=previous//heads;width=channels//heads
    layers=max(int(key.split(".")[1]) for key in old if key.startswith("blocks."))+1
    network=AttentionNetwork(channels,layers,heads,context,features=old["project.weight"].shape[1],spectral_frontend="conv1.weight" in old)
    result=network.state_dict();hidden,hcount=_mapping(previous,channels,51);head,dcount=_mapping(old_width,width,52)
    ff,fcount=_mapping(previous*2,channels*2,53)
    attention=(torch.arange(heads)[:,None]*old_width+head[None,:]).reshape(-1)
    attention_counts=dcount.repeat(heads)[attention]
    with torch.no_grad():
        for name in ("project","spectral_project"):
            if name+".weight" in old:
                result[name+".weight"].copy_(old[name+".weight"][hidden]);result[name+".bias"].copy_(old[name+".bias"][hidden])
        for name in ("conv1","conv2"):
            if name+".weight" in old:
                result[name+".weight"].copy_(old[name+".weight"]);result[name+".bias"].copy_(old[name+".bias"])
        for layer in range(layers):
            prefix=f"blocks.{layer}."
            for norm in ("norm1","norm2"):
                for suffix in ("weight","bias"):result[prefix+norm+"."+suffix].copy_(old[prefix+norm+"."+suffix][hidden])
            qscale=(width/old_width)**.25/torch.sqrt(dcount[head])
            factor=torch.stack((qscale,qscale,torch.ones_like(qscale)))
            weight=old[prefix+"qkv.weight"].reshape(3,heads,old_width,previous)[:,:,head,:][:,:,:,hidden]/hcount[hidden]
            result[prefix+"qkv.weight"].copy_((weight*factor[:,None,:,None]).reshape(channels*3,channels))
            bias=old[prefix+"qkv.bias"].reshape(3,heads,old_width)[:,:,head]
            result[prefix+"qkv.bias"].copy_((bias*factor[:,None,:]).reshape(-1))
            result[prefix+"out.weight"].copy_(old[prefix+"out.weight"][hidden][:,attention]/attention_counts)
            result[prefix+"out.bias"].copy_(old[prefix+"out.bias"][hidden])
            result[prefix+"ff1.weight"].copy_(old[prefix+"ff1.weight"][ff][:,hidden]/hcount[hidden])
            result[prefix+"ff1.bias"].copy_(old[prefix+"ff1.bias"][ff])
            result[prefix+"ff2.weight"].copy_(old[prefix+"ff2.weight"][hidden][:,ff]/fcount[ff])
            result[prefix+"ff2.bias"].copy_(old[prefix+"ff2.bias"][hidden])
        for suffix in ("weight","bias"):result["norm."+suffix].copy_(old["norm."+suffix][hidden])
        for name in ("head","breath"):
            result[name+".weight"].copy_(old[name+".weight"][:,hidden]/hcount[hidden]);result[name+".bias"].copy_(old[name+".bias"])
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("source",type=Path);parser.add_argument("output",type=Path);parser.add_argument("--heads",type=int,default=4);parser.add_argument("--channels",type=int)
    args=parser.parse_args();checkpoint=torch.load(args.source,map_location="cpu",weights_only=True)
    previous=checkpoint["model"]["project.weight"].shape[0]
    checkpoint["model"]=widen_attention_state(checkpoint["model"],args.heads,channels=args.channels);args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(checkpoint,args.output)
    current=checkpoint["model"]["project.weight"].shape[0]
    print(f"Expanded attention {previous} -> {current}; "+("initial function preserved" if current%previous==0 else "LayerNorm changes; warm start only, requires retraining and evaluation"))


if __name__=="__main__":main()
