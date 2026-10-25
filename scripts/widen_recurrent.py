"""Widen a GRU while preserving its initial inference function."""
import argparse
from pathlib import Path
import torch
if __package__:
    from .gru_network import RecurrentNetwork
else:
    from gru_network import RecurrentNetwork


def widen_state(old,channels,skip_hidden,layers=2,skip_lags=()):
    previous=old["project.weight"].shape[0]
    network=RecurrentNetwork(channels,layers,skip_lags=skip_lags,skip_hidden=skip_hidden)
    state=network.state_dict()
    if channels<previous:raise ValueError("Widening cannot remove hidden units")
    with torch.no_grad():
        state["project.weight"][:previous].copy_(old["project.weight"])
        state["project.bias"][:previous].copy_(old["project.bias"])
        for layer in range(layers):
            for name in ("weight_ih","weight_hh"):
                key=f"rnn.{name}_l{layer}"
                for gate in range(3):
                    dest=state[key][gate*channels:gate*channels+previous]
                    dest.zero_();dest[:,:previous].copy_(old[key][gate*previous:(gate+1)*previous])
            for name in ("bias_ih","bias_hh"):
                key=f"rnn.{name}_l{layer}"
                for gate in range(3):state[key][gate*channels:gate*channels+previous].copy_(old[key][gate*previous:(gate+1)*previous])
        for name in ("head","breath"):
            state[name+".weight"].zero_();state[name+".weight"][:,:previous].copy_(old[name+".weight"])
            state[name+".bias"].copy_(old[name+".bias"])
        if skip_lags:
            previous_skip=old["skip_project.weight"].shape[0]
            if skip_hidden<previous_skip:raise ValueError("Cannot shrink the aligned feature branch")
            previous_width=old["skip_project.weight"].shape[1]
            if state["skip_project.weight"].shape[1]<previous_width:raise ValueError("Cannot remove existing context inputs")
            state["skip_project.weight"][:previous_skip].zero_()
            state["skip_project.weight"][:previous_skip,:previous_width].copy_(old["skip_project.weight"])
            state["skip_project.bias"][:previous_skip].copy_(old["skip_project.bias"])
            for name in ("skip_head","skip_breath"):
                state[name+".weight"].zero_();state[name+".weight"][:,:previous_skip].copy_(old[name+".weight"])
                state[name+".bias"].copy_(old[name+".bias"])
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument("source",type=Path);p.add_argument("output",type=Path);p.add_argument("--channels",type=int,default=192);p.add_argument("--skip-hidden",type=int,default=64)
    p.add_argument("--skip-lags",default="0,2,4,6,8,10,12,14")
    args=p.parse_args();torch.manual_seed(42)
    checkpoint=torch.load(args.source,map_location="cpu",weights_only=True)
    checkpoint["model"]=widen_state(checkpoint["model"],args.channels,args.skip_hidden,skip_lags=tuple(int(v) for v in args.skip_lags.split(",")))
    torch.save(checkpoint,args.output)

if __name__=="__main__":main()
