"""Add acoustic input channels without changing an existing GRU's function."""
import argparse
import json
from pathlib import Path
import torch


def expand_state(state,features,skip_lags=()):
    old_features=state["project.weight"].shape[1]
    if features<old_features:raise ValueError("Cannot remove existing acoustic features")
    output={name:value.clone() for name,value in state.items()}
    project=state["project.weight"]
    output["project.weight"]=project.new_zeros((project.shape[0],features))
    output["project.weight"][:,:old_features]=project
    if "skip_project.weight" in state:
        if not skip_lags:raise ValueError("Skip lags are required to preserve temporal feature blocks")
        old=state["skip_project.weight"]
        if old.shape[1]!=len(skip_lags)*old_features:raise ValueError("Skip feature dimensions do not match")
        expanded=old.new_zeros((old.shape[0],len(skip_lags),features))
        expanded[:,:,:old_features]=old.reshape(old.shape[0],len(skip_lags),old_features)
        output["skip_project.weight"]=expanded.reshape(old.shape[0],-1)
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("checkpoint",type=Path);parser.add_argument("--corpus",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True);parser.add_argument("--skip-lags",default="0,2,4,6,8,10,12,14")
    args=parser.parse_args();checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    metadata=json.loads((args.corpus/"corpus.json").read_text(encoding="utf8"));features=len(metadata["mean"])
    previous=len(checkpoint["mean"]);lags=tuple(int(value) for value in args.skip_lags.split(",") if value)
    checkpoint["model"]=expand_state(checkpoint["model"],features,lags)
    for name in ("mean","scale"):
        values=torch.tensor(metadata[name],dtype=torch.float32);values[:previous]=checkpoint[name];checkpoint[name]=values
    checkpoint["feature_mode"]=metadata.get("feature_mode","base")
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(checkpoint,args.output)
    print(f"Expanded acoustic features {previous} -> {features}; existing function preserved.")


if __name__=="__main__":main()
