"""Small nonlinear residual over a fixed posterior decoder; development only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.fit_decoder import fit_linear
if __package__:
    from .train import scores
else:
    from train import scores

ROOT=Path(__file__).resolve().parents[1]


class Residual(nn.Module):
    def __init__(self,width,hidden):
        super().__init__();self.first=nn.Linear(width,hidden);self.last=nn.Linear(hidden,16);self.dropout=nn.Dropout(.1)
        nn.init.zeros_(self.last.weight);nn.init.zeros_(self.last.bias)
    def forward(self,x):return self.last(self.dropout(F.relu(self.first(x))))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("model",type=Path);parser.add_argument("--index",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--epochs",type=int,default=45);parser.add_argument("--hidden",default="32,64")
    args=parser.parse_args();threadpool_limits(limits=3);torch.set_num_threads(2)
    with np.load(ROOT/args.model) as data:arrays={key:data[key].copy() for key in data.files}
    metadata=json.loads(str(arrays["metadata"]));config=metadata["decoder"]
    ih=hashlib.sha256((ROOT/args.index).read_bytes()).hexdigest()[:8];offsets=",".join(map(str,config["offsets"]));oh=hashlib.sha256(offsets.encode()).hexdigest()[:8]
    cache=ROOT/config["feature_cache"] if "feature_cache" in config else ROOT/"cache/posterior_decoders"/(config["base_sha256"][:16]+"_"+ih+"_"+oh)/"development.npz"
    with np.load(cache) as data:x=data["x"];y=data["y"];fit=data["fit"];language=data["language"]
    comparison=~fit & (True if config.get("selection_language","all")=="all" else language==config["selection_language"])
    weight=np.where(language=="en",config.get("english_weight",1.),1.).astype(np.float32)
    scaler=StandardScaler().fit(x[fit]);scaled=scaler.transform(x).astype(np.float32)
    device="cuda" if torch.cuda.is_available() else "cpu"
    linear=fit_linear(scaled[fit],y[fit],config["C"],sample_weight=weight[fit],device=device) if config.get("solver")=="torch-lbfgs-f32" else LogisticRegression(C=config["C"],max_iter=200,tol=2e-4).fit(scaled[fit],y[fit],sample_weight=weight[fit])
    if linear.classes_.tolist()!=list(range(16)):raise ValueError("Development fit is missing a class")
    train_x=torch.from_numpy(scaled[fit]).to(device);train_y=torch.from_numpy(y[fit].astype(np.int64)).to(device);valid_x=torch.from_numpy(scaled[comparison]).to(device);train_weight=torch.from_numpy(weight[fit]).to(device)
    w=torch.tensor(linear.coef_,dtype=torch.float32,device=device);b=torch.tensor(linear.intercept_,dtype=torch.float32,device=device)
    selected=None;trials=[]
    for hidden in map(int,args.hidden.split(",")):
        torch.manual_seed(73);network=Residual(x.shape[1],hidden).to(device);optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.01)
        best=-1.;best_epoch=0;best_result=None
        for epoch in range(1,args.epochs+1):
            network.train();order=torch.randperm(len(train_x),device=device)
            for indices in order.split(8192):
                batch=train_x[indices];logits=batch@w.T+b+network(batch)
                loss=(F.cross_entropy(logits,train_y[indices],label_smoothing=.01,reduction="none")*train_weight[indices]).sum()/train_weight[indices].sum()
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            network.eval()
            with torch.inference_mode():prediction=(valid_x@w.T+b+network(valid_x)).argmax(-1).cpu().numpy()
            result=scores(y[comparison],prediction);metric=result["accuracy"] if config.get("selection_metric")=="accuracy" else .5*result["speech_accuracy"]+.5*result["macro_f1"]
            if metric>best:best=metric;best_epoch=epoch;best_result=result
            if epoch%5==0:print(f"Residual {hidden} epoch {epoch}: comparison={result['accuracy']:.4f}, best={best_epoch}",flush=True)
            if epoch-best_epoch>=8:break
        trial={"hidden":hidden,"epoch":best_epoch,"selection_metric":best,"comparison":best_result};trials.append(trial)
        if selected is None or best>selected["selection_metric"]:selected=trial
    # Refit only development data for the selected number of epochs. The fixed
    # linear arrays already contain the all-development refit from the prior step.
    del train_x,train_y,valid_x,train_weight
    all_x=torch.from_numpy(((x-arrays["decoder_mean"])/arrays["decoder_scale"]).astype(np.float32)).to(device);all_y=torch.from_numpy(y.astype(np.int64)).to(device);all_weight=torch.from_numpy(weight).to(device)
    w=torch.from_numpy(arrays["decoder_weight"]).to(device);b=torch.from_numpy(arrays["decoder_bias"]).to(device)
    torch.manual_seed(73);network=Residual(x.shape[1],selected["hidden"]).to(device);optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=.01)
    for epoch in range(selected["epoch"]):
        network.train();order=torch.randperm(len(all_x),device=device)
        for indices in order.split(8192):
            batch=all_x[indices];loss=(F.cross_entropy(batch@w.T+b+network(batch),all_y[indices],label_smoothing=.01,reduction="none")*all_weight[indices]).sum()/all_weight[indices].sum()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
    for key,value in (("w0",network.first.weight),("b0",network.first.bias),("w1",network.last.weight),("b1",network.last.bias)):arrays["decoder_residual_"+key]=value.detach().cpu().numpy()
    metadata["decoder"]["nonlinear_refinement"]={"hidden":selected["hidden"],"epochs":selected["epoch"],"selection":"speaker-separated development comparison; refit only development data; fresh test untouched"}
    arrays["metadata"]=np.array(json.dumps(metadata));output=ROOT/args.output;np.savez_compressed(output,**arrays)
    output.with_suffix(".refinement.json").write_text(json.dumps({"trials":trials,"selected":selected,"test_evaluated":False},indent=2),encoding="utf-8")
    print("Selected residual",selected["hidden"],selected["epoch"],selected["comparison"]["accuracy"],flush=True)

if __name__=="__main__":main()
