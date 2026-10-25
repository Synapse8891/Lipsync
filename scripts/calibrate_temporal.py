"""Fit on development speakers, select on other development speakers; no test use.

The existing validation set becomes development/calibration data. Fresh held-out
speakers remain unused until a final candidate and all hyperparameters are frozen.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch
from kotoba_lipsync.temporal import expand_features
from kotoba_lipsync.model import load_model
from kotoba_lipsync.features import FeatureConfig
if __package__:
    from .tcn_network import TemporalNetwork
    from .gru_network import RecurrentNetwork
    from .train import scores
else:
    from tcn_network import TemporalNetwork
    from gru_network import RecurrentNetwork
    from train import scores

ROOT=Path(__file__).resolve().parents[1]
OFFSETS=(-4,-2,0,1,2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("model",type=Path);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--device",default="cuda")
    parser.add_argument("--index",type=Path,default=Path("cache/features_v2/index_five_hour.json"))
    args=parser.parse_args();threadpool_limits(limits=3);torch.set_num_threads(2)
    path=ROOT/args.model;model=load_model(path);metadata=model.metadata
    if "decoder" in metadata:raise ValueError("Cannot calibrate an already calibrated model")
    signature=hashlib.sha256(path.read_bytes()).hexdigest()
    index_hash=hashlib.sha256((ROOT/args.index).read_bytes()).hexdigest()
    folder=ROOT/"cache/posterior_decoders"/(signature[:16]+"_"+index_hash[:8]);folder.mkdir(parents=True,exist_ok=True)
    cache=folder/"development.npz";lookahead=model.config.lookahead_frames
    if cache.exists():
        with np.load(cache) as data:x=data["x"];y=data["y"];fit=data["fit"];language=data["language"]
    else:
        network=RecurrentNetwork(metadata["channels"],metadata["gru_layers"],skip_lags=metadata.get("skip_lags",()),skip_hidden=metadata.get("skip_hidden",32)) if metadata.get("architecture")=="gru" else TemporalNetwork(metadata["channels"],metadata["dilations"],skip_lags=metadata.get("skip_lags",()),skip_hidden=metadata.get("skip_hidden",64))
        network.load_state_dict({key:torch.from_numpy(value) for key,value in model.state.items()});network.to(args.device).eval()
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        index=json.loads((ROOT/args.index).read_text())
        xs=[];ys=[];fits=[];langs=[]
        with torch.inference_mode():
            for number,row in enumerate(r for r in index["records"] if r["split"]=="validation"):
                with np.load(ROOT/row.get("feature_root","cache/features_v2")/(row["id"]+".npz")) as data:base=data["base"].astype(np.float32);truth=data["y"]
                values=(expand_features(base)-model.mean)/model.scale
                p=network(torch.from_numpy(values)[None].to(args.device))[0][0].softmax(-1).cpu().numpy()[lookahead:]
                target=np.arange(max(0,len(p)-max(OFFSETS)))
                positions=np.maximum(target[:,None]+np.asarray(OFFSETS)[None,:],0)
                xs.append(np.c_[np.log(np.maximum(p[positions],1e-6)).reshape(len(target),-1),base[target,68]])
                ys.append(truth[target]);fits.extend([int(row["speaker_hash"][:8],16)%4!=0]*len(target));langs.extend([row["language"]]*len(target))
                if (number+1)%100==0:print(f"Development posterior cache {number+1}/400",flush=True)
        x=np.concatenate(xs);y=np.concatenate(ys);fit=np.asarray(fits);language=np.asarray(langs)
        np.savez_compressed(cache,x=x,y=y,fit=fit,language=language)
        del network
        if torch.cuda.is_available():torch.cuda.empty_cache()
    scaler=StandardScaler().fit(x[fit]);scaled=scaler.transform(x).astype(np.float32)
    trials=[];best=None
    for c in (.001,.003,.01,.03,.1):
        classifier=LogisticRegression(C=c,max_iter=150,tol=2e-4)
        classifier.fit(scaled[fit],y[fit]);prediction=classifier.predict(scaled[~fit])
        result=scores(y[~fit],prediction);metric=.5*result["speech_accuracy"]+.5*result["macro_f1"]
        trial={"C":c,"selection_metric":metric,"development_comparison":result};trials.append(trial)
        if best is None or metric>best["selection_metric"]:best=trial
        print(f"C={c}: comparison accuracy={result['accuracy']:.4f}, speech={result['speech_accuracy']:.4f}, macroF1={result['macro_f1']:.4f}",flush=True)
    # Refit only development data after choosing C. The fresh test stays unopened.
    scaler=StandardScaler().fit(x);classifier=LogisticRegression(C=best["C"],max_iter=200,tol=2e-4).fit(scaler.transform(x),y)
    with np.load(path) as data:arrays={key:data[key].copy() for key in data.files}
    config=model.config.to_dict();config["lookahead_frames"]=lookahead+max(OFFSETS)
    metadata={**metadata,"feature_config":config,"decoder":{"offsets":list(OFFSETS),"base_lookahead_frames":lookahead,"extra_lookahead_frames":max(OFFSETS),"classes":classifier.classes_.tolist(),"base_sha256":signature,"C":best["C"],"training_data":"previous validation speakers; selected C with speaker-disjoint development comparison; refit on all development data; fresh test untouched"}}
    arrays.update(metadata=np.array(json.dumps(metadata)),decoder_weight=classifier.coef_.astype(np.float32),decoder_bias=classifier.intercept_.astype(np.float32),decoder_mean=scaler.mean_.astype(np.float32),decoder_scale=scaler.scale_.astype(np.float32))
    dest=ROOT/args.output;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
    report={"status":"development only; fresh test not evaluated","base_model_sha256":signature,"output":str(args.output),"extra_lookahead_ms":max(OFFSETS)*10,"trials":trials,"selected":best,"fit_frames":int(fit.sum()),"comparison_frames":int((~fit).sum()),"refit":"all previous validation speakers become calibration training data; development metrics are not test scores"}
    dest.with_suffix(".development.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Calibrated candidate -> {dest}. Fresh test untouched.",flush=True)

if __name__=="__main__":main()
