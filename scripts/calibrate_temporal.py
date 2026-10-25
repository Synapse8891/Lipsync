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
from kotoba_lipsync.normalization import sequence_features
from scripts.network_tools import build_network,infer_sequence
from scripts.fit_decoder import fit_linear
if __package__:
    from .tcn_network import TemporalNetwork
    from .gru_network import RecurrentNetwork
    from .train import scores
else:
    from tcn_network import TemporalNetwork
    from gru_network import RecurrentNetwork
    from train import scores

ROOT=Path(__file__).resolve().parents[1]


def decoder_feature_config(config,extra):
    result=config.to_dict();result["lookahead_frames"]+=extra
    result["context"]=list(result["context"])
    if result["lookahead_frames"]>max(result["context"]):result["context"]=sorted(set(result["context"]+[result["lookahead_frames"]]))
    FeatureConfig(**result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("model",type=Path);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--device",default="cuda")
    parser.add_argument("--index",type=Path,default=Path("cache/features_v2/index_five_hour.json"))
    parser.add_argument("--offsets",default="-4,-2,0,1,2")
    parser.add_argument("--embedding-features",action="store_true")
    parser.add_argument("--acoustic-features",action="store_true")
    parser.add_argument("--torch-fit",action="store_true")
    parser.add_argument("--selection-language",choices=("all","ja","en"),default="all")
    parser.add_argument("--selection-metric",choices=("balanced","accuracy"),default="balanced")
    parser.add_argument("--english-weight",type=float,default=1.)
    args=parser.parse_args();threadpool_limits(limits=3);torch.set_num_threads(2)
    offsets=tuple(int(value) for value in args.offsets.split(","))
    if 0 not in offsets:parser.error("Decoder offsets must include the current posterior")
    path=ROOT/args.model;model=load_model(path);metadata=model.metadata
    if "decoder" in metadata:raise ValueError("Cannot calibrate an already calibrated model")
    if args.embedding_features and metadata.get("architecture")!="attention":raise ValueError("Embedding capture requires attention")
    if args.english_weight<=0:raise ValueError("English weight must be positive")
    signature=hashlib.sha256(path.read_bytes()).hexdigest()
    index_hash=hashlib.sha256((ROOT/args.index).read_bytes()).hexdigest()
    feature_key=args.offsets+(f"/embedding={args.embedding_features}/acoustic={args.acoustic_features}" if args.embedding_features or args.acoustic_features else "")
    offset_hash=hashlib.sha256(feature_key.encode()).hexdigest()[:8]
    folder=ROOT/"cache/posterior_decoders"/(signature[:16]+"_"+index_hash[:8]+"_"+offset_hash);folder.mkdir(parents=True,exist_ok=True)
    cache=folder/"development.npz";lookahead=model.config.lookahead_frames
    if cache.exists():
        with np.load(cache) as data:x=data["x"];y=data["y"];fit=data["fit"];language=data["language"]
    else:
        network=build_network(metadata,len(model.mean))
        network.load_state_dict({key:torch.from_numpy(value) for key,value in model.state.items()});network.to(args.device).eval()
        torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
        index=json.loads((ROOT/args.index).read_text(encoding="utf8"))
        xs=[];ys=[];fits=[];langs=[]
        development_rows=[r for r in index["records"] if r["split"]=="validation"]
        with torch.inference_mode():
            for number,row in enumerate(development_rows):
                with np.load(ROOT/row.get("feature_root","cache/features_v2")/(row["id"]+".npz")) as data:base=data["base"].astype(np.float32);truth=data["y"]
                values=(sequence_features(base,metadata.get("feature_mode","base"))-model.mean)/model.scale
                inferred=infer_sequence(network,values,args.device,capture_embedding=args.embedding_features)
                p=inferred[0][0].softmax(-1).cpu().numpy()[lookahead:]
                target=np.arange(max(0,len(p)-max(offsets)))
                positions=np.maximum(target[:,None]+np.asarray(offsets)[None,:],0)
                parts=[np.log(np.maximum(p[positions],1e-6)).reshape(len(target),-1),base[target,68:69]]
                if args.embedding_features:parts.append(inferred[2][0,lookahead:].cpu().numpy()[target])
                if args.acoustic_features:parts.append(expand_features(base[target]))
                xs.append(np.concatenate(parts,axis=1).astype(np.float32))
                ys.append(truth[target]);fits.extend([int(row["speaker_hash"][:8],16)%4!=0]*len(target));langs.extend([row["language"]]*len(target))
                if (number+1)%100==0:print(f"Development posterior cache {number+1}/{len(development_rows)}",flush=True)
        x=np.concatenate(xs);y=np.concatenate(ys);fit=np.asarray(fits);language=np.asarray(langs)
        del xs,ys,fits,langs
        np.savez_compressed(cache,x=x,y=y,fit=fit,language=language)
        del network
        if torch.cuda.is_available():torch.cuda.empty_cache()
    scaler=StandardScaler().fit(x[fit]);scaled=scaler.transform(x).astype(np.float32,copy=False)
    weight=np.where(language=="en",args.english_weight,1.).astype(np.float32)
    selected_language=np.ones(len(y),dtype=bool) if args.selection_language=="all" else language==args.selection_language
    comparison=~fit & selected_language
    if not comparison.any():raise ValueError("No comparison data in the requested language")
    def fit_classifier(values,truth,c,weights):
        if args.torch_fit:return fit_linear(values,truth,c,sample_weight=weights,device=args.device)
        return LogisticRegression(C=c,max_iter=200,tol=2e-4).fit(values,truth,sample_weight=weights)
    trials=[];best=None
    for c in (.001,.003,.01,.03,.1):
        classifier=fit_classifier(scaled[fit],y[fit],c,weight[fit]);prediction=classifier.predict(scaled[comparison])
        result=scores(y[comparison],prediction);metric=result["accuracy"] if args.selection_metric=="accuracy" else .5*result["speech_accuracy"]+.5*result["macro_f1"]
        trial={"C":c,"selection_metric":metric,"development_comparison":result};trials.append(trial)
        if best is None or metric>best["selection_metric"]:best=trial
        print(f"C={c}: comparison accuracy={result['accuracy']:.4f}, speech={result['speech_accuracy']:.4f}, macroF1={result['macro_f1']:.4f}",flush=True)
    # Refit only development data after choosing C. The fresh test stays unopened.
    del scaled
    scaler=StandardScaler().fit(x);classifier=fit_classifier(scaler.transform(x).astype(np.float32,copy=False),y,best["C"],weight)
    with np.load(path) as data:arrays={key:data[key].copy() for key in data.files}
    config=decoder_feature_config(model.config,max(offsets))
    metadata={**metadata,"feature_config":config,"decoder":{"offsets":list(offsets),"base_lookahead_frames":lookahead,"extra_lookahead_frames":max(offsets),"classes":classifier.classes_.tolist(),"base_sha256":signature,"C":best["C"],"training_data":"previous validation speakers; selected C with speaker-disjoint development comparison; refit on all development data; fresh test untouched"}}
    metadata["decoder"].update(embedding_features=args.embedding_features,acoustic_features=args.acoustic_features,feature_cache=str(cache.relative_to(ROOT)).replace("\\","/"),solver="torch-lbfgs-f32" if args.torch_fit else "sklearn-lbfgs",selection_language=args.selection_language,selection_metric=args.selection_metric,english_weight=args.english_weight)
    arrays.update(metadata=np.array(json.dumps(metadata)),decoder_weight=classifier.coef_.astype(np.float32),decoder_bias=classifier.intercept_.astype(np.float32),decoder_mean=scaler.mean_.astype(np.float32),decoder_scale=scaler.scale_.astype(np.float32))
    dest=ROOT/args.output;dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**arrays)
    report={"status":"development only; fresh test not evaluated","base_model_sha256":signature,"output":str(args.output),"offsets":list(offsets),"extra_lookahead_ms":max(offsets)*10,"trials":trials,"selected":best,"fit_frames":int(fit.sum()),"comparison_frames":int((~fit).sum()),"refit":"all previous validation speakers become calibration training data; development metrics are not test scores"}
    report.update(decoder_features=x.shape[1],embedding_features=args.embedding_features,acoustic_features=args.acoustic_features,selection_language=args.selection_language,selected_comparison_frames=int(comparison.sum()),english_weight=args.english_weight,solver=metadata["decoder"]["solver"])
    dest.with_suffix(".development.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Calibrated candidate -> {dest}. Fresh test untouched.",flush=True)

if __name__=="__main__":main()
