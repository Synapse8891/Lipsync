"""Development-only test of bounded-lag temporal priors; no runtime promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from scripts.fit_decoder import fit_linear
from scripts.train import scores

ROOT=Path(__file__).resolve().parents[1]


def smooth(p,transition,power,lag):
    emission=np.maximum(p,1e-6)**power
    forward=np.empty_like(emission);previous=np.ones(16,dtype=np.float32)/16
    for i,e in enumerate(emission):
        previous=(previous@transition)*e;previous/=max(previous.sum(),1e-30);forward[i]=previous
    if not lag:return forward
    if lag==1:back=emission[1:]@transition.T
    elif lag==2:back=(emission[1:-1]*(emission[2:]@transition.T))@transition.T
    else:raise ValueError("Diagnostic lag must be 0, 1 or 2")
    result=forward[:-lag]*back
    return result/np.maximum(result.sum(axis=1,keepdims=True),1e-30)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("model",type=Path);parser.add_argument("--index",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--device",default="cuda")
    args=parser.parse_args();threadpool_limits(2)
    import torch
    torch.set_num_threads(2)
    with np.load(ROOT/args.model) as data:metadata=json.loads(str(data["metadata"]))
    config=metadata["decoder"]
    with np.load(ROOT/config["feature_cache"]) as data:x=data["x"];y=data["y"];fit=data["fit"];language=data["language"]
    index_raw=(ROOT/args.index).read_bytes();index=json.loads(index_raw)
    rows=[r for r in index["records"] if r["split"]=="validation"]
    lookahead=config["base_lookahead_frames"]+config["extra_lookahead_frames"]
    lengths=[max(0,r["frames"]-lookahead) for r in rows]
    if sum(lengths)!=len(x):raise ValueError("Cannot recover recording boundaries from this cache/index")
    boundaries=np.r_[0,np.cumsum(lengths)]
    comparison=[]
    for row,left,right in zip(rows,boundaries[:-1],boundaries[1:]):
        if right-left<=2:continue
        expected=int(row["speaker_hash"][:8],16)%4!=0
        if not np.all(fit[left:right]==expected):raise ValueError("Development speaker split differs")
        if not expected and row["language"]=="ja":comparison.append((int(left),int(right)))
    scaler=StandardScaler().fit(x[fit]);scaled=scaler.transform(x).astype(np.float32,copy=False)
    weight=np.where(language=="en",config.get("english_weight",1.),1.).astype(np.float32)
    classifier=fit_linear(scaled[fit],y[fit],config["C"],sample_weight=weight[fit],device=args.device)
    if classifier.classes_.tolist()!=list(range(16)):raise ValueError("Missing calibration class")
    pairs=[]
    for left,right in comparison:
        logits=scaled[left:right]@classifier.coef_.T+classifier.intercept_;logits-=logits.max(axis=1,keepdims=True)
        p=np.exp(logits);p/=p.sum(axis=1,keepdims=True);pairs.append((y[left:right],p))
    del x,scaled
    counts=np.full((16,16),.5,dtype=np.float64);recordings=0
    for row in index["records"]:
        if row["split"]!="train" or row["language"]!="ja" or "augmentation" in row:continue
        with np.load(ROOT/row.get("feature_root","cache/features_v2")/(row["id"]+".npz")) as data:truth=data["y"]
        valid=(truth[:-1]>=0)&(truth[1:]>=0)&(truth[:-1]<16)&(truth[1:]<16)
        counts+=np.bincount((truth[:-1][valid]*16+truth[1:][valid]).astype(np.int64),minlength=256).reshape(16,16);recordings+=1
    transition=counts/counts.sum(axis=1,keepdims=True)
    truth=np.concatenate([yy[:-2] for yy,p in pairs]);baseline=scores(truth,np.concatenate([p[:-2].argmax(1) for yy,p in pairs]))
    trials=[]
    for gamma in (0.,.05,.1,.2,.4,.8):
        matrix=transition**gamma;matrix=(matrix/matrix.sum(axis=1,keepdims=True)).astype(np.float32)
        for power in (1.,2.):
            for lag in (0,1,2):
                prediction=np.concatenate([smooth(p,matrix,power,lag)[:len(yy)-2].argmax(1) for yy,p in pairs])
                result=scores(truth,prediction)
                trials.append({"transition_power":gamma,"emission_power":power,"lag_frames":lag,"accuracy":result["accuracy"],"speech_accuracy":result["speech_accuracy"]})
        print("Transition power",gamma,"best development accuracy",max(t["accuracy"] for t in trials),flush=True)
    report={"status":"development-only diagnostic; no final test, runtime implementation or promotion","model_sha256":hashlib.sha256((ROOT/args.model).read_bytes()).hexdigest(),"training_index_sha256":hashlib.sha256(index_raw).hexdigest(),"transition_training_original_japanese_recordings":recordings,"comparison_clips":len(pairs),"comparison_scope":"Japanese speakers excluded from the calibration fit; all trials and baseline omit the same final two cached frames per clip for coverage equality","baseline":baseline,"trials":trials,"selected":max(trials,key=lambda t:t["accuracy"])}
    dest=ROOT/args.output;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":main()
