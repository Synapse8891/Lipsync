"""Development-only diagnosis; tolerant scores never count as exact accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from kotoba_lipsync.model import load_model
from kotoba_lipsync.normalization import sequence_features
from scripts.gru_network import RecurrentNetwork
from scripts.network_tools import build_network,infer_sequence


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",default="models/kotoba_int8.npz")
    parser.add_argument("--index",default="cache/features_v2/index_extended_complete_v3.json")
    parser.add_argument("--count",type=int,default=100);parser.add_argument("--output",default="reports/experiments/2026-10-25_85_diagnosis.json")
    args=parser.parse_args();threadpool_limits(2);torch.set_num_threads(2)
    model=load_model(args.model);base=model.base if hasattr(model,"base") else model;meta=base.metadata
    lookahead=meta.get("decoder",{}).get("base_lookahead_frames",base.config.lookahead_frames)
    network=build_network(meta,len(base.mean))
    network.load_state_dict({k:torch.from_numpy(v) for k,v in base.state.items()});network.cuda().eval()
    document=json.loads(Path(args.index).read_text(encoding="utf8"));output={}
    for split in ("train","validation"):
        rows=[r for r in document["records"] if r["split"]==split and r["language"]=="ja"][:args.count]
        exact=[];speech=[];interior=[];tolerant={radius:[] for radius in (1,2,3,5,10)}
        with torch.inference_mode():
            for row in rows:
                with np.load(Path(row.get("feature_root","cache/features_v2"))/(row["id"]+".npz")) as data:features=data["base"].astype(np.float32);truth=data["y"]
                x=(sequence_features(features,meta.get("feature_mode","base"))-base.mean)/base.scale
                probability=infer_sequence(network,x,"cuda")[0][0].softmax(-1).cpu().numpy()[lookahead:]
                if hasattr(model,"offsets"):
                    t=np.arange(len(probability)-model.extra);positions=np.maximum(t[:,None]+np.asarray(model.offsets),0)
                    probability=model._classify(np.c_[np.log(np.maximum(probability[positions],1e-6)).reshape(len(t),-1),features[t,68]])
                predicted=probability.argmax(-1);y=truth[:len(predicted)];hit=predicted==y;exact.extend(hit);speech.extend(hit[y!=0])
                stable=np.ones(len(y),dtype=bool)
                for offset in range(-3,4):stable&=y==y[np.clip(np.arange(len(y))+offset,0,len(y)-1)]
                interior.extend(hit[stable&(y!=0)])
                for radius in tolerant:
                    match=np.zeros(len(y),dtype=bool)
                    for offset in range(-radius,radius+1):match|=predicted==y[np.clip(np.arange(len(y))+offset,0,len(y)-1)]
                    tolerant[radius].extend(match[y!=0])
        output[split]={"recordings":len(rows),"exact_accuracy":float(np.mean(exact)),"exact_speech_accuracy":float(np.mean(speech)),"stable_speech_accuracy_30ms":float(np.mean(interior)),"speech_label_tolerance_diagnostics":{f"plus_minus_{r*10}ms":float(np.mean(v)) for r,v in tolerant.items()}}
        print(split,json.dumps(output[split]),flush=True)
    output["scope"]="Already available development and training data only. Stable/tolerant diagnostics do not replace the exact frame accuracy criterion. No untouched final test used."
    Path(args.output).write_text(json.dumps(output,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":main()
