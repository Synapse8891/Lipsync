"""Choose a breath threshold and quiet-input gate on development clips only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from threadpoolctl import threadpool_limits
from kotoba_lipsync.model import load_model
from kotoba_lipsync.temporal import expand_features
from kotoba_lipsync.normalization import sequence_features
from scripts.network_tools import build_network,infer_sequence
if __package__:
    from .temporal_data import Corpus
    from .gru_network import RecurrentNetwork
    from .tcn_network import TemporalNetwork
else:
    from temporal_data import Corpus
    from gru_network import RecurrentNetwork
    from tcn_network import TemporalNetwork

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument("model",type=Path);p.add_argument("--corpus",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--device",default="cuda")
    args=p.parse_args();torch.set_num_threads(2);threadpool_limits(limits=2)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    model=load_model(ROOT/args.model);metadata=model.metadata
    if "decoder" in metadata or metadata.get("breath_logit_bias",0):raise ValueError("Calibrate the unmodified base model before the posterior decoder")
    if not model.has_breath_head:raise ValueError("No independently trained breath head")
    corpus=Corpus(ROOT/args.corpus);lookahead=model.config.lookahead_frames
    network=build_network(metadata,len(model.mean))
    network.load_state_dict({key:torch.from_numpy(value) for key,value in model.state.items()});network.to(args.device).eval()
    clips=[]
    with torch.inference_mode():
        for row in corpus.rows:
            if row["split"]!="validation":continue
            base,y,br=corpus.arrays(row);x=(sequence_features(base,metadata.get("feature_mode","base"))-model.mean)/model.scale
            raw=infer_sequence(network,x,args.device)[1][0,lookahead:].sigmoid().cpu().numpy()
            active=np.asarray(br[:len(raw)])>=0;db=np.asarray(base[:len(raw),68],dtype=np.float32)*80
            if active.any():clips.append((row["id"],row["language"]=="breath",raw[active],db[active]))
    truths=np.asarray([row[1] for row in clips]);trials=[]
    thresholds=(.005,.01,.02,.03,.05,.075,.1,.15,.2,.3,.4,.5,.6,.7,.8,.9,.95)
    for gate in (-55.,-60.,-65.):
        clip_scores=np.asarray([np.percentile(np.where(row[3]>=gate,row[2],0),90) for row in clips])
        for threshold in thresholds:
            predicted=clip_scores>=threshold
            precision,recall,f1,_=precision_recall_fscore_support(truths,predicted,average="binary",zero_division=0)
            trials.append({"threshold":threshold,"gate_db":gate,"precision":float(precision),"recall":float(recall),"f1":float(f1)})
    peak_f1=max(row["f1"] for row in trials)
    # Prefer fewer false activations when F1 differs by less than one point.
    best=max((row for row in trials if row["f1"]>=peak_f1-.01),key=lambda row:(row["precision"],row["f1"],row["threshold"],row["gate_db"]))
    with np.load(ROOT/args.model) as data:arrays={key:data[key].copy() for key in data.files}
    metadata={**metadata,"breath_logit_bias":float(-np.log(best["threshold"]/(1-best["threshold"]))),"breath_gate_offset_db":best["gate_db"]+55,"breath_calibration":{"selection":"development clips only; source-described weak labels; raw gate before EMA","raw_threshold":best["threshold"],"development_result":best,"positive_clips":int(truths.sum()),"negative_clips":int((~truths).sum())}}
    arrays["metadata"]=np.array(json.dumps(metadata));output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    report={"source_sha256":hashlib.sha256((ROOT/args.model).read_bytes()).hexdigest(),"selected":best,"trials":trials,"data_use":"validation only; no fresh speech or event test read","note":"Scalar strength calibration, not human-labelled frame probability; final runtime metrics require an untouched test"}
    output.with_suffix(".breath_development.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(best),flush=True)

if __name__=="__main__":main()
