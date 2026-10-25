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
    model=load_model(ROOT/args.model);metadata=model.metadata
    if "decoder" in metadata or metadata.get("breath_logit_bias",0):raise ValueError("Calibrate the unmodified base model before the posterior decoder")
    if not model.has_breath_head:raise ValueError("No independently trained breath head")
    corpus=Corpus(ROOT/args.corpus);lookahead=model.config.lookahead_frames
    if metadata.get("architecture")=="gru":network=RecurrentNetwork(metadata["channels"],metadata["gru_layers"],skip_lags=metadata.get("skip_lags",()),skip_hidden=metadata.get("skip_hidden",32))
    else:network=TemporalNetwork(metadata["channels"],metadata["dilations"],skip_lags=metadata.get("skip_lags",()),skip_hidden=metadata.get("skip_hidden",64))
    network.load_state_dict({key:torch.from_numpy(value) for key,value in model.state.items()});network.to(args.device).eval()
    clips=[]
    with torch.inference_mode():
        for row in corpus.rows:
            if row["split"]!="validation":continue
            base,y,br=corpus.arrays(row);x=(expand_features(base)-model.mean)/model.scale
            raw=network(torch.from_numpy(x)[None].to(args.device))[1][0,lookahead:].sigmoid().cpu().numpy()
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
    best=max(trials,key=lambda row:(row["f1"],row["precision"],row["gate_db"]))
    with np.load(ROOT/args.model) as data:arrays={key:data[key].copy() for key in data.files}
    metadata={**metadata,"breath_logit_bias":float(-np.log(best["threshold"]/(1-best["threshold"]))),"breath_gate_offset_db":best["gate_db"]+55,"breath_calibration":{"selection":"development clips only; source-described weak labels; raw gate before EMA","raw_threshold":best["threshold"],"development_result":best,"positive_clips":int(truths.sum()),"negative_clips":int((~truths).sum())}}
    arrays["metadata"]=np.array(json.dumps(metadata));output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(output,**arrays)
    report={"source_sha256":hashlib.sha256((ROOT/args.model).read_bytes()).hexdigest(),"selected":best,"trials":trials,"data_use":"validation only; no fresh speech or event test read","note":"Scalar strength calibration, not human-labelled frame probability; final runtime metrics require an untouched test"}
    output.with_suffix(".breath_development.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(best),flush=True)

if __name__=="__main__":main()
