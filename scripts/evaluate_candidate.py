"""One final, frozen-candidate evaluation on previously unused speech and events."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.metrics import precision_recall_fscore_support
from kotoba_lipsync.model import load_model
if __package__:
    from .train import scores
    from .temporal_data import breath_features
else:
    from train import scores
    from temporal_data import breath_features

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("model",type=Path);parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--fresh-index",type=Path,default=Path("cache/features_fresh_v2/index.json"))
    parser.add_argument("--development-index",type=Path,default=Path("cache/features_v2/index_five_hour.json"))
    parser.add_argument("--breath-manifest",type=Path,default=Path("data/breath_cc0/manifest.json"))
    args=parser.parse_args();threadpool_limits(limits=3)
    model=load_model(ROOT/args.model);lookahead=model.config.lookahead_frames
    fresh=json.loads((ROOT/args.fresh_index).read_text())
    previous=json.loads((ROOT/args.development_index).read_text())
    for field in ("id","speaker_hash","transcript_hash"):
        if {r[field] for r in fresh["records"]}&{r[field] for r in previous["records"]}:raise ValueError("Fresh holdout overlap: "+field)
    if fresh["annotation_signatures"]!=previous["annotation_signatures"]:raise ValueError("Teacher signature mismatch")
    output=ROOT/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise RuntimeError("Use a new report path; never overwrite an evaluation attempt")
    model_hash=hashlib.sha256((ROOT/args.model).read_bytes()).hexdigest()
    marker=output.with_suffix(".started.json")
    if marker.exists():raise RuntimeError("This evaluation attempt already started; inspect its record")
    marker.write_text(json.dumps({"model":str(args.model),"sha256":model_hash,"time":time.time(),"note":"Candidate frozen before fresh test inference"},indent=2),encoding="utf-8")
    truths=[];preds=[];languages=[];runtime_preds=[];breath_clips=[];per_record=[]

    def process(ident,base,y,br,language):
        probability,breath=model.predict_sequence(base)
        probability=probability[lookahead:];breath=breath[lookahead:]
        y=np.asarray(y[:len(probability)]);br=np.asarray(br[:len(probability)])
        db=np.asarray(base[:len(probability),68],dtype=np.float32)*80
        state=np.zeros(16,dtype=np.float32);state[0]=1;breath_state=0.
        online=[];online_breath=[];smoothing=model.metadata.get("recommended_smoothing_ms",25.)
        for p,b,level in zip(probability,breath,db):
            target=p if level>=-55 else np.eye(16,dtype=np.float32)[0]
            tau=smoothing*(.6 if target[1]>state[1] else 1.)
            alpha=1. if tau<=0 else 1-np.exp(-10/tau)
            state+=alpha*(target-state);state=np.clip(state,0,1);state/=max(float(state.sum()),1e-9)
            breath_state+=alpha*((float(b) if level>=-55 else 0.)-breath_state)
            online.append(int(state.argmax()));online_breath.append(breath_state)
        if language in ("ja","en"):
            truths.append(y);preds.append(probability.argmax(-1));languages.extend([language]*len(y));runtime_preds.extend(online)
            per_record.append({"id":ident,"language":language,"frames":len(y),"correct":int((probability.argmax(-1)==y).sum())})
        active=br>=0
        if active.any():
            breath_clips.append({"id":ident,"positive":language=="breath","kind":language,"probability":float(np.percentile(breath[active],90)),"runtime_probability":float(np.percentile(np.asarray(online_breath)[active],90))})

    for number,row in enumerate(fresh["records"]):
        with np.load(ROOT/args.fresh_index.parent/(row["id"]+".npz")) as data:base=data["base"];y=data["y"]
        process(row["id"],base,y,np.where(y!=0,0,-1),row["language"])
        if (number+1)%50==0:print(f"Fresh speech evaluated {number+1}/{len(fresh['records'])}",flush=True)
    manifest=json.loads((ROOT/args.breath_manifest).read_text(encoding="utf-8"))
    for row in manifest["records"]:
        if row["split"]=="test":process(row["id"],*breath_features(row),row["kind"])
    y=np.concatenate(truths);p=np.concatenate(preds);ll=np.asarray(languages);online=np.asarray(runtime_preds)
    speech={"all":scores(y,p),**{lang:scores(y[ll==lang],p[ll==lang]) for lang in ("ja","en")}}
    runtime={"all":scores(y,online),**{lang:scores(y[ll==lang],online[ll==lang]) for lang in ("ja","en")}}
    yy=np.asarray([r["positive"] for r in breath_clips])
    breathing={}
    for field in ("probability","runtime_probability"):
        pp=np.asarray([r[field]>=.5 for r in breath_clips]);precision,recall,f1,_=precision_recall_fscore_support(yy,pp,average="binary",zero_division=0)
        breathing[field]={"precision":float(precision),"recall":float(recall),"f1":float(f1),"true_positive_clips":int((yy&pp).sum()),"false_negative_clips":int((yy&~pp).sum()),"false_positive_clips":int((~yy&pp).sum()),"positive_clips":int(yy.sum()),"negative_clips":int((~yy).sum()),"speech_false_positive_clips":sum(not r["positive"] and r["kind"] in ("ja","en") and r[field]>=.5 for r in breath_clips),"noise_false_positive_clips":sum(r["kind"]=="noise" and r[field]>=.5 for r in breath_clips)}
    report={"model":str(args.model),"sha256":model_hash,"parameters":model.parameters,"model_bytes":(ROOT/args.model).stat().st_size,"test":speech,"runtime_test":runtime,"natural_breath":breathing,"breath_clip_scores":breath_clips,"per_record":per_record,"model_selection":"Candidate, C and weights frozen before this evaluation; previous validation is calibration training data; no fresh labels used for fitting","fresh_overlap_check":"No recording ID, speaker hash or transcript hash shared with any previous split","evaluation_kind":"Agreement with automatic CTC labels, not human-verified mouth shapes; breathing uses weak clip labels from uploader descriptions. Natural breaths within speech are not exhaustively annotated.","breath_decision":"90th percentile over source activity / non-silence teacher labels; fixed threshold .5; raw and actual gate/smoothing reported separately","speech_recordings":len(fresh["records"]),"data_pipeline":fresh["pipeline_version"],"feature_config":model.config.to_dict(),"smoothing_ms":model.metadata.get("recommended_smoothing_ms",25.)}
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"accuracy":speech["all"]["accuracy"],"speech_accuracy":speech["all"]["speech_accuracy"],"runtime_accuracy":runtime["all"]["accuracy"],"breath":breathing},indent=2),flush=True)

if __name__=="__main__":main()
