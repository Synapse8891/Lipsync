"""Record actual development progress without treating it as a final score."""
import argparse
from collections import Counter
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--active-run",default=None)
    args=parser.parse_args()
    config=json.loads((ROOT/"configs/accuracy85.json").read_text(encoding="utf8"))
    actual=hashlib.sha256((ROOT/"models/kotoba_int8.npz").read_bytes()).hexdigest()
    if actual!=config["baseline_sha256"]:raise ValueError("Production changed; review the status writer before updating")
    result={"updated_at_jst":datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),"status":"training and development; 85 percent not verified","target_japanese_agreement":config["target_exact_agreement"],"production_unchanged_sha256":actual,"final_test_scored":False,"candidate_runs":{},"active_run":args.active_run}
    for run in config["candidate_runs"]:
        path=ROOT/"cache/tcn_experiments"/run/"history.json"
        if not path.exists():continue
        history=json.loads(path.read_text(encoding="utf8"))
        if not history:continue
        best=max(history,key=lambda r:r["selection_metric"]);validation=best["validation"]
        result["candidate_runs"][run]={"completed_epochs":history[-1]["epoch"],"selected_epoch":best["epoch"],"development_japanese_agreement":validation["ja"]["accuracy"],"development_speech_accuracy":validation["ja"]["speech_accuracy"],"development_breath_f1":validation.get("natural_breath",{}).get("f1"),"final_test_scored":False}
    path=ROOT/config.get("complete_training_index","cache/features_v2/index_ja85_stage3_20261025.json")
    if path.exists():
        raw=path.read_bytes();rows=json.loads(raw)["records"];counts=Counter((r["language"],r["split"]) for r in rows)
        result["training_data"]={"index":str(path.relative_to(ROOT)).replace("\\","/"),"sha256":hashlib.sha256(raw).hexdigest(),"counts":{lang:{split:counts[lang,split] for split in ("train","validation","test")} for lang in ("ja","en")},"derived_training_examples":sum("augmentation" in r for r in rows),"note":"Derived examples are not independent recordings; protected final speech index is separate"}
    protected=ROOT/config["protected_final_speech"]
    result["cv22_data_preparation"]={}
    for suffix in ("v1","v2"):
        path=ROOT/f"cache/features_v2/index_ja85_cv22_{suffix}.json"
        if path.exists():
            progress=json.loads(path.read_text(encoding="utf8"))["cross_version_extension"]
            result["cv22_data_preparation"][suffix]={key:progress[key] for key in ("selected","added","elapsed_seconds")}
            result["cv22_data_preparation"][suffix]["source_revision"]=progress["source"]["revision"]
    result["protected_final_index_sha256"]=hashlib.sha256(protected.read_bytes()).hexdigest()
    result["active_lookahead_ms"]=200
    result["previous_float64_cost_probe"]="384 channels previously exceeded the frame budget; unintended float64 propagation corrected"
    result["float32_cost_probes"]={}
    for name in ("attention256","attention384","attention512x6"):
        path=ROOT/"reports/experiments"/f"2026-10-25_{name}_float32_cost_probe.json"
        if path.exists():
            cost=json.loads(path.read_text(encoding="utf8"))
            result["float32_cost_probes"][name]={key:cost[key] for key in ("frame_ms_p50","frame_ms_p95","real_time_factor","model_sha256")}
    dest=ROOT/"reports/accuracy85_status.json";temp=dest.with_suffix(".tmp.json")
    temp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8");temp.replace(dest)
    print(result["status"])


if __name__=="__main__":main()
