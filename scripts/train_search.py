"""Memory-bounded validation search. Held-out test is opened only after selection."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from kotoba_lipsync.features import FeatureConfig
from kotoba_lipsync.model import export_model, TinyModel
from kotoba_lipsync.visemes import LABELS
if __package__:
    from .train import audit, scores, validate_pipeline, PIPELINE_VERSION
else:
    from train import audit, scores, validate_pipeline, PIPELINE_VERSION

ROOT=Path(__file__).resolve().parents[1]
FEATURE_DIR=ROOT/"cache/features_v2"

def check_deadline(deadline):
    if deadline is not None and time.time()>=deadline:
        raise TimeoutError("計算時間の上限に達しました")

def iter_batches(index,split,config,*,feature_dir=FEATURE_DIR,stride=2,batch_frames=8192,rng=None,deadline=None):
    """Bound feature memory by a batch plus one utterance, not corpus duration."""
    records=[row for row in index["records"] if row["split"]==split]
    if rng is not None:
        rng.shuffle(records)
    xs,ys,langs=[],[],[]
    count=0
    for row in records:
        check_deadline(deadline)
        with np.load(feature_dir/f"{row['id']}.npz",allow_pickle=False) as data:
            if str(data.get("annotation_signature",""))!=index["annotation_signatures"][row["language"]]:
                raise RuntimeError(f"教師設定の異なるキャッシュ: {row['id']}")
            base=data["base"].astype(np.float32)
            t=np.arange(0,len(base)-config.lookahead_frames,stride)
            if not len(t):continue
            x=np.concatenate([base[np.maximum(t+config.lookahead_frames-delay,0)] for delay in config.context],axis=1)
            xs.append(x)
            ys.append(data["y"][t].copy())
            langs.append(np.full(len(t),row["language"],dtype="U2"))
            count+=len(t)
        if count>=batch_frames:
            x,y,ll=np.concatenate(xs),np.concatenate(ys),np.concatenate(langs)
            if rng is not None:
                order=rng.permutation(len(y));x,y,ll=x[order],y[order],ll[order]
            yield x,y,ll
            xs,ys,langs,count=[],[],[],0
    if xs:
        yield np.concatenate(xs),np.concatenate(ys),np.concatenate(langs)

def evaluation(model,index,config,*,scaler=None,split="validation",stride=2,feature_dir=FEATURE_DIR,deadline=None):
    true,predicted,languages=[],[],[]
    for x,y,ll in iter_batches(index,split,config,feature_dir=feature_dir,stride=stride,deadline=deadline):
        if scaler is not None:
            pred=model.predict(scaler.transform(x))
        else:
            pred=model.predict_proba(x).argmax(axis=-1)
        true.append(y);predicted.append(pred);languages.append(ll)
    if not true:raise RuntimeError(f"評価データが空です: {split}")
    y,p,ll=np.concatenate(true),np.concatenate(predicted),np.concatenate(languages)
    return {"all":scores(y,p),**{lang:scores(y[ll==lang],p[ll==lang]) for lang in ("ja","en") if np.any(ll==lang)}}

def selection_metric(result):
    s=result["all"]
    return .5*s["macro_f1"]+.5*(s["speech_accuracy"] or 0.0)

def choose_trial(results):
    eligible=[trial for trial in results if trial.get("selection_score") is not None]
    if not eligible:raise RuntimeError("検証済みモデルがありません。保存済みラベルで再実行してください。")
    # Exact ties choose the smaller model. The test set is never consulted.
    return max(eligible,key=lambda t:(t["selection_score"],-t["parameter_count"]))

def run_search(index,profile,config,deadline,*,epochs=100,patience=18,feature_dir=FEATURE_DIR,work_dir=None):
    validate_pipeline(index);audit(index)
    for lang in ("ja","en"):
        for split in ("train","validation","test"):
            if not any(r["language"]==lang and r["split"]==split for r in index["records"]):
                raise RuntimeError(f"{lang}/{split} のラベルが不足しています。ラベル生成を再開してください。")
    work_dir=work_dir or ROOT/"cache/search_models"/f"five_hour_{time.time_ns()}"
    work_dir.mkdir(parents=True,exist_ok=True)
    scaler=StandardScaler()
    counts=np.zeros(16,dtype=np.int64)
    for x,y,_ in iter_batches(index,"train",config,feature_dir=feature_dir,deadline=deadline):
        scaler.partial_fit(x);counts+=np.bincount(y,minlength=16)
    classes=np.flatnonzero(counts)
    if len(classes)<2:raise RuntimeError("学習データの口形クラスが不足しています")
    trials=[]
    specifications=profile["trials"]
    for number,spec in enumerate(specifications):
        if time.time()>=deadline:break
        # Share remaining search time across remaining candidates.
        trial_deadline=time.time()+(deadline-time.time())/(len(specifications)-number)
        classifier=MLPClassifier(hidden_layer_sizes=tuple(spec["hidden"]),batch_size=512,learning_rate_init=.0007,alpha=.01,random_state=spec["seed"],max_iter=1,shuffle=True)
        rng=np.random.default_rng(spec["seed"])
        best=-1.0;best_epoch=0;history=[]
        checkpoint=work_dir/f"trial_{number+1}.npz"
        began=time.time()
        print(f"Trial {number+1}/{len(specifications)}: hidden={spec['hidden']} seed={spec['seed']}",flush=True)
        for epoch in range(1,epochs+1):
            try:
                for x,y,_ in iter_batches(index,"train",config,feature_dir=feature_dir,rng=rng,deadline=trial_deadline):
                    weight=np.minimum((counts.sum()/(len(classes)*np.maximum(counts[y],1)))**.35,4)
                    classifier.partial_fit(scaler.transform(x),y,classes=classes,sample_weight=weight)
                validation=evaluation(classifier,index,config,scaler=scaler,feature_dir=feature_dir,deadline=trial_deadline)
            except TimeoutError:
                break
            metric=selection_metric(validation)
            s=validation["all"]
            history.append({"epoch":epoch,"validation_accuracy":s["accuracy"],"validation_speech_accuracy":s["speech_accuracy"],"validation_macro_f1":s["macro_f1"]})
            if metric>best:
                best,best_epoch=metric,epoch
                export_model(checkpoint,classifier.coefs_,classifier.intercepts_,scaler.mean_,scaler.scale_,config,{"classes":classes.tolist(),"labels":list(LABELS),"best_epoch":epoch,"label_kind":index["label_kind"],"data_license":"CC0-1.0","seed":spec["seed"],"pipeline_version":PIPELINE_VERSION,"annotation_signatures":index["annotation_signatures"]})
            print(f"  epoch {epoch}: validation={s['accuracy']:.4f}, speech={s['speech_accuracy']:.4f}, macroF1={s['macro_f1']:.4f}",flush=True)
            if epoch-best_epoch>=patience:break
        if best_epoch:
            deployed=TinyModel(checkpoint)
            # Re-score the actual quantized candidate on validation, never on test.
            quantized=evaluation(deployed,index,config,feature_dir=feature_dir)
            result={"trial":number+1,**spec,"best_epoch":best_epoch,"history":history,"validation":quantized,"selection_score":selection_metric(quantized),"parameter_count":deployed.parameters,"model_bytes":checkpoint.stat().st_size,"checkpoint":str(checkpoint),"elapsed_seconds":time.time()-began}
            trials.append(result)
            (work_dir/"search_progress.json").write_text(json.dumps(trials,indent=2),encoding="utf-8")
        del classifier
    return choose_trial(trials),trials

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--index",type=Path,default=FEATURE_DIR/"index_five_hour.json")
    parser.add_argument("--deadline",type=float,default=None,help="Unix timestamp for search cutoff; final evaluation follows")
    parser.add_argument("--epochs",type=int,default=100)
    parser.add_argument("--patience",type=int,default=18)
    args=parser.parse_args()
    if args.epochs<1 or args.patience<1:parser.error("epochs/patience must be positive")
    profile=json.loads((ROOT/"configs/training_profiles.json").read_text(encoding="utf-8"))["five-hour"]
    index=json.loads(args.index.read_text(encoding="utf-8"))
    config=FeatureConfig(**{**index["feature_config"],"context":tuple(index["feature_config"]["context"]),"lookahead_frames":3})
    threadpool_limits(limits=4)
    started=time.time()
    winner,trials=run_search(index,profile,config,args.deadline or time.time()+5*3600,epochs=args.epochs,patience=args.patience)
    model=TinyModel(winner["checkpoint"])
    # This is the sole test evaluation, after architecture/seed/epoch selection.
    test=evaluation(model,index,config,split="test",stride=1)
    validation=evaluation(model,index,config,split="validation",stride=1)
    output=ROOT/"models/kotoba_int8.npz"
    output.parent.mkdir(exist_ok=True)
    shutil.copyfile(winner["checkpoint"],output.with_suffix(".tmp.npz"))
    output.with_suffix(".tmp.npz").replace(output)
    report={"evaluation_kind":"Agreement with automatic CTC teacher labels; not human-verified lip-sync accuracy","target":.8,"human_verified_accuracy":None,"human_verified_target_met":False,"breath_validated":False,"breath_note":index["breath_note"],"test_used_for_model_selection":False,"test_evaluations":1,"label_kind":index["label_kind"],"feature_config":config.to_dict(),"best_epoch":winner["best_epoch"],"parameter_count":model.parameters,"model_bytes":output.stat().st_size,"weight_storage":"int8 storage, float32 inference","training_config":{"hidden":winner["hidden"],"seed":winner["seed"],"epochs_max":args.epochs,"patience":args.patience,"stride":2,"selection":"validation macro-F1 and speech accuracy; test evaluated once","mode":"bounded-memory search"},"history":winner["history"],"validation":validation,"test":test,"search_trials":trials,"selected_trial":winner["trial"],"data_audit":"Recording, speaker and transcript hashes disjoint across splits","pipeline_version":PIPELINE_VERSION,"teacher_preprocessing":index["teacher_preprocessing"],"elapsed_seconds":time.time()-started,"evaluation_stage":"Raw quantized MLP argmax, before runtime silence gate and smoothing","data_summary":{f"{lang}/{split}":{"recordings":sum(r["language"]==lang and r["split"]==split for r in index["records"]),"speakers":len({r["speaker_hash"] for r in index["records"] if r["language"]==lang and r["split"]==split}),"seconds":sum(r["seconds"] for r in index["records"] if r["language"]==lang and r["split"]==split)} for lang in ("ja","en") for split in ("train","validation","test")}}
    reports=ROOT/"reports";reports.mkdir(exist_ok=True)
    (reports/"training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (reports/"split_manifest.json").write_text(json.dumps({k:v for k,v in index.items() if k!="skipped"},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Selected trial {winner['trial']}, test agreement {test['all']['accuracy']:.4f}. Human-verified accuracy unknown.",flush=True)

if __name__=="__main__":main()
