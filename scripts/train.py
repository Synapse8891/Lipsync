"""Train tiny causal MLP and evaluate the frozen quantized model once.

Validation picks the epoch. Test data never updates normalization, weights or
early stopping. Reports explicitly describe automatic-label agreement.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from threadpoolctl import threadpool_limits
from kotoba_lipsync.features import FeatureConfig
from kotoba_lipsync.model import export_model, TinyModel
from kotoba_lipsync.visemes import LABELS

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION="ctc-midpoint-v2"
FEATURE_DIR=ROOT/"cache/features_v2"

def validate_pipeline(index):
    if index.get("pipeline_version")!=PIPELINE_VERSION:
        raise RuntimeError("旧ラベルは使えません。scripts/prepare_training.py でv2ラベルを作り直してください。")
    configs=index.get("teacher_preprocessing",{})
    if configs.get("ja",{}).get("do_normalize") is not False or configs.get("en",{}).get("do_normalize") is not True:
        raise RuntimeError("教師の正規化設定が想定と違います。前処理の確認が必要です。")

def load_split(index, split, config, stride=1):
    features, labels, langs = [], [], []
    for row in index["records"]:
        if row["split"] != split:
            continue
        with np.load(FEATURE_DIR / f"{row['id']}.npz", allow_pickle=False) as data:
            if str(data.get("annotation_signature",""))!=index["annotation_signatures"][row["language"]]:
                raise RuntimeError(f"キャッシュの教師設定が一致しません: {row['id']}")
            base = data["base"].astype(np.float32)
            t = np.arange(0, len(base)-config.lookahead_frames, stride)
            stack = np.concatenate([base[np.maximum(t+config.lookahead_frames-delay, 0)] for delay in config.context], axis=1)
            features.append(stack)
            labels.append(data["y"][t])
            langs.extend([row["language"]] * len(t))
    return np.concatenate(features), np.concatenate(labels), np.array(langs)

def audit(index):
    for field in ("id", "speaker_hash", "transcript_hash"):
        sets = {s: {r[field] for r in index["records"] if r["split"] == s} for s in ("train", "validation", "test")}
        for a,b in (("train","validation"),("train","test"),("validation","test")):
            if sets[a] & sets[b]:
                raise RuntimeError(f"Data leakage: {field} in {a} and {b}")

def scores(y, prediction):
    present = np.union1d(y, prediction)
    speech = y != 0
    return {"accuracy": float(accuracy_score(y, prediction)),
            "majority_class_baseline":float(np.bincount(y,minlength=16).max()/len(y)),
            "speech_accuracy": float(accuracy_score(y[speech], prediction[speech])) if speech.any() else None,
            "macro_f1": float(f1_score(y, prediction, labels=present, average="macro", zero_division=0)),
            "frames": len(y), "per_class": classification_report(y, prediction, labels=list(range(16)), target_names=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, prediction, labels=list(range(16))).tolist()}

def predict_batched(model, x):
    return np.concatenate([model.predict_proba(x[i:i+8192]).argmax(axis=-1) for i in range(0,len(x),8192)])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--hidden", type=int, nargs="+", default=[192, 96])
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--patience", type=int, default=18)
    p.add_argument("--lookahead",type=int,default=3,help="10 ms future frames; default 30 ms")
    p.add_argument("--model", default="models/kotoba_int8.npz")
    args = p.parse_args()
    index_path=FEATURE_DIR/"index.json"
    if not index_path.exists():
        raise RuntimeError("v2ラベルがありません。scripts/run_training.py を実行してください。")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    validate_pipeline(index)
    audit(index)
    cfg = dict(index["feature_config"])
    cfg["context"] = tuple(cfg["context"])
    cfg["lookahead_frames"] = args.lookahead
    cfg = FeatureConfig(**cfg)
    threadpool_limits(limits=4)
    started = time.perf_counter()
    x,y,langs = load_split(index, "train", cfg, args.stride)
    vx,vy,vlangs = load_split(index,"validation",cfg,2)
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x).astype(np.float32)
    vxs = scaler.transform(vx).astype(np.float32)
    rng = np.random.default_rng(42)
    classes = np.unique(y)
    count = np.bincount(y, minlength=16)
    # Mild class balancing preserves common vowels but gives consonants a voice.
    sample_weights = np.minimum((len(y) / (len(classes) * np.maximum(count[y],1))) ** .35, 4)
    model = MLPClassifier(hidden_layer_sizes=tuple(args.hidden), activation="relu", solver="adam", batch_size=512, learning_rate_init=.0007, alpha=.01, random_state=42, max_iter=1, shuffle=True)
    best = -1
    best_epoch = 0
    history = []
    model_path = ROOT / args.model
    print(f"Training {len(y)} frames; input {x.shape[1]}, hidden {args.hidden}; classes {count.tolist()}", flush=True)
    for epoch in range(1,args.epochs+1):
        model.partial_fit(x,y,classes=classes,sample_weight=sample_weights)
        vp = model.predict(vxs)
        acc = accuracy_score(vy,vp)
        macro = f1_score(vy,vp,average="macro",zero_division=0)
        speech = accuracy_score(vy[vy!=0],vp[vy!=0])
        metric = .5*macro + .5*speech
        history.append({"epoch":epoch,"loss":float(model.loss_),"validation_accuracy":float(acc),"validation_speech_accuracy":float(speech),"validation_macro_f1":float(macro)})
        if metric > best:
            best = metric
            best_epoch = epoch
            export_model(model_path,model.coefs_,model.intercepts_,scaler.mean_,scaler.scale_,cfg,{"classes":classes.tolist(),"labels":list(LABELS),"best_epoch":epoch,"label_kind":index["label_kind"],"data_license":"CC0-1.0","seed":42,"pipeline_version":PIPELINE_VERSION,"annotation_signatures":index["annotation_signatures"]})
        if epoch%5==0 or epoch==1:
            print(f"epoch {epoch:3d}: loss={model.loss_:.4f} val={acc:.4f} speech={speech:.4f} macroF1={macro:.4f} best={best_epoch}; {time.perf_counter()-started:.0f}s",flush=True)
        if epoch-best_epoch >= args.patience:
            break
    del x, y, langs, vx, vxs
    deployed = TinyModel(model_path)
    report = {"evaluation_kind":"Agreement with automatic CTC teacher labels; not human-verified lip-sync accuracy", "target":.8,"human_verified_accuracy":None,"human_verified_target_met":False,"breath_validated":False,"breath_note":index["breath_note"],"test_used_for_model_selection":False,"label_kind":index["label_kind"],"feature_config":cfg.to_dict(),"best_epoch":best_epoch,"parameter_count":deployed.parameters,"model_bytes":model_path.stat().st_size,"weight_storage":"symmetric per-output-channel int8, float32 inference", "training_config":{"hidden":args.hidden,"epochs_max":args.epochs,"stride":args.stride,"patience":args.patience,"seed":42,"learning_rate":.0007,"alpha":.01,"batch_size":512,"selection":"0.5 * validation macro-F1 + 0.5 * validation speech accuracy"}, "history":history, "data_audit":"Recording, speaker and transcript hashes disjoint across train/validation/test", "data_summary":{f"{lang}/{s}":{"recordings":sum(r['language']==lang and r['split']==s for r in index['records']),"speakers":len({r['speaker_hash'] for r in index['records'] if r['language']==lang and r['split']==s}),"seconds":sum(r['seconds'] for r in index['records'] if r['language']==lang and r['split']==s)} for lang in ('ja','en') for s in ('train','validation','test')}}
    for split in ("validation","test"):
        xx,yy,ll = load_split(index,split,cfg)
        pred = predict_batched(deployed,xx)
        report[split] = {"all":scores(yy,pred),**{lang:scores(yy[ll==lang],pred[ll==lang]) for lang in ('ja','en')}}
        print(f"{split}: {report[split]['all']['accuracy']:.4f} speech={report[split]['all']['speech_accuracy']:.4f} macroF1={report[split]['all']['macro_f1']:.4f}",flush=True)
        del xx,yy,ll,pred
    report["elapsed_seconds"] = time.perf_counter()-started
    report["evaluation_stage"] = "Raw quantized MLP argmax, before runtime silence gate and temporal smoothing"
    report["pipeline_version"]=PIPELINE_VERSION
    report["teacher_preprocessing"]=index["teacher_preprocessing"]
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (reports / "split_manifest.json").write_text(json.dumps({k:v for k,v in index.items() if k!='skipped'},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Saved {model_path}: {deployed.parameters:,} parameters, {model_path.stat().st_size:,} bytes",flush=True)

if __name__=="__main__":
    main()
