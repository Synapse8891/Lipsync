"""Compare auxiliary logits and nonlinear residuals on development speakers only."""
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


def main():
    parser=argparse.ArgumentParser();parser.add_argument("model",type=Path);parser.add_argument("--auxiliary",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--stride",type=int,default=4);parser.add_argument("--device",default="cuda")
    parser.add_argument("--tree-auxiliary",action="store_true")
    args=parser.parse_args();threadpool_limits(2)
    import torch
    import xgboost as xgb
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    if args.stride<1:raise ValueError("Stride must be positive")
    model_path=ROOT/args.model;model_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    with np.load(model_path) as data:metadata=json.loads(str(data["metadata"]))
    config=metadata["decoder"]
    with np.load(ROOT/config["feature_cache"]) as data:raw=data["x"];y=data["y"];fit=data["fit"];language=data["language"]
    train=np.flatnonzero(fit)[::args.stride];comparison=np.flatnonzero(~fit&(language=="ja"))
    train_x=np.ascontiguousarray(raw[train]);valid_x=np.ascontiguousarray(raw[comparison]);del raw
    scaler=StandardScaler(copy=False);train_x=scaler.fit_transform(train_x);valid_x=scaler.transform(valid_x)
    weights=np.where(language[train]=="en",config.get("english_weight",1.),1.).astype(np.float32)
    # Keep comparable ridge strength when thinning only calibration fitting frames.
    c=config["C"]*args.stride
    linear=fit_linear(train_x,y[train],c,sample_weight=weights,device=args.device)
    if linear.classes_.tolist()!=list(range(16)):raise ValueError("Subsampled fit is missing a class")
    baseline=scores(y[comparison],linear.predict(valid_x));print("Subsampled linear baseline",baseline["accuracy"],flush=True)
    with np.load(ROOT/args.auxiliary) as data:aux=data["x"];provenance=json.loads(str(data["metadata"]))
    if provenance["primary_model_sha256"]!=model_sha or len(aux)!=len(y):raise ValueError("Auxiliary cache belongs to a different primary")
    aux_scaler=StandardScaler(copy=False);train_aux=aux_scaler.fit_transform(np.ascontiguousarray(aux[train]));valid_aux=aux_scaler.transform(np.ascontiguousarray(aux[comparison]));del aux
    joined_train=np.c_[train_x,train_aux];joined_valid=np.c_[valid_x,valid_aux];del train_aux,valid_aux
    fusion=[];selected_fusion=None
    for strength in (c,c*3):
        classifier=fit_linear(joined_train,y[train],strength,sample_weight=weights,device=args.device)
        result=scores(y[comparison],classifier.predict(joined_valid));fusion.append({"C":strength,"comparison":result})
        print("Auxiliary ridge",strength,result["accuracy"],flush=True)
        if selected_fusion is None or result["accuracy"]>selected_fusion[0]:selected_fusion=(result["accuracy"],classifier)
    if args.tree_auxiliary:train_x,valid_x,linear=joined_train,joined_valid,selected_fusion[1]
    else:del joined_train,joined_valid
    torch.cuda.empty_cache()
    margin_train=train_x@linear.coef_.T+linear.intercept_;margin_valid=valid_x@linear.coef_.T+linear.intercept_
    training=xgb.DMatrix(train_x,label=y[train],weight=weights,base_margin=margin_train)
    validation=xgb.DMatrix(valid_x,label=y[comparison],base_margin=margin_valid)
    parameters={"device":args.device,"tree_method":"hist","objective":"multi:softprob","num_class":16,"max_depth":4,"eta":.04,"lambda":20,"min_child_weight":10,"max_bin":128,"subsample":.8,"colsample_bytree":.7,"nthread":2,"seed":73,"eval_metric":"merror"}
    booster=xgb.train(parameters,training,num_boost_round=700,evals=[(validation,"development_comparison")],early_stopping_rounds=80,verbose_eval=50)
    best_iteration=int(booster.best_iteration);booster=booster[:best_iteration+1]
    probability=booster.predict(validation);tree_result=scores(y[comparison],probability.argmax(1))
    destination=ROOT/args.output;destination.parent.mkdir(parents=True,exist_ok=True);tree_path=ROOT/"cache/decoder_experiments"/destination.stem/"comparison.json";tree_path.parent.mkdir(parents=True,exist_ok=True);booster.save_model(tree_path)
    report={"status":"development diagnostic only; no final test, runtime change or promotion","primary_model_sha256":model_sha,"auxiliary_provenance":provenance,"fit_stride":args.stride,"fit_frames":len(train),"comparison_frames":len(comparison),"scope":"Only calibration fitting frames thinned; Japanese comparison remains at 100 Hz; earlier validation speakers only","linear_baseline":baseline,"auxiliary_ridge":fusion,"nonlinear_primary_residual":{"parameters":parameters,"selected_rounds":best_iteration+1,"comparison":tree_result,"model_file":str(tree_path.relative_to(ROOT)),"xgboost_version":xgb.__version__}}
    report["nonlinear_primary_residual"]["uses_auxiliary_features"]=args.tree_auxiliary
    destination.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8");print("Primary nonlinear residual",tree_result["accuracy"],flush=True)


if __name__=="__main__":main()
