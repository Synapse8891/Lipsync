"""Refit selected numeric residual trees on calibration data and export NumPy arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from kotoba_lipsync.tree_decoder import pack_numeric_trees,TreeResidual
from kotoba_lipsync.model import load_model

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("model",type=Path);parser.add_argument("--trial-report",type=Path,required=True);parser.add_argument("--output",type=Path,required=True);parser.add_argument("--device",default="cuda")
    args=parser.parse_args();threadpool_limits(2)
    import xgboost as xgb
    path=ROOT/args.model;source_raw=path.read_bytes();report=json.loads((ROOT/args.trial_report).read_text(encoding="utf8"))
    if report["primary_model_sha256"]!=hashlib.sha256(source_raw).hexdigest():raise ValueError("Trial report belongs to a different primary")
    with np.load(path) as data:arrays={k:data[k].copy() for k in data.files}
    metadata=json.loads(str(arrays["metadata"]));config=metadata["decoder"]
    if report["nonlinear_primary_residual"].get("uses_auxiliary_features",False):raise ValueError("This exporter requires a primary-only residual; auxiliary fusion is diagnostic only")
    if "tree_residual" in config or config["classes"]!=list(range(16)):raise ValueError("Expected a 16-class linear decoder without trees")
    with np.load(ROOT/config["feature_cache"]) as data:raw=data["x"];truth=data["y"];language=data["language"]
    indices=np.arange(0,len(truth),report["fit_stride"]);x=np.ascontiguousarray(raw[indices]);del raw
    x-=arrays["decoder_mean"];x/=arrays["decoder_scale"]
    baseline=x@arrays["decoder_weight"].T+arrays["decoder_bias"]
    if "decoder_residual_w0" in arrays:baseline+=np.maximum(x@arrays["decoder_residual_w0"].T+arrays["decoder_residual_b0"],0)@arrays["decoder_residual_w1"].T+arrays["decoder_residual_b1"]
    weight=np.where(language[indices]=="en",config.get("english_weight",1.),1.).astype(np.float32)
    training=xgb.DMatrix(x,label=truth[indices],weight=weight,base_margin=baseline)
    parameters={**report["nonlinear_primary_residual"]["parameters"],"device":args.device}
    rounds=report["nonlinear_primary_residual"]["selected_rounds"]
    booster=xgb.train(parameters,training,num_boost_round=rounds,verbose_eval=False)
    folder=ROOT/"cache/decoder_experiments"/Path(args.output).stem;folder.mkdir(parents=True,exist_ok=True);native_path=folder/"residual.json";booster.save_model(native_path)
    packed,description=pack_numeric_trees(json.loads(native_path.read_text(encoding="utf8")))
    description.update(selected_rounds=rounds,calibration_fit_stride=report["fit_stride"],training_data="Tree count selected with speaker-separated development comparison; refit on all previous validation recordings only; no final test",xgboost_version=xgb.__version__)
    reference=xgb.DMatrix(x[:512],base_margin=baseline[:512]);expected=booster.predict(reference,output_margin=True)
    actual=baseline[:512]+TreeResidual(packed,description).predict(x[:512]);np.testing.assert_allclose(actual,expected,atol=2e-5)
    metadata["decoder"]["tree_residual"]=description;arrays.update(packed);arrays["metadata"]=np.array(json.dumps(metadata))
    destination=ROOT/args.output;destination.parent.mkdir(parents=True,exist_ok=True);temporary=destination.with_suffix(".tmp.npz");np.savez_compressed(temporary,**arrays);temporary.replace(destination)
    restored=load_model(destination);raw_probe=x[:32]*arrays["decoder_scale"]+arrays["decoder_mean"];probability=restored._classify(raw_probe)
    expected_prob=np.exp(expected[:32]-expected[:32].max(axis=1,keepdims=True));expected_prob/=expected_prob.sum(axis=1,keepdims=True);np.testing.assert_allclose(probability,expected_prob,atol=2e-5)
    result={"status":"development refit only; final test untouched, production unchanged","source_sha256":hashlib.sha256(source_raw).hexdigest(),"model_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),"model_bytes":destination.stat().st_size,"tree_residual":description,"comparison_from_selected_trial":report["nonlinear_primary_residual"]["comparison"],"numerical_equivalence_max_abs_error":float(np.max(np.abs(actual-expected))),"inference_dependency":"NumPy only; XGBoost not required at runtime"}
    destination.with_suffix(".tree_refit.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8");print("NumPy residual refit exported",destination,flush=True)


if __name__=="__main__":main()
