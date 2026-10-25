"""Cache a small independent backbone for a development-only fusion experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from kotoba_lipsync.model import load_model

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("primary",type=Path);parser.add_argument("--secondary",type=Path,default=Path("models/kotoba_int8.npz"));parser.add_argument("--index",type=Path,required=True);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();threadpool_limits(1)
    dest=ROOT/args.output
    if dest.exists():raise ValueError("Auxiliary feature snapshots must not be overwritten")
    primary_raw=(ROOT/args.primary).read_bytes();secondary_raw=(ROOT/args.secondary).read_bytes()
    with np.load(ROOT/args.primary) as data:primary=json.loads(str(data["metadata"]))
    config=primary["decoder"];total=primary["feature_config"]["lookahead_frames"];offsets=np.asarray(config["offsets"])
    model=load_model(ROOT/args.secondary);base=model.base if hasattr(model,"base") else model
    lookahead=base.metadata.get("decoder",{}).get("base_lookahead_frames",base.config.lookahead_frames)
    if lookahead>total-max(offsets):raise ValueError("Auxiliary branch requires more future audio than the primary")
    index_raw=(ROOT/args.index).read_bytes();index=json.loads(index_raw);rows=[r for r in index["records"] if r["split"]=="validation"]
    xs=[];lengths=[]
    for number,row in enumerate(rows,1):
        with np.load(ROOT/row.get("feature_root","cache/features_v2")/(row["id"]+".npz")) as data:features=data["base"].astype(np.float32)
        p,_=base.predict_sequence(features)
        targets=np.arange(max(0,len(features)-total));positions=np.maximum(targets[:,None]+offsets[None],0)+lookahead
        xs.append(np.log(np.maximum(p[positions],1e-6)).reshape(len(targets),-1));lengths.append(len(targets))
        if number%100==0:print(f"Auxiliary development cache {number}/{len(rows)}",flush=True)
    if hashlib.sha256((ROOT/args.secondary).read_bytes()).digest()!=hashlib.sha256(secondary_raw).digest():raise RuntimeError("Auxiliary model changed during the cache run")
    provenance={"primary_model_sha256":hashlib.sha256(primary_raw).hexdigest(),"secondary_model_sha256":hashlib.sha256(secondary_raw).hexdigest(),"secondary_calibration_removed":True,"secondary_base_lookahead":lookahead,"total_primary_lookahead":total,"offsets":offsets.tolist(),"index_sha256":hashlib.sha256(index_raw).hexdigest(),"scope":"previous validation recordings only; auxiliary backbone logits without its already-fitted posterior decoder; no final test or runtime change"}
    dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,x=np.concatenate(xs).astype(np.float32),lengths=lengths,ids=np.asarray([r["id"] for r in rows]),metadata=np.array(json.dumps(provenance)))
    print("Auxiliary cache saved; production unchanged",flush=True)


if __name__=="__main__":main()
