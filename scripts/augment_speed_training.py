"""Training-only waveform speed/pitch perturbation of selected CC0 recordings."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
from threadpoolctl import threadpool_limits
from kotoba_lipsync.features import FeatureConfig,extract_audio

ROOT=Path(__file__).resolve().parents[1]


def stretch_labels(labels,times,speed,db,config=None):
    config=config or FeatureConfig()
    position=np.rint((np.asarray(times)*speed+config.frame_size/(2*config.sample_rate))*config.sample_rate/config.hop_size-1).astype(np.int64)
    output=np.asarray(labels)[np.clip(position,0,len(labels)-1)].copy();output[np.asarray(db)<-55]=0
    return output


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--index",default="cache/features_v2/index_ja85_stage1_20261025.json")
    parser.add_argument("--output",default="cache/features_v2/index_ja85_speed_v1.json")
    parser.add_argument("--count",type=int,default=8000);parser.add_argument("--workers",type=int,default=2)
    args=parser.parse_args();threadpool_limits(1)
    source_path=ROOT/args.index;source=json.loads(source_path.read_text(encoding="utf8"))
    rows=[r for r in source["records"] if r["split"]=="train" and r["language"]=="ja" and "augmentation" not in r]
    grouped={}
    for row in sorted(rows,key=lambda r:hashlib.sha256(r["id"].encode()).hexdigest()):grouped.setdefault(row["speaker_hash"],[]).append(row)
    selected={}
    for offset in range(max((len(group) for group in grouped.values()),default=0)):
        for speaker in sorted(grouped):
            if offset<len(grouped[speaker]):selected[grouped[speaker][offset]["id"]]=grouped[speaker][offset]
            if len(selected)>=args.count:break
        if len(selected)>=args.count:break
    destination=ROOT/"cache/features_speed_v1";destination.mkdir(exist_ok=True)
    output=ROOT/args.output;started=time.monotonic();added=[];skipped=[];cfg=FeatureConfig()
    def save():
        record={**source,"records":source["records"]+sorted(added,key=lambda r:r["id"]),"waveform_augmentation":{"version":"speed-v1","original_index_sha256":hashlib.sha256(source_path.read_bytes()).hexdigest(),"added":len(added),"selected":len(selected),"elapsed_seconds":time.monotonic()-started,"factors":[.9,1.1],"scope":"Derived training examples only, not additional independent recordings; original evaluation labels remain unchanged","skipped":skipped}}
        temporary=output.with_suffix(".tmp.json");temporary.write_text(json.dumps(record,indent=2)+"\n",encoding="utf8");temporary.replace(output)
    def prepare(row,blob):
        speed=.9 if int(hashlib.sha256(row["id"].encode()).hexdigest()[:8],16)%2==0 else 1.1
        ident=f"speed{round(speed*100):03d}_"+row["id"];path=destination/(ident+".npz")
        with np.load(ROOT/row.get("feature_root","cache/features_v2")/(row["id"]+".npz")) as original:
            expected=source["annotation_signatures"]["ja"]
            if str(original["annotation_signature"])!=expected:raise ValueError("Original teacher signature differs")
            truth=original["y"].copy();unknown=int(original["unknown"])
        signature=hashlib.sha256(json.dumps({"teacher":expected,"audio":hashlib.sha256(blob).hexdigest(),"labels":hashlib.sha256(truth.tobytes()).hexdigest(),"speed":speed,"features":cfg.to_dict(),"version":"waveform-speed-v1"},sort_keys=True).encode()).hexdigest()
        try:
            current=False
            if path.exists():
                with np.load(path) as cached:current=str(cached["annotation_signature"])==signature
            if not current:
                audio,rate=sf.read(io.BytesIO(blob),dtype="float32",always_2d=True);audio=audio.mean(1)
                ratio=Fraction(16000,rate)/Fraction(str(speed));audio=resample_poly(audio,ratio.numerator,ratio.denominator).astype(np.float32)
                features,times,diagnostics=extract_audio(audio,cfg);labels=stretch_labels(truth,times,speed,[d["rms_db"] for d in diagnostics],cfg)
                temporary=path.with_suffix(".tmp.npz");np.savez_compressed(temporary,base=features[:,:78].astype(np.float16),y=labels,unknown=unknown,seconds=len(audio)/16000,annotation_signature=signature,teacher_signature=expected,pipeline_version="ctc-midpoint-v2/speed-v1",source_id=row["id"],speed_factor=speed);temporary.replace(path)
            with np.load(path) as data:
                return {**row,"id":ident,"frames":len(data["y"]),"seconds":float(data["seconds"]),"feature_root":"cache/features_speed_v1","annotation_signature":signature,"source_recording_id":row["id"],"augmentation":{"kind":"waveform-speed-v1","factor":speed,"labels":"Original training labels warped to the transformed frame centers, then the same -55 dBFS silence gate applied"}},None
        except (ValueError,RuntimeError,sf.LibsndfileError) as error:return None,{"id":ident,"reason":str(error)}
    def consume(futures):
        for future in futures:
            row,error=future.result()
            if error:skipped.append(error)
            if row is not None:added.append(row)
            if added and len(added)%25==0:save();print(f"Derived speed examples {len(added)}/{len(selected)}, elapsed={time.monotonic()-started:.0f}s",flush=True)
    pending=set();seen=set();stopped=False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for shard in sorted((ROOT/"data/common_voice_17/ja").glob("*.parquet")):
            if stopped:break
            for batch in pq.ParquetFile(shard).iter_batches(batch_size=32):
                if stopped:break
                for row in batch.to_pylist():
                    ident=Path(row["path"]).stem
                    if ident not in selected or ident in seen:continue
                    if output.with_suffix(".STOP").exists():stopped=True;break
                    while len(pending)>=args.workers:
                        done,pending=wait(pending,return_when=FIRST_COMPLETED);consume(done)
                    seen.add(ident);pending.add(pool.submit(prepare,selected[ident],row["audio"]["bytes"]))
        consume(pending)
    save();print(f"Saved {len(added)} derived training examples; evaluation set unchanged.",flush=True)


if __name__=="__main__":main()
