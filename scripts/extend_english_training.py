"""Add CC0 English training speech without changing any existing partition."""
import argparse
import io
import json
from math import gcd
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from kotoba_lipsync.features import FeatureConfig,extract_audio
if __package__:
    from .prepare_training import Teacher,key,cache_is_current,PIPELINE_VERSION
else:
    from prepare_training import Teacher,key,cache_is_current,PIPELINE_VERSION

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--count",type=int,default=1600);parser.add_argument("--hours",type=float,default=1)
    parser.add_argument("--index",default="cache/features_v2/index_development_v3.json");parser.add_argument("--output",default="cache/features_v2/index_extended_v3.json")
    parser.add_argument("--reserve-index",default="cache/features_fresh_v3/index.json");parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--shards",help="Comma-separated train shard indices; default uses downloaded shards")
    args=parser.parse_args();index=json.loads((ROOT/args.index).read_text());reserve=json.loads((ROOT/args.reserve_index).read_text())
    ids={r["id"] for r in index["records"]+reserve["records"]};texts={r["transcript_hash"] for r in index["records"]+reserve["records"]}
    excluded={r["speaker_hash"] for r in index["records"] if r["split"]!="train"}|{r["speaker_hash"] for r in reserve["records"]}
    global_reserved={r["global_speaker_hash"] for r in reserve["records"] if "global_speaker_hash" in r}
    candidates=[]
    selected_shards={int(value) for value in args.shards.split(",")} if args.shards else None
    for path in sorted((ROOT/"data/common_voice_17/en").glob("train-*.parquet")):
        if selected_shards is not None and int(path.name.split("-")[1]) not in selected_shards:continue
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                if not row["client_id"] or row.get("down_votes",0)>row.get("up_votes",0):continue
                name=Path(row["audio"]["path"] or row["path"]).stem;speaker=key("en"+str(row["client_id"]));text=key("en"+str(row["sentence"]).strip())
                if name in ids or text in texts or speaker in excluded or key(str(row["client_id"])) in global_reserved:continue
                candidates.append((key(name),name,speaker,text,row["audio"]["bytes"]))
    candidates.sort();unique=[];candidate_texts=set();candidate_ids=set()
    for candidate in candidates:
        if candidate[1] in candidate_ids or candidate[3] in candidate_texts:continue
        unique.append(candidate);candidate_ids.add(candidate[1]);candidate_texts.add(candidate[3])
    teacher=Teacher("en",args.threads)
    if teacher.signature!=index["annotation_signatures"]["en"]:raise ValueError("Teacher preprocessing/signature changed")
    added=0;started=time.monotonic();skipped=[];output=ROOT/args.output
    def save():
        index["extended_english"]={"added":added,"requested":args.count,"elapsed_seconds":time.monotonic()-started,"reserved_index":args.reserve_index,"skipped":skipped}
        temporary=output.with_suffix(".tmp.json");temporary.write_text(json.dumps(index,indent=2),encoding="utf-8");temporary.replace(output)
    def prepare(candidate):
        _,name,speaker,text,blob=candidate
        destination=ROOT/"cache/features_v2"/(name+".npz")
        try:
            if not cache_is_current(destination,teacher.signature):
                audio,sr=sf.read(io.BytesIO(blob),dtype="float32",always_2d=True);audio=audio.mean(axis=1)
                if sr!=16000:
                    div=gcd(sr,16000);audio=resample_poly(audio,16000//div,sr//div).astype(np.float32)
                if not .4<=len(audio)/16000<=18:return None,None
                features,times,diagnostics=extract_audio(audio,FeatureConfig());labels,unknown=teacher.annotate(audio,times,diagnostics)
                temporary=destination.with_suffix(".tmp.npz")
                np.savez_compressed(temporary,base=features[:,:78].astype(np.float16),y=labels,unknown=unknown,seconds=len(audio)/16000,annotation_signature=teacher.signature,pipeline_version=PIPELINE_VERSION);temporary.replace(destination)
            with np.load(destination) as data:
                return {"id":name,"language":"en","split":"train","speaker_hash":speaker,"transcript_hash":text,"frames":len(data["y"]),"seconds":float(data["seconds"]),"unknown_phones":int(data["unknown"])},None
        except (ValueError,RuntimeError,sf.LibsndfileError) as error:return None,{"id":name,"reason":str(error)}
    # The CPU ONNX session shares read-only weights. Only the main thread writes
    # the index; bounded pending jobs avoid a second corpus-sized queue in RAM.
    iterator=iter(unique)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=set()
        for _ in range(args.workers):
            candidate=next(iterator,None)
            if candidate is not None:pending.add(pool.submit(prepare,candidate))
        while pending and added<args.count and time.monotonic()-started<args.hours*3600 and not output.with_suffix(".STOP").exists():
            done,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                record,error=future.result()
                if error:skipped.append(error)
                if record is not None and added<args.count:
                    index["records"].append(record);added+=1
                    if added%25==0:
                        save();print(f"Added English {added}/{args.count}, elapsed={time.monotonic()-started:.0f}s",flush=True)
                if added<args.count:
                    candidate=next(iterator,None)
                    if candidate is not None:pending.add(pool.submit(prepare,candidate))
    save();print(f"Extended index saved with {added} new English recordings. Reserved holdout unchanged.",flush=True)

if __name__=="__main__":main()
