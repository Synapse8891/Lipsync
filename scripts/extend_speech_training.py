"""Repartition unused CC0 speech while protecting all development/final speakers."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import time
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
from threadpoolctl import threadpool_limits
from kotoba_lipsync.features import FeatureConfig,extract_audio
from scripts.prepare_training import Teacher,key,cache_is_current,PIPELINE_VERSION

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--language",choices=("ja","en"),default="ja")
    parser.add_argument("--index",default="cache/features_v2/index_extended_complete_v3.json")
    parser.add_argument("--output",default="cache/features_v2/index_ja85_extended_v1.json")
    parser.add_argument("--reserve-index",action="append",required=True)
    parser.add_argument("--source-splits",default="train,validation,test")
    parser.add_argument("--count",type=int,default=6000);parser.add_argument("--hours",type=float,default=3)
    parser.add_argument("--workers",type=int,default=2);parser.add_argument("--threads",type=int,default=4)
    args=parser.parse_args();threadpool_limits(1);lang=args.language
    index_path=ROOT/args.index;index=json.loads(index_path.read_text(encoding="utf8"));reserved=[]
    for filename in args.reserve_index:reserved+=json.loads((ROOT/filename).read_text(encoding="utf8"))["records"]
    previous=index["records"];protected=[r for r in previous if r["split"]!="train"]+reserved
    ids={r["id"] for r in previous+reserved};texts={r["transcript_hash"] for r in previous+reserved}
    speakers={r["speaker_hash"] for r in protected};global_ids={r["global_speaker_hash"] for r in protected if "global_speaker_hash" in r}
    protected_ids={r["id"] for r in protected}
    for path in (ROOT/"data/common_voice_17").glob("*/*.parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048,columns=["client_id","path"]):
            for row in batch.to_pylist():
                if Path(row["path"]).stem in protected_ids:global_ids.add(key(str(row["client_id"])))
    allowed=set(args.source_splits.split(","));paths=[p for p in sorted((ROOT/"data/common_voice_17"/lang).glob("*.parquet")) if p.name.split("-")[0] in allowed]
    candidates=[]
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024,columns=["client_id","path","sentence","up_votes","down_votes"]):
            for row in batch.to_pylist():
                if not row["client_id"] or row["down_votes"]>row["up_votes"]:continue
                ident=Path(row["path"]).stem;speaker=key(lang+str(row["client_id"]));text=key(lang+str(row["sentence"]).strip());global_id=key(str(row["client_id"]))
                if ident in ids or speaker in speakers or text in texts or global_id in global_ids:continue
                candidates.append((key(ident),ident,speaker,text,global_id,path.name))
    grouped={}
    for candidate in sorted(candidates):grouped.setdefault(candidate[2],[]).append(candidate)
    selected={};chosen_texts=set()
    for position in range(max((len(group) for group in grouped.values()),default=0)):
        for speaker in sorted(grouped):
            if position>=len(grouped[speaker]):continue
            candidate=grouped[speaker][position]
            if candidate[1] in selected or candidate[3] in chosen_texts:continue
            selected[candidate[1]]=candidate;chosen_texts.add(candidate[3])
            if len(selected)>=args.count:break
        if len(selected)>=args.count:break
    print(f"Selected {len(selected)} additional {lang} clips from {len(grouped)} eligible speakers; protected global client IDs={len(global_ids)}",flush=True)
    teacher=Teacher(lang,args.threads)
    if teacher.signature!=index["annotation_signatures"][lang]:raise ValueError("Teacher signature changed")
    output=ROOT/args.output;started=time.monotonic();added=[];skipped=[]
    pinned=json.loads((ROOT/"configs/training_sources.json").read_text(encoding="utf8"))["dataset"]
    def save():
        document={**index,"records":previous+sorted(added,key=lambda row:row["id"]),"speech_extension":{"language":lang,"added":len(added),"requested":args.count,"selected":len(selected),"elapsed_seconds":time.monotonic()-started,"base_index_sha256":hashlib.sha256(index_path.read_bytes()).hexdigest(),"reserved_indices":args.reserve_index,"source":pinned,"source_splits":sorted(allowed),"repartition_note":"Unused source validation/test clips explicitly reassigned to training; all earlier evaluation and final-holdout IDs, speakers and transcripts excluded","skipped":skipped}}
        temporary=output.with_suffix(".tmp.json");temporary.write_text(json.dumps(document,indent=2)+"\n",encoding="utf8");temporary.replace(output)
    def prepare(candidate,blob):
        _,ident,speaker,text,global_id,source=candidate;destination=ROOT/"cache/features_v2"/(ident+".npz")
        try:
            if not cache_is_current(destination,teacher.signature):
                audio,rate=sf.read(io.BytesIO(blob),dtype="float32",always_2d=True);audio=audio.mean(1)
                if rate!=16000:
                    divisor=gcd(rate,16000);audio=resample_poly(audio,16000//divisor,rate//divisor).astype(np.float32)
                if not .4<=len(audio)/16000<=18:return None,{"id":ident,"reason":"Duration outside 0.4 to 18 seconds"}
                features,times,diagnostics=extract_audio(audio,FeatureConfig());labels,unknown=teacher.annotate(audio,times,diagnostics)
                temporary=destination.with_suffix(".tmp.npz");np.savez_compressed(temporary,base=features[:,:78].astype(np.float16),y=labels,unknown=unknown,seconds=len(audio)/16000,annotation_signature=teacher.signature,pipeline_version=PIPELINE_VERSION);temporary.replace(destination)
            with np.load(destination) as data:return {"id":ident,"language":lang,"split":"train","speaker_hash":speaker,"global_speaker_hash":global_id,"transcript_hash":text,"frames":len(data["y"]),"seconds":float(data["seconds"]),"unknown_phones":int(data["unknown"]),"origin_shard":source},None
        except (ValueError,RuntimeError,sf.LibsndfileError) as error:return None,{"id":ident,"reason":str(error)}
    def consume(futures):
        for future in futures:
            record,error=future.result()
            if error:skipped.append(error)
            if record is not None:added.append(record)
            if len(added) and len(added)%25==0:save();print(f"Added {lang} {len(added)}/{len(selected)}, elapsed={time.monotonic()-started:.0f}s",flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=set();done_ids=set();stopped=False
        for path in paths:
            if stopped:break
            if not any(candidate[5]==path.name for candidate in selected.values()):continue
            for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
                if stopped:break
                for row in batch.to_pylist():
                    ident=Path(row["path"]).stem
                    if ident not in selected or ident in done_ids:continue
                    if time.monotonic()-started>=args.hours*3600 or output.with_suffix(".STOP").exists():stopped=True;break
                    while len(pending)>=args.workers:
                        done,pending=wait(pending,return_when=FIRST_COMPLETED);consume(done)
                    done_ids.add(ident);pending.add(pool.submit(prepare,selected[ident],row["audio"]["bytes"]))
        consume(pending)
    save();print(f"Saved {len(added)} additional {lang} clips; untouched final holdout remains protected.",flush=True)


if __name__=="__main__":main()
