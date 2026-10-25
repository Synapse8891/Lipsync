"""Prepare previously unused recordings/speakers/texts; never score a candidate."""
import io
import json
from pathlib import Path
from math import gcd
import argparse
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from kotoba_lipsync.features import FeatureConfig,extract_audio
if __package__:
    from .download_training import download
    from .prepare_training import Teacher,key,cache_is_current,PIPELINE_VERSION
else:
    from download_training import download
    from prepare_training import Teacher,key,cache_is_current,PIPELINE_VERSION

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument("--ja-count",type=int,default=200);p.add_argument("--en-count",type=int,default=60);p.add_argument("--threads",type=int,default=2)
    p.add_argument("--shard",type=int,default=1);p.add_argument("--folder",default="cache/features_fresh_v2");p.add_argument("--exclude-index",action="append",default=[])
    args=p.parse_args()
    baseline=json.loads((ROOT/"cache/features_v2/index_five_hour.json").read_text(encoding="utf-8"))
    old=baseline["records"]
    for file in args.exclude_index:old+=json.loads((ROOT/file).read_text(encoding="utf-8"))["records"]
    ids={r["id"] for r in old};speakers={r["speaker_hash"] for r in old};texts={r["transcript_hash"] for r in old}
    global_speakers=set()
    # Anonymous Common Voice client IDs can occur in both languages. Reserve
    # against every downloaded training shard as well as all prior recordings.
    for metadata_path in (ROOT/"data/common_voice_17").glob("*/*.parquet"):
        training=metadata_path.name.startswith("train-")
        for batch in pq.ParquetFile(metadata_path).iter_batches(batch_size=1024,columns=["client_id","path"]):
            for row in batch.to_pylist():
                if training or Path(row["path"]).stem in ids:global_speakers.add(key(str(row["client_id"])))
    pinned=json.loads((ROOT/"configs/training_sources.json").read_text(encoding="utf-8"))["dataset"]
    folder=ROOT/args.folder;folder.mkdir(exist_ok=True)
    records=[];signatures={};skipped=[]
    for lang,limit,shards in (("ja",args.ja_count,8),("en",args.en_count,19)):
        source=f"{lang}/test-{args.shard:05d}-of-{shards:05d}.parquet"
        path=ROOT/"data/common_voice_17"/source
        download(f"https://huggingface.co/datasets/{pinned['repo']}/resolve/{pinned['revision']}/{source}",path)
        candidates=[]
        for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
            for row in batch.to_pylist():
                speaker=key(lang+str(row["client_id"]));text=key(lang+str(row["sentence"]).strip())
                global_speaker=key(str(row["client_id"]))
                name=Path(row["audio"]["path"] or row["path"]).stem
                if not row["client_id"] or row.get("down_votes",0)>row.get("up_votes",0):continue
                if name in ids or speaker in speakers or text in texts or global_speaker in global_speakers:continue
                candidates.append((key(name),name,speaker,text,row["audio"]["bytes"],global_speaker))
        candidates.sort()
        grouped={}
        for row in candidates:grouped.setdefault(row[2],[]).append(row)
        order=[grouped[s][i] for i in range(max((len(g) for g in grouped.values()),default=0)) for s in sorted(grouped) if i<len(grouped[s])]
        teacher=Teacher(lang,args.threads);signatures[lang]=teacher.signature
        count=0
        for _,name,speaker,text,blob,global_speaker in order:
            if count>=limit:break
            if name in ids or text in texts:continue
            dest=folder/(name+".npz")
            try:
                if not cache_is_current(dest,teacher.signature):
                    audio,sr=sf.read(io.BytesIO(blob),dtype="float32",always_2d=True);audio=audio.mean(axis=1)
                    if sr!=16000:
                        div=gcd(sr,16000);audio=resample_poly(audio,16000//div,sr//div).astype(np.float32)
                    if not .4<=len(audio)/16000<=18:continue
                    features,times,diag=extract_audio(audio,FeatureConfig())
                    y,unknown=teacher.annotate(audio,times,diag)
                    np.savez_compressed(dest,base=features[:,:78].astype(np.float16),y=y,unknown=unknown,seconds=len(audio)/16000,annotation_signature=teacher.signature,pipeline_version=PIPELINE_VERSION)
                with np.load(dest) as data:
                    record={"id":name,"speaker_hash":speaker,"global_speaker_hash":global_speaker,"transcript_hash":text,"language":lang,"split":"fresh_test","frames":len(data["y"]),"seconds":float(data["seconds"]),"unknown_phones":int(data["unknown"])}
                records.append(record);ids.add(name);texts.add(text);count+=1
                if count%20==0:print(f"Fresh {lang}: {count}/{limit}",flush=True)
            except (ValueError,RuntimeError,sf.LibsndfileError) as error:skipped.append({"id":name,"reason":str(error)})
        del teacher
        print(f"Fresh {lang}: {count} recordings, {len({r['speaker_hash'] for r in records if r['language']==lang})} previously unused speakers",flush=True)
        if count<limit:raise RuntimeError(f"Only {count} eligible {lang} recordings; need another unused shard")
    result={"records":records,"pipeline_version":PIPELINE_VERSION,"annotation_signatures":signatures,"feature_config":FeatureConfig().to_dict(),"source":pinned,"source_shard":args.shard,"excluded_indices":args.exclude_index,"skipped":skipped,"overlap_check":"recording IDs, speaker hashes and transcript hashes absent from all previous train/validation/test records","note":"Labels only; no candidate inference or metric calculated"}
    (folder/"index.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("Fresh holdout prepared. No model evaluated.",flush=True)

if __name__=="__main__":main()
