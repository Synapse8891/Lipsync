"""Add pinned CC0 Japanese recordings with cross-version speaker protection."""
import argparse
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import csv
import hashlib
import io
import json
from math import gcd
from pathlib import Path
import shutil
import tarfile
import time
import numpy as np
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
from threadpoolctl import threadpool_limits
from kotoba_lipsync.features import FeatureConfig,extract_audio
from scripts.download_training import download
from scripts.prepare_training import Teacher,key,PIPELINE_VERSION

ROOT=Path(__file__).resolve().parents[1]


def select_candidates(rows,previous,reserved,protected_globals,count):
    protected=[r for r in previous if r["split"]!="train"]+reserved
    ids={r["id"] for r in previous+reserved};texts={r["transcript_hash"] for r in previous+reserved}
    speakers={r["speaker_hash"] for r in protected};groups={}
    protected_globals=set(protected_globals)|{r["global_speaker_hash"] for r in protected if "global_speaker_hash" in r}
    for row in rows:
        if not row.get("client_id") or int(row.get("down_votes") or 0)>int(row.get("up_votes") or 0):continue
        ident=Path(row["path"]).stem;speaker=key("ja"+row["client_id"]);text=key("ja"+row["sentence"].strip());glob=key(row["client_id"])
        if ident in ids or text in texts or speaker in speakers or glob in protected_globals:continue
        candidate={"id":ident,"speaker_hash":speaker,"global_speaker_hash":glob,"transcript_hash":text,"source_split":row["source_split"]}
        groups.setdefault(speaker,{})[ident]=candidate
    groups={speaker:sorted(group.values(),key=lambda r:key(r["id"])) for speaker,group in groups.items()}
    selected={};chosen_texts=set()
    for position in range(max((len(g) for g in groups.values()),default=0)):
        for speaker in sorted(groups):
            if position>=len(groups[speaker]):continue
            candidate=groups[speaker][position]
            if candidate["transcript_hash"] in chosen_texts:continue
            selected[candidate["id"]]=candidate;chosen_texts.add(candidate["transcript_hash"])
            if len(selected)>=count:return selected
    return selected


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--index",default="cache/features_v2/index_ja85_stage3_20261025.json");parser.add_argument("--output",default="cache/features_v2/index_ja85_cv22_v1.json")
    parser.add_argument("--reserve-index",action="append",required=True);parser.add_argument("--count",type=int,default=3000);parser.add_argument("--workers",type=int,default=2);parser.add_argument("--threads",type=int,default=2)
    args=parser.parse_args();threadpool_limits(1)
    if args.count<=0:parser.error("Count must be positive")
    config=json.loads((ROOT/"configs/common_voice22_ja.json").read_text(encoding="utf8"));index_raw=(ROOT/args.index).read_bytes();index=json.loads(index_raw)
    if config["license"]!="CC0-1.0":raise ValueError("Only the reviewed CC0 source is allowed")
    output=ROOT/args.output
    if output.exists():raise ValueError("Use a new output path to preserve the previous snapshot")
    if shutil.disk_usage(ROOT).free<4*1024**3:raise RuntimeError("At least 4 GiB free space required for download and features")
    source_dir=ROOT/"data/common_voice_22_ja";receipts=[]
    def fetch(name):
        path=source_dir/name;url=f"https://huggingface.co/datasets/{config['repo']}/resolve/{config['revision']}/{name}"
        download(url,path)
        with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
        if digest!=config["files"][name]:raise ValueError("Source SHA256 differs: "+name)
        return {"file":name,"sha256":digest,"bytes":path.stat().st_size}
    names=[name for name in config["files"] if not name.endswith(".tar")]
    with ThreadPoolExecutor(max_workers=2) as pool:receipts.extend(pool.map(fetch,names))
    rows=[]
    for split in ("test","dev","train"):
        with (source_dir/f"transcript/ja/{split}.tsv").open(encoding="utf8",newline="") as stream:
            for row in csv.DictReader(stream,delimiter="\t",quoting=csv.QUOTE_NONE):row["source_split"]=split;rows.append(row)
    reserved=[]
    for name in args.reserve_index:reserved+=json.loads((ROOT/name).read_text(encoding="utf8"))["records"]
    protected=[r for r in index["records"] if r["split"]!="train"]+reserved;protected_ids={r["id"] for r in protected};globals_=set()
    for path in (ROOT/"data/common_voice_17").glob("*/*.parquet"):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2048,columns=["client_id","path"]):
            for row in batch.to_pylist():
                if Path(row["path"]).stem in protected_ids:globals_.add(key(str(row["client_id"])))
    failed={r["id"] for r in index.get("cross_version_extension",{}).get("skipped",[])}
    selected=select_candidates((r for r in rows if Path(r["path"]).stem not in failed),index["records"],reserved,globals_,args.count);del rows
    print(f"Selected {len(selected)} cross-version Japanese clips, {len({r['speaker_hash'] for r in selected.values()})} speakers",flush=True)
    names=[f"audio/ja/{split}/ja_{split}_0.tar" for split in ("test","dev","train") if any(r["source_split"]==split for r in selected.values())]
    with ThreadPoolExecutor(max_workers=2) as pool:receipts.extend(pool.map(fetch,names))
    teacher=Teacher("ja",args.threads)
    if teacher.signature!=index["annotation_signatures"]["ja"]:raise ValueError("Teacher changed")
    added=[];skipped=[];started=time.monotonic()
    def save():
        document={**index,"records":index["records"]+sorted(added,key=lambda r:r["id"]),"cross_version_extension":{"source":config,"verified_files":receipts,"base_index_sha256":hashlib.sha256(index_raw).hexdigest(),"selected":len(selected),"added":len(added),"skipped":skipped,"elapsed_seconds":time.monotonic()-started,"reserved_indices":args.reserve_index,"repartition_note":"Unused original Common Voice 22 train/dev/test recordings explicitly assigned to training after global speaker/ID/text exclusions; protected final set not scored"}}
        temp=output.with_suffix(".tmp.json");temp.write_text(json.dumps(document,indent=2)+"\n",encoding="utf8");temp.replace(output)
    def prepare(candidate,blob):
        ident=candidate["id"];audio_sha=hashlib.sha256(blob).hexdigest();destination=ROOT/"cache/features_v2"/(ident+".npz")
        try:
            current=False
            if destination.exists():
                with np.load(destination) as data:current=str(data.get("source_audio_sha256",""))==audio_sha and str(data.get("annotation_signature",""))==teacher.signature
            if not current:
                audio,rate=sf.read(io.BytesIO(blob),dtype="float32",always_2d=True);audio=audio.mean(1)
                if rate!=16000:
                    divisor=gcd(rate,16000);audio=resample_poly(audio,16000//divisor,rate//divisor).astype(np.float32)
                if not .4<=len(audio)/16000<=18:raise ValueError("Duration outside 0.4 to 18 seconds")
                features,times,diagnostics=extract_audio(audio,FeatureConfig());labels,unknown=teacher.annotate(audio,times,diagnostics)
                temp=destination.with_suffix(".tmp.npz");np.savez_compressed(temp,base=features[:,:78].astype(np.float16),y=labels,unknown=unknown,seconds=len(audio)/16000,annotation_signature=teacher.signature,pipeline_version=PIPELINE_VERSION,source_audio_sha256=audio_sha);temp.replace(destination)
            with np.load(destination) as data:return {**candidate,"language":"ja","split":"train","frames":len(data["y"]),"seconds":float(data["seconds"]),"unknown_phones":int(data["unknown"]),"source_audio_sha256":audio_sha,"source_dataset":config["repo"],"source_revision":config["revision"]},None
        except (ValueError,RuntimeError,sf.LibsndfileError) as error:return None,{"id":ident,"reason":str(error)}
    def consume(futures):
        for future in futures:
            record,error=future.result()
            if record is not None:added.append(record)
            if error:skipped.append(error)
        if len(added)%25<len(futures):save();print(f"CV22 features {len(added)}/{len(selected)}, elapsed={time.monotonic()-started:.0f}s",flush=True)
    stopped=False;seen=set()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=set()
        for name in names:
            if stopped:break
            with tarfile.open(source_dir/name,"r:") as archive:
                for member in archive:
                    ident=Path(member.name).stem
                    if not member.isfile() or ident not in selected or ident in seen:continue
                    if output.with_suffix(".STOP").exists():stopped=True;break
                    if member.size>10*1024**2:raise ValueError("Unexpectedly large audio member")
                    while len(pending)>=args.workers:
                        done,pending=wait(pending,return_when=FIRST_COMPLETED);consume(done)
                    seen.add(ident);blob=archive.extractfile(member).read();pending.add(pool.submit(prepare,selected[ident],blob))
        consume(pending)
    save();print(f"Saved {len(added)} CV22 recordings; protected final speech untouched",flush=True)


if __name__=="__main__":main()
