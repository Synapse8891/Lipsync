"""Independently select individual CC0 breath/noise recordings by uploader metadata.

The mixed-license FSD50K catalog is a lookup source only: its event annotations
and splits are not imported. Only clips explicitly marked CC0 are downloaded.
"""
import hashlib
import json
from pathlib import Path
import requests
import re
import soundfile as sf
import sys
import argparse

ROOT=Path(__file__).resolve().parents[1]
REPO="Fhrozen/FSD50k"
EXCLUDED_IDS={"233137","427163","422161","390769"}  # animal or mixed impact/cough recordings

def classify(info):
    title=info["title"].lower();description=info.get("description","").lower()
    words=("dog","cat","sheep","oveja","chihuahua","animal","pig","horse","monster","zombie","goblin","creepy","demon","dragon","robot","trumpet","didger","breathump","balloon","bottle","underwater","scuba","burp","cigarette","smok","iron_","instrument","whimper","scream","yell","huuuah","wooden floor","sigh","spit","snor","riser","distort","sobbing","mask","dulled")
    edited=("paulstretch","distort","transform","pitch","numerous effects","put some effects","processed","edited in audacity to sound")
    if any(x in title for x in words) or any(x in description for x in edited):return None
    if any(x in title for x in ("breath","inhale","exhale","exhala","respiraci","sigh")):
        if "nonesense talk" in description:return None
        return "breath"
    if any(x in title for x in ("white noise","pink noise","air conditioner","room tone","rustl")) or re.search(r"\b(fan|wind|rain)\b",title):return "noise"
    return None

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--discover",action="store_true",help="Explore metadata instead of the fixed reviewed selection")
    args=parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    folder=ROOT/"data/breath_cc0";folder.mkdir(exist_ok=True)
    pinned=json.loads((ROOT/"configs/cc0_events.json").read_text(encoding="utf-8")) if not args.discover else None
    selected={r["id"]:r for r in pinned["records"]} if pinned else None
    if pinned:revision=pinned["revision"]
    else:
        response=requests.get(f"https://huggingface.co/api/datasets/{REPO}",timeout=30);response.raise_for_status()
        revision=response.json()["sha"]
    records=[];seen=set();per_author={}
    for split in ("dev","eval"):
        file=ROOT/f"cache/accuracy_research/fsd50k_{split}_info.json"
        if not file.exists():
            r=requests.get(f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/metadata/{split}_clips_info_FSD50K.json",timeout=60);r.raise_for_status();file.write_bytes(r.content)
        catalog=json.loads(file.read_text(encoding="utf-8"))
        for ident,info in catalog.items():
            if selected is not None and (ident not in selected or selected[ident]["source_split"]!=split):continue
            if ident in EXCLUDED_IDS:continue
            if "creativecommons.org/publicdomain/zero/1.0" not in info.get("license",""):continue
            kind=selected[ident]["kind"] if selected is not None else classify(info)
            if kind is None or ident in seen:continue
            author=info["uploader"]
            if per_author.get((author,kind),0)>=4:continue
            if sum(r["kind"]==kind for r in records)>=(90 if kind=="breath" else 40):continue
            audio=folder/f"fs_{ident}.wav"
            url=f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/clips/{split}/{ident}.wav"
            if not audio.exists():
                response=requests.get(url,timeout=60)
                if not response.ok:continue
                audio.write_bytes(response.content)
            data=sf.info(audio)
            if not .25<=data.duration<=90:continue
            digest=hashlib.sha256(audio.read_bytes()).hexdigest()
            if selected is not None and digest!=selected[ident]["sha256"]:raise ValueError("Downloaded event recording hash differs from pinned selection")
            description=info.get("description","")
            record={"id":"fs_"+ident,"author":author,"title":info["title"],"description":description,"kind":kind,"duration":data.duration,"source":f"https://freesound.org/people/{author}/sounds/{ident}/","download_url":url,"audio_license":"CC0-1.0","license":"CC0-1.0","license_url":info["license"],"path":str(audio.relative_to(ROOT)).replace("\\","/"),"sha256":hashlib.sha256(audio.read_bytes()).hexdigest(),"label_kind":"independent selection from uploader title/description; weak clip event label","metadata_lookup":f"https://huggingface.co/datasets/{REPO}/blob/{revision}/metadata/{split}_clips_info_FSD50K.json"}
            records.append(record);seen.add(ident);per_author[(author,kind)]=per_author.get((author,kind),0)+1
            print(kind,ident,author,info["title"],flush=True)
    authors=sorted({r["author"] for r in records},key=lambda a:hashlib.sha256(("natural-breath-v1/"+a).encode()).hexdigest())
    # Allocate breath uploaders first so every partition has positive examples.
    breath_authors=[a for a in authors if any(r["author"]==a and r["kind"]=="breath" for r in records)]
    count=max(3,len(breath_authors)//5)
    groups={a:("test" if i<count else "validation" if i<2*count else "train") for i,a in enumerate(breath_authors)}
    for i,a in enumerate(a for a in authors if a not in groups):groups[a]=("test","validation","train","train","train")[i%5]
    for record in records:record["split"]=selected[record["id"][3:]]["partition"] if selected is not None else groups[record["author"]]
    if selected is not None and len(records)!=len(selected):raise RuntimeError("Some pinned CC0 events are missing; refusing a partial manifest")
    output={"audio_license":"CC0-1.0 individual audio files only","catalog_revision":revision,"metadata_reference":"FSD50K catalog (CC BY 4.0), credited as lookup; no FSD50K event annotations or standard splits imported","split_kind":"uploader-disjoint; not a claim of known performer identity","records":records}
    (folder/"manifest.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({f"{kind}/{s}":sum(r["kind"]==kind and r["split"]==s for r in records) for kind in ("breath","noise") for s in ("train","validation","test")}))

if __name__=="__main__":main()
