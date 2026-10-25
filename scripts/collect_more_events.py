"""Restore the reviewed, versioned CC0 event selection for improvement attempt 2."""
import hashlib
import json
from pathlib import Path
import requests
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]


def main():
    pinned=json.loads((ROOT/"configs/cc0_events_v2.json").read_text(encoding="utf-8"))
    repo=pinned["repo"];revision=pinned["revision"];catalog={}
    folder=ROOT/"data/breath_cc0";folder.mkdir(parents=True,exist_ok=True)
    for split in ("dev","eval"):
        path=ROOT/f"cache/accuracy_research/fsd50k_{split}_info.json"
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            response=requests.get(f"https://huggingface.co/datasets/{repo}/resolve/{revision}/metadata/{split}_clips_info_FSD50K.json",timeout=60)
            response.raise_for_status();path.write_bytes(response.content)
        catalog[split]=json.loads(path.read_text(encoding="utf-8"))
    records=[];owners={}
    for number,selection in enumerate(pinned["records"],1):
        ident=selection["id"];split=selection["source_split"];info=catalog[split][ident]
        if "creativecommons.org/publicdomain/zero/1.0" not in info.get("license",""):raise ValueError("Individual audio is not declared CC0")
        url=f"https://huggingface.co/datasets/{repo}/resolve/{revision}/clips/{split}/{ident}.wav"
        audio=folder/f"fs_{ident}.wav"
        if not audio.exists():
            response=requests.get(url,timeout=60);response.raise_for_status()
            temporary=audio.with_suffix(".part");temporary.write_bytes(response.content);temporary.replace(audio)
        digest=hashlib.sha256(audio.read_bytes()).hexdigest()
        if digest!=selection["sha256"]:raise ValueError("Pinned event audio hash mismatch")
        author=info["uploader"];partition=selection["partition"]
        if owners.get(author,partition)!=partition:raise ValueError("Uploader appears in multiple partitions")
        owners[author]=partition
        records.append({"id":"fs_"+ident,"kind":selection["kind"],"split":partition,"event_pipeline":selection["event_pipeline"],"author":author,"title":info["title"],"description":info.get("description",""),"path":audio.relative_to(ROOT).as_posix(),"source":f"https://freesound.org/people/{author}/sounds/{ident}/","download_url":url,"audio_license":"CC0-1.0","license_url":info["license"],"sha256":digest,"duration":sf.info(audio).duration})
        if number%25==0:print(f"Verified CC0 events {number}/{len(pinned['records'])}",flush=True)
    output={"records":records,"catalog_revision":revision,"audio_license":"CC0-1.0 individual recordings","lookup_reference":"FSD50K catalog, CC BY 4.0, credited in THIRD_PARTY_NOTICES.md; no event annotations imported","prior_test_status":"Attempt 1 is now development only; new test uploaders absent from all earlier event partitions","event_pipeline":"250 ms silent continuation; synthetic tail excluded from breath scoring"}
    (folder/"manifest_v2.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")

if __name__=="__main__":main()
