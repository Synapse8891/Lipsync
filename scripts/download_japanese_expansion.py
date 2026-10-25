"""Fetch the remaining pinned Japanese CC0 shards; no repartitioning or scoring."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from scripts.download_training import download

ROOT=Path(__file__).resolve().parents[1]


def main():
    source=json.loads((ROOT/"configs/training_sources.json").read_text(encoding="utf8"))["dataset"]
    names=[f"ja/validation-{i:05d}-of-00008.parquet" for i in range(1,8)]+[f"ja/test-{i:05d}-of-00008.parquet" for i in range(4,8)]
    def fetch(name):
        url=f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/{name}"
        path=ROOT/"data/common_voice_17"/name;download(url,path)
        with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
        return {"source_file":name,"url":url,"sha256":digest,"bytes":path.stat().st_size}
    with ThreadPoolExecutor(max_workers=2) as pool:files=list(pool.map(fetch,names))
    report={"source":source,"audio_license":"CC0-1.0","purpose":"Additional training candidates only after recording, global speaker and transcript exclusion; original source test/validation names are not the final model split","files":files}
    (ROOT/"reports/experiments/2026-10-25_japanese_expansion_sources.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(f"Verified {len(files)} pinned CC0 Japanese shards",flush=True)


if __name__=="__main__":main()
