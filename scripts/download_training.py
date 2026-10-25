"""Download versioned CC0 Common Voice shards and training-only CTC teachers.

No corpus audio is packaged in releases. Model weights stay in cache/teachers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_REPO = "fixie-ai/common_voice_17_0"
TEACHERS = {
    "ja": ("air-air/japanese-hubert-base-phoneme-ctc-v4-ONNX", "onnx/model_q4.onnx"),
    "en": ("onnx-community/wav2vec2-lv-60-espeak-cv-ft-ONNX", "onnx/model_int8.onnx"),
}

def download(url, path, deadline=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path
    for attempt in range(4):
        if deadline is not None and time.time()>=deadline:
            raise TimeoutError("ダウンロードの計算枠が終了しました。完了ファイルは再利用できます。")
        try:
            with requests.get(url, stream=True, timeout=(20, 180)) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        if deadline is not None and time.time()>=deadline:
                            raise TimeoutError("ダウンロードの計算枠が終了しました")
                        f.write(chunk)
                tmp.replace(path)
            print(f"Downloaded {path.relative_to(ROOT)} ({path.stat().st_size/1e6:.1f} MB)", flush=True)
            return path
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ja-shards", type=int, default=3)
    p.add_argument("--en-shards", type=int, default=1)
    p.add_argument("--deadline",type=float)
    args = p.parse_args()
    if not 1 <= args.ja_shards <= 8:
        p.error("--ja-shards must be between 1 and 8")
    if not 1 <= args.en_shards <= 90:
        p.error("--en-shards must be between 1 and 90")
    pinned=json.loads((ROOT/"configs/training_sources.json").read_text(encoding="utf-8"))
    revision = pinned["dataset"]["revision"]
    jobs = []
    for lang in ("ja", "en"):
        for split in ("train", "validation", "test"):
            for shard in range((args.ja_shards if lang == "ja" else args.en_shards) if split == "train" else 1):
                count = {"ja": 8, "en": {"train": 90, "validation": 19, "test": 19}[split]}[lang]
                file = f"{lang}/{split}-{shard:05d}-of-{count:05d}.parquet"
                jobs.append((f"https://huggingface.co/datasets/{DATA_REPO}/resolve/{revision}/{file}", ROOT / "data" / "common_voice_17" / file))
    source = {"dataset": "Mozilla Common Voice 17.0", "source_repo": DATA_REPO, "revision": revision, "license": "CC0-1.0", "attribution_required": False,
              "original_terms": "https://commonvoice.mozilla.org/en/terms", "files": [u for u, _ in jobs], "teachers": {}}
    for lang, (repo, model_path) in TEACHERS.items():
        rev = pinned["teachers"][lang]["revision"]
        for file in (model_path, "vocab.json", "config.json", "preprocessor_config.json", "README.md"):
            jobs.append((f"https://huggingface.co/{repo}/resolve/{rev}/{file}", ROOT / "cache" / "teachers" / lang / ("model.onnx" if file == model_path else file)))
        source["teachers"][lang] = {"repo": repo, "revision": rev, "license": "Apache-2.0", "purpose": "training-only automatic frame annotations"}
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda pair: download(*pair,deadline=args.deadline), jobs))
    source["sha256"] = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.file_digest(p.open("rb"), "sha256").hexdigest() for _, p in jobs}
    (ROOT / "data" / "provenance.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
