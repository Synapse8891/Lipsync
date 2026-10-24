"""Create speaker/transcript-disjoint pseudo labels from real CC0 speech.

CTC phone spikes are expanded at their temporal midpoints. These labels are NOT
human phoneme boundaries and do NOT certify true lip motion or breath accuracy.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from math import gcd
import time
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
from scipy.signal import resample_poly
import soundfile as sf
from kotoba_lipsync.features import FeatureConfig, extract_audio
from kotoba_lipsync.visemes import phone_to_id, LABELS
from kotoba_lipsync.visemes import JP_PHONES, IPA_PHONES

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "ctc-midpoint-v2"
FEATURE_DIR = ROOT / "cache" / "features_v2"

def prepare_teacher_input(audio, preprocessing):
    """Respect each teacher's feature-extractor config; Japanese is NOT normalized."""
    if "do_normalize" not in preprocessing or preprocessing.get("sampling_rate") != 16000:
        raise ValueError("教師の前処理設定が不正です。download_training.pyを実行してください。")
    values=np.asarray(audio,dtype=np.float32)
    if values.ndim!=1 or not np.isfinite(values).all():
        raise ValueError("Teacher input must be finite mono audio")
    if preprocessing["do_normalize"]:
        values=(values-values.mean())/np.sqrt(values.var()+1e-7)
    return values

def cache_is_current(path, signature):
    if not path.exists():return False
    try:
        with np.load(path,allow_pickle=False) as data:
            return str(data["annotation_signature"])==signature and str(data["pipeline_version"])==PIPELINE_VERSION
    except (ValueError,KeyError,OSError,EOFError):
        return False

def key(value):
    return hashlib.sha256(str(value).encode()).hexdigest()

def select_records(limits):
    selected = []
    seen_audio = set()
    owners = {}
    text_owners = {}
    for split in ("test", "validation", "train"):
        for lang in ("ja", "en"):
            count = 0
            candidates = []
            for path in sorted((ROOT / "data" / "common_voice_17" / lang).glob(f"{split}-*.parquet")):
                for batch in pq.ParquetFile(path).iter_batches(batch_size=64):
                    for row in batch.to_pylist():
                        speaker = key(lang + str(row["client_id"]))
                        text = key(lang + str(row["sentence"]).strip())
                        filename = Path(row["audio"]["path"] or row["path"]).stem
                        if row.get("down_votes", 0) > row.get("up_votes", 0) or not row["client_id"]:
                            continue
                        if owners.get(speaker, split) != split or text_owners.get(text, split) != split or filename in seen_audio:
                            continue
                        candidates.append((key(filename), filename, speaker, text, row["audio"]["bytes"]))
            candidates.sort(key=lambda r: r[0])
            # Round robin prevents a prolific speaker from dominating this subset.
            groups = {}
            for item in candidates:
                groups.setdefault(item[2], []).append(item)
            index = 0
            while count < limits[lang][split]:
                progress = False
                for speaker in sorted(groups):
                    if index >= len(groups[speaker]):
                        continue
                    _, name, speaker, text, blob = groups[speaker][index]
                    if name in seen_audio or text_owners.get(text, split) != split:
                        continue
                    owners[speaker] = split
                    text_owners[text] = split
                    seen_audio.add(name)
                    selected.append((lang, split, name, speaker, text, blob))
                    count += 1
                    progress = True
                    if count == limits[lang][split]:
                        break
                if not progress:
                    break
                index += 1
            if count < limits[lang][split]:
                raise RuntimeError(f"Only {count} eligible {lang}/{split} recordings; requested {limits[lang][split]}")
            print(f"Selected {lang}/{split}: {count} recordings, {len({r[3] for r in selected if r[0]==lang and r[1]==split})} speakers", flush=True)
    return selected

class Teacher:
    def __init__(self, lang, threads=4):
        path = ROOT / "cache" / "teachers" / lang
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path / "model.onnx"), options, providers=["CPUExecutionProvider"])
        vocab = json.loads((path / "vocab.json").read_text(encoding="utf-8"))
        self.vocab = {int(v): k for k, v in vocab.items()}
        self.lang = lang
        self.blank = vocab.get("PAD", vocab.get("<pad>", 0))
        config_path=path/"preprocessor_config.json"
        if not config_path.exists():
            raise RuntimeError("教師の前処理設定がありません。先に scripts/download_training.py を実行してください。")
        self.preprocessing=json.loads(config_path.read_text(encoding="utf-8"))
        prepare_teacher_input(np.zeros(16,dtype=np.float32),self.preprocessing)
        with (path/"model.onnx").open("rb") as handle:
            model_hash=hashlib.file_digest(handle,"sha256").hexdigest()
        signature_data={"version":PIPELINE_VERSION,"model":model_hash,"vocab":vocab,"preprocessing":self.preprocessing,"features":FeatureConfig().to_dict(),"jp_phones":JP_PHONES,"ipa_phones":IPA_PHONES}
        self.signature=key(json.dumps(signature_data,sort_keys=True))

    def annotate(self, audio, times, diagnostics):
        values = prepare_teacher_input(audio,self.preprocessing)
        logits = self.session.run(None, {"input_values": values[None].astype(np.float32)})[0][0]
        tokens = np.argmax(logits, axis=-1)
        spikes = []
        previous = -1
        unknown = 0
        for i, token in enumerate(tokens):
            if token != self.blank and token != previous:
                phone = self.vocab[int(token)]
                label = phone_to_id(phone, self.lang)
                if label >= 0:
                    spikes.append((i * .02 + .0125, label))
                elif phone not in ("EOS", "SOS", "UNK", "<unk>", "<s>", "</s>", "|"):
                    unknown += 1
            previous = int(token)
        if not spikes:
            raise ValueError("No recognizable phone spikes")
        st = np.array([x[0] for x in spikes])
        sy = np.array([x[1] for x in spikes], dtype=np.int16)
        midpoints = (st[1:] + st[:-1]) / 2
        labels = sy[np.searchsorted(midpoints, times)]
        db = np.array([d["rms_db"] for d in diagnostics])
        # A fixed absolute gate only; relative peak gates can erase quiet speech.
        labels[db < -55] = 0
        return labels, unknown

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ja-train", type=int, default=1800)
    p.add_argument("--en-train", type=int, default=200)
    p.add_argument("--ja-eval", type=int, default=180)
    p.add_argument("--en-eval", type=int, default=40)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    limits = {lang: {"train": tr, "validation": ev, "test": ev} for lang,tr,ev in [("ja",args.ja_train,args.ja_eval),("en",args.en_train,args.en_eval)]}
    records = select_records(limits)
    config = FeatureConfig()
    out = FEATURE_DIR
    out.mkdir(parents=True, exist_ok=True)
    index = []
    errors = []
    started = time.perf_counter()
    signatures={}
    preprocessing={}
    for lang in ("ja", "en"):
        teacher = Teacher(lang, args.threads)
        signatures[lang]=teacher.signature
        preprocessing[lang]=teacher.preprocessing
        subset = [r for r in records if r[0] == lang]
        for i, (_, split, name, speaker, text, blob) in enumerate(subset):
            dest = out / f"{name}.npz"
            try:
                if not cache_is_current(dest,teacher.signature):
                    audio, sr = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
                    audio = audio.mean(axis=1)
                    d = gcd(sr, config.sample_rate)
                    if sr != config.sample_rate:
                        audio = resample_poly(audio, config.sample_rate // d, sr // d).astype(np.float32)
                    if not .4 <= len(audio) / config.sample_rate <= 18:
                        raise ValueError("Duration outside 0.4 to 18 seconds")
                    features, times, diag = extract_audio(audio, config)
                    labels, unknown = teacher.annotate(audio, times, diag)
                    # Store only the current frame; temporal stacking is rebuilt at training.
                    width = features.shape[1] // len(config.context)
                    temporary=dest.with_suffix(".tmp.npz")
                    np.savez_compressed(temporary, base=features[:, :width].astype(np.float16), y=labels, unknown=unknown, seconds=len(audio) / config.sample_rate, annotation_signature=teacher.signature,pipeline_version=PIPELINE_VERSION)
                    temporary.replace(dest)
                    if split == "test" and i < 3:
                        sample_dir = ROOT / "data" / "demo"
                        sample_dir.mkdir(exist_ok=True)
                        sf.write(sample_dir / f"{name}.wav", audio, config.sample_rate)
                with np.load(dest, allow_pickle=False) as data:
                    index.append({"id": name, "language": lang, "split": split, "speaker_hash": speaker, "transcript_hash": text, "frames": len(data["y"]), "seconds": float(data["seconds"]), "unknown_phones": int(data["unknown"])})
            except (ValueError, RuntimeError, sf.LibsndfileError) as e:
                errors.append({"id": name, "reason": str(e)})
            if (i + 1) % 25 == 0 or i + 1 == len(subset):
                print(f"{lang}: {i+1}/{len(subset)}, elapsed {time.perf_counter()-started:.0f}s, skipped {len(errors)}", flush=True)
        del teacher
    if not index:
        raise RuntimeError("No usable audio")
    document = {"feature_config": config.to_dict(), "labels": list(LABELS), "label_kind": "CTC teacher pseudo labels with midpoint boundaries; absolute silence gate -55 dBFS", "breath_note": "The breath label includes /h/. Natural inhalations are not annotated or independently validated.", "records": index, "skipped": errors, "limits": limits}
    document.update(pipeline_version=PIPELINE_VERSION,annotation_signatures=signatures,teacher_preprocessing=preprocessing)
    temporary=out/"index.tmp.json"
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(out/"index.json")
    print(f"Wrote {len(index)} records; skipped {len(errors)}", flush=True)

if __name__ == "__main__":
    main()
