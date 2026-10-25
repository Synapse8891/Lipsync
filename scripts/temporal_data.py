"""Disk-backed acoustic sequences. No waveform data enters the exported model."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.fft import dct
from kotoba_lipsync.features import FeatureConfig,FeatureExtractor,read_audio
from kotoba_lipsync.temporal import expand_features
from kotoba_lipsync.normalization import spectral_residuals

ROOT=Path(__file__).resolve().parents[1]


def apply_gain(base,gain_db):
    """Feature-domain constant waveform gain; MFCC excludes C0 by design."""
    out=np.array(base,dtype=np.float32,copy=True)
    factor=10**(gain_db/10)
    floor=1.01e-10
    out[:,:40]=np.log(np.maximum(np.exp(out[:,:40])-floor,0)*factor+floor)
    out[:,40:53]=dct(out[:,:40],axis=-1,norm="ortho")[:,1:14]
    rms=np.maximum(10**(out[:,68]*80/20)-1e-8,0)
    new_rms=np.sqrt(np.maximum(rms*rms-1e-12,0)*factor+1e-12)
    out[:,68]=20*np.log10(new_rms+1e-8)/80
    return out


def apply_channel_colour(base,curve_db):
    """Training perturbation of mel/MFCC colour; not a full waveform simulator."""
    out=np.array(base,dtype=np.float32,copy=True)
    out[:,:40]+=np.asarray(curve_db,dtype=np.float32)*np.log(10)/10
    out[:,40:53]=dct(out[:,:40],axis=-1,norm="ortho")[:,1:14]
    return out


def audit_augmentation_row(row,records_by_id):
    if not row.get("augmentation"):return
    source=records_by_id.get(row.get("source_recording_id"))
    if source is None or source["split"]!="train" or row["split"]!="train":raise ValueError("Augmentation is restricted to existing training recordings")
    for field in ("language","speaker_hash","transcript_hash"):
        if row.get(field)!=source.get(field):raise ValueError("Augmentation source metadata differs: "+field)
    if row["augmentation"].get("kind")!="waveform-speed-v1":raise ValueError("Unsupported augmentation provenance")

def breath_features(record):
    audio_path=ROOT/record["path"]
    if record.get("audio_license",record.get("license"))!="CC0-1.0":
        raise ValueError("Event recordings must individually declare CC0-1.0")
    actual_hash=hashlib.sha256(audio_path.read_bytes()).hexdigest()
    if record.get("sha256")!=actual_hash:raise ValueError("Event audio hash does not match manifest")
    tail=record.get("event_pipeline")=="tail-v2"
    signature=hashlib.sha256((actual_hash+record.get("kind","breath")+json.dumps(FeatureConfig().to_dict(),sort_keys=True)+("/event-weak-tail-v2" if tail else "/event-weak-v1")).encode()).hexdigest()
    folder=ROOT/"cache/breath_features";folder.mkdir(exist_ok=True)
    path=folder/(record["id"]+"_"+record.get("kind","breath")+("_tail-v2" if tail else "")+".npz")
    if path.exists():
        with np.load(path) as d:
            if "signature" in d and str(d["signature"])==signature:return d["base"],d["y"],d["breath"]
    audio=read_audio(audio_path)
    original_frames=len(audio)//160
    if tail:audio=np.r_[audio,np.zeros(4000,dtype=np.float32)]
    cfg=FeatureConfig();extractor=FeatureExtractor(cfg)
    padded=np.r_[np.zeros(cfg.frame_size-cfg.hop_size,dtype=np.float32),audio]
    frames=[]
    for end in range(cfg.frame_size,len(padded)+1,cfg.hop_size):
        feat,_=extractor.frame(padded[end-cfg.frame_size:end]);frames.append(feat)
    base=np.stack(frames).astype(np.float16)
    db=base[:,68].astype(np.float32)*80
    threshold=max(-65.,float(np.percentile(db[:original_frames],90))-35)
    active=db>threshold
    labels=np.where(active,15 if record.get("kind","breath")=="breath" else 0,0).astype(np.int16)
    breath=np.where(active,1 if record.get("kind","breath")=="breath" else 0,-1).astype(np.float16)
    if tail:
        labels[original_frames:]=0;breath[original_frames:]=-1
    np.savez_compressed(path,base=base,y=labels,breath=breath,signature=signature)
    return base,labels,breath

def pack_corpus(index_path,folder,breath_manifest=None,feature_mode="base"):
    index=json.loads(Path(index_path).read_text(encoding="utf-8"))
    digest=hashlib.sha256(Path(index_path).read_bytes())
    if breath_manifest:digest.update(Path(breath_manifest).read_bytes())
    signature=digest.hexdigest()+"/tcn-corpus-v1"+("/"+feature_mode if feature_mode!="base" else "")
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    metadata=folder/"corpus.json"
    if metadata.exists():
        cached=json.loads(metadata.read_text(encoding="utf-8"))
        if cached.get("signature")==signature:return Corpus(folder)
        raise RuntimeError("Choose a new corpus directory when the source changes")
    extras=[]
    if breath_manifest:
        document=json.loads(Path(breath_manifest).read_text(encoding="utf-8"))
        owners={};identifiers=set()
        for record in document["records"]:
            if record["id"] in identifiers:raise ValueError("Duplicate event recording ID")
            identifiers.add(record["id"])
            if owners.get(record["author"],record["split"])!=record["split"]:raise ValueError("Event uploader occurs in multiple splits")
            owners[record["author"]]=record["split"]
            base,y,br=breath_features(record)
            extras.append(({"id":record["id"],"language":record.get("kind","breath"),"split":record["split"],"speaker_hash":hashlib.sha256(("uploader/"+record["author"]).encode()).hexdigest(),"transcript_hash":record["id"],"frames":len(y),"seconds":len(y)/100,"source":record["source"]},base,y,br))
    rows=index["records"]
    records_by_id={row["id"]:row for row in rows}
    for row in rows:audit_augmentation_row(row,records_by_id)
    count=sum(r["frames"] for r in rows)+sum(len(y) for _,_,y,_ in extras)
    base_map=np.memmap(folder/"features.f16",mode="w+",dtype=np.float16,shape=(count,78))
    y_map=np.memmap(folder/"labels.i16",mode="w+",dtype=np.int16,shape=(count,))
    br_map=np.memmap(folder/"breath.f16",mode="w+",dtype=np.float16,shape=(count,))
    cmn_map=np.memmap(folder/"cmn.f16",mode="w+",dtype=np.float16,shape=(count,40)) if feature_mode in ("cmn-v1","cmn-v2") else None
    scaler=StandardScaler();counts=np.zeros(16,dtype=np.int64);output=[];offset=0
    def write(row,base,y,breath):
        nonlocal offset
        end=offset+len(base)
        base_map[offset:end]=base;y_map[offset:end]=y;br_map[offset:end]=breath
        normalized=spectral_residuals(base,feature_mode) if cmn_map is not None else None
        if cmn_map is not None:cmn_map[offset:end]=normalized
        if row["split"]=="train":
            expanded=expand_features(base[::2])
            if normalized is not None:expanded=np.c_[expanded,normalized[::2]]
            scaler.partial_fit(expanded)
            counts[:]+=np.bincount(y,minlength=16)
        output.append({**row,"offset":offset,"frames":len(base)})
        offset=end
    for n,row in enumerate(rows):
        with np.load(ROOT/row.get("feature_root","cache/features_v2")/(row["id"]+".npz")) as data:
            if str(data["annotation_signature"])!=row.get("annotation_signature",index["annotation_signatures"][row["language"]]):raise RuntimeError("Cache signature mismatch")
            if row.get("augmentation"):
                if str(data["teacher_signature"])!=index["annotation_signatures"][row["language"]] or str(data["source_id"])!=row["source_recording_id"]:raise RuntimeError("Augmentation teacher/source receipt differs")
                if abs(float(data["speed_factor"])-row["augmentation"]["factor"])>1e-6:raise RuntimeError("Augmentation speed differs")
            base=data["base"];y=data["y"]
            breath=np.where(y!=0,0,-1).astype(np.float16)
            write(row,base,y,breath)
        if (n+1)%1000==0:print(f"Packed {n+1}/{len(rows)} recordings",flush=True)
    for row,base,y,br in extras:write(row,base,y,br)
    base_map.flush();y_map.flush();br_map.flush()
    if cmn_map is not None:cmn_map.flush()
    record={"signature":signature,"frames":count,"records":output,"mean":scaler.mean_.tolist(),"scale":scaler.scale_.tolist(),"counts":counts.tolist(),"source_index":str(index_path),"breath_manifest":str(breath_manifest) if breath_manifest else None}
    record["feature_mode"]=feature_mode
    metadata.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
    return Corpus(folder)

class Corpus:
    def __init__(self,folder):
        self.folder=Path(folder)
        self.metadata=json.loads((self.folder/"corpus.json").read_text(encoding="utf-8"))
        n=self.metadata["frames"]
        self.base=np.memmap(self.folder/"features.f16",mode="r",dtype=np.float16,shape=(n,78))
        self.labels=np.memmap(self.folder/"labels.i16",mode="r",dtype=np.int16,shape=(n,))
        self.breath=np.memmap(self.folder/"breath.f16",mode="r",dtype=np.float16,shape=(n,))
        self.cmn=np.memmap(self.folder/"cmn.f16",mode="r",dtype=np.float16,shape=(n,40)) if self.metadata.get("feature_mode") in ("cmn-v1","cmn-v2") else None
        self.rows=self.metadata["records"]
        self.mean=np.asarray(self.metadata["mean"],dtype=np.float32)
        self.scale=np.asarray(self.metadata["scale"],dtype=np.float32)

    def arrays(self,row):
        a=row["offset"];b=a+row["frames"]
        return self.base[a:b],self.labels[a:b],self.breath[a:b]
