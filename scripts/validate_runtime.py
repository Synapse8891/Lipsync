"""Check an actual release model with local, reproducible audio input."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from kotoba_lipsync.features import read_audio,extract_audio,FeatureConfig
from kotoba_lipsync.model import load_model
from kotoba_lipsync.runtime import LipSyncEngine


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("audio",type=Path)
    parser.add_argument("--model",type=Path,default=Path("models/kotoba_int8.npz"))
    parser.add_argument("--output",type=Path,default=Path("reports/runtime_validation.json"))
    args=parser.parse_args();threadpool_limits(1)
    audio=read_audio(args.audio)
    base=extract_audio(audio,FeatureConfig(lookahead_frames=0))[0][:,:78]
    model=load_model(args.model);batch,breaths=model.predict_sequence(base)
    stream=np.stack([model.predict_proba(frame) for frame in base])
    error=float(np.max(np.abs(batch-stream)))
    assert error<1e-5,error
    def process(sizes):
        engine=LipSyncEngine(args.model);frames=[];offset=0;number=0
        while offset<len(audio):
            length=sizes[number%len(sizes)];frames.extend(engine.push(audio[offset:offset+length]));offset+=length;number+=1
        return frames
    a=process([160]);b=process([17,4096,123,512])
    values=lambda frames:np.array([f.weights+[f.breath,f.timestamp] for f in frames])
    chunk_error=float(np.max(np.abs(values(a)-values(b))))
    assert chunk_error<1e-5,chunk_error
    assert np.isfinite(values(a)).all()
    engine=LipSyncEngine(args.model);engine.push(audio)
    tail=engine.push(np.zeros(16000,dtype=np.float32))
    assert tail[-1].viseme=="sil" and tail[-1].weights[0]>.999 and tail[-1].breath<1e-6
    report={"status":"ok","model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),"audio_seconds":len(audio)/16000,"output_frames":len(a),"batch_stream_max_error":error,"different_audio_block_sizes_max_error":chunk_error,"finite_outputs":True,"silence_recovery":True,"natural_breath_is_independent":bool(model.has_breath_head),"scope":"CPU inference and stream state only; no real avatar, microphone driver or rendered response measurement"}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf8");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
