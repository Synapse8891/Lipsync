import argparse
import hashlib
import json
from pathlib import Path
import queue
import time
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from .features import read_audio
from .runtime import LipSyncEngine
from .transport import OutputSender

ROOT = Path(sys.executable).resolve().parent if getattr(sys,"frozen",False) else Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "kotoba_int8.npz"

def main():
    p=argparse.ArgumentParser(description="軽量日本語・英語リップシンク")
    p.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("devices",help="入力デバイス一覧")
    for name in ("file","mic"):
        s=sub.add_parser(name)
        s.add_argument("--profile",choices=["monitor","unity","vrchat","vmc"],default="monitor")
        s.add_argument("--host",default="127.0.0.1")
        s.add_argument("--port",type=int)
        s.add_argument("--silence-db",type=float,default=-55)
        s.add_argument("--smoothing-ms",type=float,default=None,help="Smoothing time; default comes from model metadata")
        s.add_argument("--vrm1",action="store_true")
        if name=="file":
            s.add_argument("input",type=Path)
            s.add_argument("--output",type=Path,default=Path("outputs/visemes.jsonl"))
            s.add_argument("--realtime",action="store_true")
        else:
            s.add_argument("--device",type=int)
            s.add_argument("--seconds",type=float,default=0)
    s=sub.add_parser("benchmark")
    s.add_argument("input",type=Path)
    s.add_argument("--output",type=Path,default=Path("reports/benchmark.json"))
    args=p.parse_args()
    threadpool_limits(limits=1)
    if args.command=="devices":
        from .audio import input_devices
        for device,name,rate in input_devices():
            print(f"{device}: {name} ({rate} Hz)")
        return
    if not args.model.exists():
        p.error(f"学習済みモデルがありません: {args.model}")
    engine=LipSyncEngine(args.model,getattr(args,"silence_db",-55),getattr(args,"smoothing_ms",None))
    if args.command=="benchmark":
        audio=read_audio(args.input)
        # Warm up outside the timed region.
        engine.push(np.zeros(1600,dtype=np.float32))
        engine=LipSyncEngine(args.model)
        times=[];model_times=[]
        begin=time.perf_counter()
        for i in range(0,len(audio),160):
            start=time.perf_counter()
            frames=engine.push(audio[i:i+160])
            times.append((time.perf_counter()-start)*1000)
            model_times.extend(frame.inference_ms for frame in frames)
        elapsed=time.perf_counter()-begin
        result={"audio_seconds":len(audio)/16000,"elapsed_seconds":elapsed,"real_time_factor":elapsed/(len(audio)/16000),"frame_ms_p50":float(np.percentile(times,50)),"frame_ms_p95":float(np.percentile(times,95)),"frame_ms_p99":float(np.percentile(times,99)),"parameters":engine.model.parameters,"model_bytes":args.model.stat().st_size,"model_float_weights_bytes":engine.model.float_weights_bytes,"analysis_window_ms":25,"lookahead_ms":engine.model.config.lookahead_frames*10,"resampler_buffer_ms_at_48khz":10,"smoothing_ms":engine.smoothing_ms,"scope":"CPU feature extraction + inference; excludes audio driver, network, rendering and application startup"}
        result.update({"model_sha256":hashlib.sha256(args.model.read_bytes()).hexdigest(),"configured_frame_rate_hz":100,"output_frames":len(model_times),"processing_hops_per_second":len(times)/elapsed,"model_ms_p50":float(np.percentile(model_times,50)) if model_times else None,"model_ms_p95":float(np.percentile(model_times,95)) if model_times else None,"analysis_center_delay_ms":12.5,"timing_note":f"Throughput is offline capacity, not the live update rate. The {engine.smoothing_ms:g} ms smoothing time is an EMA time constant, not a fixed deadline."})
        if sys.platform=="win32":
            import ctypes
            from ctypes import wintypes
            class MemoryCounters(ctypes.Structure):
                _fields_=[("cb",wintypes.DWORD),("PageFaultCount",wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ("PeakWorkingSetSize","WorkingSetSize","QuotaPeakPagedPoolUsage","QuotaPagedPoolUsage","QuotaPeakNonPagedPoolUsage","QuotaNonPagedPoolUsage","PagefileUsage","PeakPagefileUsage")]
            counters=MemoryCounters()
            counters.cb=ctypes.sizeof(counters)
            get_current=ctypes.windll.kernel32.GetCurrentProcess
            get_current.restype=wintypes.HANDLE
            get_memory=ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD]
            if get_memory(get_current(),ctypes.byref(counters),counters.cb):
                result["process_working_set_bytes"]=counters.WorkingSetSize
                result["memory_scope"]="This CLI process after imports and inference; GUI/driver/receiver memory excluded"
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
        print(json.dumps(result,indent=2))
        return
    with OutputSender(args.profile,args.host,args.port,args.vrm1) as sender:
        if args.command=="file":
            audio=read_audio(args.input)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            begin=time.perf_counter()
            count=0
            with args.output.open("w",encoding="utf-8") as f:
                for i in range(0,len(audio),160):
                    for result in engine.push(audio[i:i+160]):
                        if args.realtime:
                            delay=(i+160)/16000-(time.perf_counter()-begin)
                            if delay>0:
                                time.sleep(delay)
                        sender.send(result)
                        f.write(json.dumps(result.to_dict(),separators=(",",":"))+"\n")
                        count+=1
            print(f"{count} frames -> {args.output}")
        else:
            from .audio import Microphone
            last=0
            begin=time.perf_counter()
            last_audio=begin
            reset_sent=False
            try:
                with Microphone(args.device) as mic:
                    print(f"Listening ({mic.rate} Hz -> 16000 Hz). Ctrl+C to stop.")
                    while not args.seconds or time.perf_counter()-begin<args.seconds:
                        try:
                            samples=mic.read()
                        except queue.Empty:
                            if time.perf_counter()-last_audio>.25 and not reset_sent:
                                from .runtime import neutral_result
                                sender.send(neutral_result())
                                engine=LipSyncEngine(args.model,args.silence_db,args.smoothing_ms)
                                reset_sent=True
                            continue
                        last_audio=time.perf_counter()
                        reset_sent=False
                        for result in engine.push(samples):
                            sender.send(result)
                            if time.perf_counter()-last>.2:
                                print(f"\r{result.viseme:>6}  {result.rms_db:6.1f} dB  model {result.inference_ms:.2f} ms  drops={mic.dropped}",end="",flush=True)
                                last=time.perf_counter()
            except KeyboardInterrupt:
                print("\nStopped.")

if __name__=="__main__":
    main()
