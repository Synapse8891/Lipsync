"""Microphone capture. Inference stays off the audio callback thread."""
import queue
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd

def input_devices():
    return [(i,d["name"],int(d["default_samplerate"])) for i,d in enumerate(sd.query_devices()) if d["max_input_channels"]>0]

class Microphone:
    def __init__(self, device=None, target_rate=16000):
        self.queue = queue.Queue(maxsize=8)
        self.dropped = 0
        self.status = ""
        info = sd.query_devices(device,"input")
        self.rate = int(info["default_samplerate"])
        # Use an exact 10 ms chunk so resampling is continuous on normal devices.
        if self.rate % 100:
            raise ValueError("Input device must support a sample rate divisible by 100 (16/44.1/48 kHz)")
        self.target_rate = target_rate
        self.div = gcd(self.rate,target_rate)
        self.resampler = StreamingResampler(self.rate,target_rate)
        self.stream = sd.InputStream(device=device,samplerate=self.rate,channels=1,dtype="float32",blocksize=self.rate//100,callback=self._callback)

    def _callback(self, data, frames, timing, status):
        if status:
            self.status = str(status)
        try:
            self.queue.put_nowait(data[:,0].copy())
        except queue.Full:
            self.dropped += 1
            # Drop oldest samples to bound latency.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.queue.put_nowait(data[:,0].copy())
            except queue.Full:
                pass

    def read(self, timeout=.2):
        raw = self.queue.get(timeout=timeout)
        return self.resampler.push(raw)

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self,*args):
        self.stream.stop()
        self.stream.close()

class StreamingResampler:
    """Polyphase resampling with overlap and fixed 10 ms blocks.

An extra 10 ms of input is buffered for the symmetric anti-alias filter. Keeping
overlap prevents the clicks caused by resetting resample_poly for every block.
"""
    def __init__(self, source_rate, target_rate=16000):
        self.source_rate, self.target_rate = source_rate,target_rate
        self.block = source_rate//100
        if source_rate % 100 or target_rate % 100:
            raise ValueError("Use an audio rate divisible by 100")
        self.pending = np.empty(0,dtype=np.float32)
        self.left = np.zeros(self.block,dtype=np.float32)
        self.div = gcd(source_rate,target_rate)

    def push(self, samples):
        if self.source_rate == self.target_rate:
            return np.asarray(samples,dtype=np.float32)
        self.pending = np.r_[self.pending,np.asarray(samples,dtype=np.float32)]
        out=[]
        while len(self.pending)>=2*self.block:
            window=np.r_[self.left,self.pending[:2*self.block]]
            converted=resample_poly(window,self.target_rate//self.div,self.source_rate//self.div)
            hop=self.target_rate//100
            out.append(converted[hop:2*hop])
            self.left=self.pending[:self.block].copy()
            self.pending=self.pending[self.block:]
        return np.concatenate(out).astype(np.float32) if out else np.empty(0,dtype=np.float32)
