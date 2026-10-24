"""Dependency-free OSC 1.0 encoder and loopback-first UDP bridges."""
import json
import socket
import struct
from .visemes import LABELS, VRM_NAMES
from .runtime import neutral_result

def osc_string(value):
    data = value.encode("utf-8") + b"\0"
    return data + b"\0" * (-len(data) % 4)

def osc_message(address, *values):
    if not address.startswith("/") or "\0" in address:
        raise ValueError("OSC address must start with '/' and contain no NUL")
    types, body = ",", b""
    for value in values:
        if isinstance(value, str):
            types += "s"
            body += osc_string(value)
        elif isinstance(value, bool):
            types += "T" if value else "F"
        elif isinstance(value, int):
            types += "i"
            body += struct.pack(">i",value)
        else:
            types += "f"
            body += struct.pack(">f",float(value))
    return osc_string(address) + osc_string(types) + body

def osc_bundle(messages):
    return b"#bundle\0" + struct.pack(">Q",1) + b"".join(struct.pack(">I",len(m))+m for m in messages)

class OutputSender:
    def __init__(self, profile="monitor", host="127.0.0.1", port=None, vrm1=False):
        if profile not in ("monitor","unity","vrchat","vmc"):
            raise ValueError("Unknown output profile")
        self.profile, self.vrm1 = profile, vrm1
        self.address = (host, int(port or {"monitor":39541,"unity":39541,"vrchat":9000,"vmc":39540}[profile]))
        self.socket = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) if profile != "monitor" else None
        self.closed = False

    def packet(self, result, enabled=True):
        if self.profile == "unity":
            data = {"protocol":"kotoba-lipsync/1", **result.to_dict()}
            return json.dumps(data,separators=(",",":"),allow_nan=False).encode("utf-8")
        if self.profile == "vrchat":
            values = dict(zip(LABELS[:15],result.weights))
            values["breath"] = result.breath
            messages = [osc_message(f"/avatar/parameters/Kotoba_{k}",float(v)) for k,v in values.items()]
            messages.append(osc_message("/avatar/parameters/Kotoba_Enable",bool(enabled)))
            return osc_bundle(messages)
        if self.profile == "vmc":
            names = ("aa","ih","ou","ee","oh") if self.vrm1 else VRM_NAMES
            messages = [osc_message("/VMC/Ext/Blend/Val",name,float(value)) for name,value in zip(names,result.vrm)]
            messages.append(osc_message("/VMC/Ext/Blend/Apply"))
            return osc_bundle(messages)
        return b""

    def send(self, result):
        if self.closed:
            raise RuntimeError("Sender is closed")
        if self.socket:
            self.socket.sendto(self.packet(result),self.address)

    def close(self):
        if self.closed:
            return
        try:
            if self.socket:
                neutral = neutral_result()
                self.socket.sendto(self.packet(neutral,enabled=False),self.address)
        finally:
            if self.socket:
                self.socket.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
