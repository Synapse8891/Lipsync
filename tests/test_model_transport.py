import json
import socket
import struct
import numpy as np
from kotoba_lipsync.features import FeatureConfig
from kotoba_lipsync.model import TinyModel,export_model
from kotoba_lipsync.runtime import neutral_result,LipSyncEngine
from kotoba_lipsync.transport import osc_message,osc_bundle,OutputSender
from kotoba_lipsync.visemes import phone_to_id,LABELS

def test_quantized_model_agrees_with_float(tmp_path):
    rng=np.random.default_rng(2)
    weights=[rng.normal(size=(10,8)).astype(np.float32),rng.normal(size=(8,3)).astype(np.float32)]
    biases=[rng.normal(size=8).astype(np.float32),rng.normal(size=3).astype(np.float32)]
    path=tmp_path/"test.npz"
    export_model(path,weights,biases,np.zeros(10),np.ones(10),FeatureConfig(),{"classes":[0,10,12]})
    model=TinyModel(path)
    x=rng.normal(size=(300,10)).astype(np.float32)
    expected=np.maximum(x@weights[0]+biases[0],0)@weights[1]+biases[1]
    expected=np.exp(expected-expected.max(axis=1,keepdims=True));expected/=expected.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(model.predict_proba(x)[:,[0,10,12]],expected,atol=.025)
    assert (model.predict_proba(x).argmax(axis=1)==np.array([0,10,12])[expected.argmax(axis=1)]).mean()>.98

def unpack_string(blob,pos):
    end=blob.index(b"\0",pos)
    return blob[pos:end].decode(),(end+4)//4*4

def unpack_message(blob):
    address,pos=unpack_string(blob,0)
    types,pos=unpack_string(blob,pos)
    args=[]
    for typ in types[1:]:
        if typ=="s":value,pos=unpack_string(blob,pos)
        elif typ in ("f","i"):value=struct.unpack_from(">"+typ,blob,pos)[0];pos+=4
        else:value=typ=="T"
        args.append(value)
    assert pos==len(blob)
    return address,args

def unpack_bundle(blob):
    assert blob[:8]==b"#bundle\0"
    assert struct.unpack_from(">Q",blob,8)[0]==1
    pos=16
    out=[]
    while pos<len(blob):
        size=struct.unpack_from(">I",blob,pos)[0];pos+=4
        out.append(unpack_message(blob[pos:pos+size]));pos+=size
    return out

def test_osc_wire_format():
    assert unpack_message(osc_message("/test",7,.25,"口",True,False))==("/test",[7,.25,"口",True,False])
    assert unpack_message(osc_message("/VMC/Ext/Blend/Apply"))==("/VMC/Ext/Blend/Apply",[])

def test_vrchat_uses_only_writable_custom_parameters():
    sender=OutputSender("vrchat")
    packet=sender.packet(neutral_result())
    msgs=unpack_bundle(packet)
    assert len(msgs)==17
    assert all(a.startswith("/avatar/parameters/Kotoba_") for a,_ in msgs)
    assert msgs[-1]==("/avatar/parameters/Kotoba_Enable",[True])
    assert msgs[0]==("/avatar/parameters/Kotoba_sil",[1.0])
    sender.close()

def test_vmc_names_and_commit_order():
    sender=OutputSender("vmc")
    msgs=unpack_bundle(sender.packet(neutral_result()))
    assert [m[1][0] for m in msgs[:-1]]==["A","I","U","E","O"]
    assert msgs[-1]==("/VMC/Ext/Blend/Apply",[])
    sender.close()

def test_udp_delivery_and_neutral_on_shutdown():
    receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1",0));receiver.settimeout(2)
    with OutputSender("unity",port=receiver.getsockname()[1]) as sender:
        sender.send(neutral_result(17))
        packet=json.loads(receiver.recv(8192))
        assert packet["protocol"]=="kotoba-lipsync/1"
        assert packet["sequence"]==17
    reset=json.loads(receiver.recv(8192))
    assert reset["weights"]==[1.]+[0.]*14
    receiver.close()

def test_phoneme_coverage():
    for phone in ["a","I","N","cl","kw","fy","ry"]:
        assert phone_to_id(phone,"ja")>=0
    for phone in ["aɪ","ð","ŋ","ɔː","tʃ","ɜː"]:
        assert phone_to_id(phone,"en")>=0
