"""Train a small causal TCN on sequences; model selection uses validation only."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from threadpoolctl import threadpool_limits
from kotoba_lipsync.features import FeatureConfig
from kotoba_lipsync.temporal import expand_features,export_temporal,TemporalModel
if __package__:
    from .tcn_network import TemporalNetwork
    from .gru_network import RecurrentNetwork
    from .temporal_data import pack_corpus,apply_gain
    from .train import scores,audit
else:
    from tcn_network import TemporalNetwork
    from gru_network import RecurrentNetwork
    from temporal_data import pack_corpus,apply_gain
    from train import scores,audit

ROOT=Path(__file__).resolve().parents[1]

class SequenceDataset(Dataset):
    def __init__(self,corpus,lookahead,context,frames=128,gain_db=0):
        self.corpus=corpus;self.lookahead=lookahead;self.context=context;self.frames=frames
        self.gain_db=gain_db
        self.segments=[(r,s) for r in corpus.rows if r["split"]=="train" for s in range(0,r["frames"],frames)]
    def __len__(self):return len(self.segments)
    def __getitem__(self,number):
        row,start=self.segments[number]
        base,y,br=self.corpus.arrays(row)
        t=np.arange(start-self.context,start+self.frames)
        valid=(t>=0)&(t<len(base))
        raw=base[np.clip(t,0,len(base)-1)]
        if self.gain_db:raw=apply_gain(raw,np.random.uniform(-self.gain_db,self.gain_db))
        x=(expand_features(raw)-self.corpus.mean)/self.corpus.scale
        x[~valid]=0
        target=t-self.lookahead
        correct=valid&(target>=0)&(target<len(base))&(np.arange(len(t))>=self.context)
        yy=np.full(len(t),-100,dtype=np.int64);bb=np.full(len(t),-1,dtype=np.float32)
        yy[correct]=y[target[correct]];bb[correct]=br[target[correct]]
        return torch.from_numpy(x),torch.from_numpy(yy),torch.from_numpy(bb),torch.from_numpy(valid)

def evaluate(network,corpus,lookahead,split="validation"):
    truths=[];preds=[];languages=[];breath_clips=[]
    device=next(network.parameters()).device
    network.eval()
    with torch.inference_mode():
        for row in corpus.rows:
            if row["split"]!=split:continue
            base,y,br=corpus.arrays(row)
            x=(expand_features(base)-corpus.mean)/corpus.scale
            logits,breath=network(torch.from_numpy(x)[None].to(device))
            p=logits[0,lookahead:].argmax(-1).cpu().numpy()
            target=np.asarray(y[:len(p)])
            if row["language"] in ("ja","en"):
                truths.append(target);preds.append(p);languages.extend([row["language"]]*len(p))
            breath_prob=breath[0,lookahead:].sigmoid().cpu().numpy()
            active=np.asarray(br[:len(p)])>=0
            if active.any():breath_clips.append({"id":row["id"],"positive":row["language"]=="breath","probability":float(np.percentile(breath_prob[active],90))})
    y,p,ll=np.concatenate(truths),np.concatenate(preds),np.array(languages)
    result={"all":scores(y,p),**{lang:scores(y[ll==lang],p[ll==lang]) for lang in ("ja","en")}}
    if any(c["positive"] for c in breath_clips):
        from sklearn.metrics import precision_recall_fscore_support
        yy=np.array([c["positive"] for c in breath_clips]);pp=np.array([c["probability"]>=.5 for c in breath_clips])
        pr,rc,f1,_=precision_recall_fscore_support(yy,pp,average="binary",zero_division=0)
        result["natural_breath"]={"precision":float(pr),"recall":float(rc),"f1":float(f1),"positive_clips":int(yy.sum()),"negative_clips":int((~yy).sum()),"false_positive_rate":float(pp[~yy].mean()),"clip_scores":breath_clips,"annotation":"source-described breathing versus transcribed speech; weak clip labels, no gold frame boundaries"}
    return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--index",default="cache/features_v2/index_five_hour.json")
    parser.add_argument("--corpus",default="cache/temporal_corpus/speech_v1")
    parser.add_argument("--breath-manifest")
    parser.add_argument("--run",default="speech64_v1")
    parser.add_argument("--epochs",type=int,default=50)
    parser.add_argument("--steps-per-epoch",type=int,default=600)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--channels",type=int,default=64)
    parser.add_argument("--dilations",default="1,2,4,8")
    parser.add_argument("--lookahead",type=int,default=3)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--gain-augment-db",type=float,default=0)
    parser.add_argument("--skip-lags",default="")
    parser.add_argument("--skip-hidden",type=int,default=64)
    parser.add_argument("--architecture",choices=["tcn","gru"],default="tcn")
    parser.add_argument("--gru-layers",type=int,default=2)
    parser.add_argument("--english-weight",type=float,default=1.)
    parser.add_argument("--breath-selection-weight",type=float,default=.05)
    parser.add_argument("--patience",type=int,default=18)
    parser.add_argument("--resume")
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    torch.set_num_threads(args.threads);threadpool_limits(limits=args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA build/device unavailable")
    folder=ROOT/"cache/tcn_experiments"/args.run;folder.mkdir(parents=True,exist_ok=True)
    corpus=pack_corpus(ROOT/args.index,ROOT/args.corpus,ROOT/args.breath_manifest if args.breath_manifest else None)
    audit({"records":corpus.rows})
    skip_lags=tuple(int(d) for d in args.skip_lags.split(",") if d)
    network=RecurrentNetwork(args.channels,args.gru_layers) if args.architecture=="gru" else TemporalNetwork(args.channels,tuple(int(d) for d in args.dilations.split(",")),skip_lags=skip_lags,skip_hidden=args.skip_hidden)
    if args.resume:
        checkpoint=torch.load(ROOT/args.resume,map_location="cpu",weights_only=True)
        missing=network.load_state_dict(checkpoint["model"],strict=False)
        if missing.unexpected_keys or any(not key.startswith("skip_") for key in missing.missing_keys):raise ValueError("Checkpoint architecture mismatch")
        corpus.mean=checkpoint["mean"].numpy();corpus.scale=checkpoint["scale"].numpy()
    network.to(device)
    (folder/"run_config.json").write_text(json.dumps({**vars(args),"actual_device":str(device),"torch_version":str(torch.__version__),"corpus_signature":corpus.metadata["signature"],"resume_kind":"weights and feature normalization; optimizer starts fresh"},indent=2),encoding="utf-8")
    dataset=SequenceDataset(corpus,args.lookahead,network.receptive_frames-1,gain_db=args.gain_augment_db)
    weights=np.ones(len(dataset))
    weights[np.array([r[0]["language"]=="en" for r in dataset.segments])]*=args.english_weight
    is_breath=np.array([r[0]["language"]=="breath" for r in dataset.segments])
    is_noise=np.array([r[0]["language"]=="noise" for r in dataset.segments])
    event_fraction=(.15 if is_breath.any() else 0)+(.10 if is_noise.any() else 0)
    speech_mass=weights[~(is_breath|is_noise)].sum()
    if is_breath.any():weights[is_breath]=.15*speech_mass/((1-event_fraction)*is_breath.sum())
    if is_noise.any():weights[is_noise]=.10*speech_mass/((1-event_fraction)*is_noise.sum())
    sampler=WeightedRandomSampler(torch.from_numpy(weights),args.steps_per_epoch*args.batch_size,replacement=True,generator=torch.Generator().manual_seed(args.seed))
    loader=DataLoader(dataset,batch_size=args.batch_size,sampler=sampler,num_workers=0)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.001,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=3)
    counts=np.asarray(corpus.metadata["counts"])
    class_weight=np.minimum((counts.sum()/(16*np.maximum(counts,1)))**.2,3)
    class_weight=torch.from_numpy(class_weight.astype(np.float32)).to(device)
    best=-1.;best_epoch=0;history=[];started=time.monotonic()
    print(f"{args.architecture.upper()} device={device}, parameters={sum(p.numel() for p in network.parameters())}, training_context={network.receptive_frames}, segments={len(dataset)}",flush=True)
    for epoch in range(1,args.epochs+1):
        network.train();losses=[]
        for step,(x,y,b,valid) in enumerate(loader,1):
            x,y,b,valid=(v.to(device) for v in (x,y,b,valid))
            optimizer.zero_grad(set_to_none=True)
            logits,breath=network(x,valid)
            loss=F.cross_entropy(logits.transpose(1,2),y,weight=class_weight,ignore_index=-100,label_smoothing=.03)
            mask=b>=0
            if args.breath_manifest and mask.any():loss=loss+.3*F.binary_cross_entropy_with_logits(breath[mask],b[mask],pos_weight=torch.tensor(3.,device=device))
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
            if step%200==0:print(f"epoch={epoch} step={step}/{args.steps_per_epoch} loss={np.mean(losses):.4f} elapsed={time.monotonic()-started:.0f}s",flush=True)
        validation=evaluate(network,corpus,args.lookahead)
        s=validation["all"];metric=.5*s["speech_accuracy"]+.5*s["macro_f1"]
        if "natural_breath" in validation:metric=(1-args.breath_selection_weight)*metric+args.breath_selection_weight*validation["natural_breath"]["f1"]
        scheduler.step(metric)
        record={"epoch":epoch,"loss":float(np.mean(losses)),"validation":validation,"selection_metric":metric,"elapsed_seconds":time.monotonic()-started}
        history.append(record)
        state={"model":network.state_dict(),"mean":torch.from_numpy(corpus.mean),"scale":torch.from_numpy(corpus.scale),"epoch":epoch}
        torch.save(state,folder/"last.pt")
        if metric>best:
            best,best_epoch=metric,epoch;torch.save(state,folder/"best.pt")
            from kotoba_lipsync.recurrent import export_recurrent
            exporter=export_recurrent if args.architecture=="gru" else export_temporal
            exporter(folder/"best_int8.npz",{k:v.detach().cpu().numpy() for k,v in network.state_dict().items()},corpus.mean,corpus.scale,FeatureConfig(lookahead_frames=args.lookahead),{"channels":args.channels,"gru_layers":args.gru_layers,"architecture":args.architecture,"dilations":list(network.dilations),"skip_lags":list(skip_lags),"skip_hidden":args.skip_hidden,"gain_augmentation_db":args.gain_augment_db,"has_breath_head":bool(args.breath_manifest),"best_epoch":epoch,"seed":args.seed,"pipeline_version":"ctc-midpoint-v2","receptive_frames":network.receptive_frames if args.architecture=="tcn" else None,"training_warmup_frames":network.receptive_frames-1,"labels":["sil","PP","FF","TH","DD","kk","CH","SS","nn","RR","aa","E","ih","oh","ou","HH"],"recommended_smoothing_ms":10,"training_device":str(device),"torch_version":str(torch.__version__)})
        (folder/"history.json").write_text(json.dumps(history,indent=2),encoding="utf-8")
        print(f"EPOCH {epoch}: validation={s['accuracy']:.4f} speech={s['speech_accuracy']:.4f} macroF1={s['macro_f1']:.4f} breath={validation.get('natural_breath',{}).get('f1')} best={best_epoch}",flush=True)
        if epoch-best_epoch>=args.patience:break
        if (folder/"STOP").exists():
            print("Cooperative stop requested; checkpoints retained.",flush=True)
            break
    print(f"Completed. Best epoch={best_epoch}. Candidate saved in {folder}. No test set used and production model unchanged.",flush=True)

if __name__=="__main__":main()
