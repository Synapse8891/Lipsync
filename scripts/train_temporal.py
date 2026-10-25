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
from kotoba_lipsync.normalization import sequence_features
from kotoba_lipsync.normalization import CausalSpectralMean
from scripts.temporal_data import apply_channel_colour
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
from scripts.attention_network import AttentionNetwork
from scripts.network_tools import infer_sequence


def save_checkpoint(state,path):
    temporary=path.with_suffix(".tmp.pt");torch.save(state,temporary);temporary.replace(path)


def model_feature_config(lookahead):
    context=FeatureConfig().context
    if lookahead>max(context):context=context+(lookahead,)
    return FeatureConfig(context=context,lookahead_frames=lookahead)

class SequenceDataset(Dataset):
    def __init__(self,corpus,lookahead,context,frames=128,gain_db=0,events_only=False,channel_db=0,context_augment=False,random_crop=False):
        self.corpus=corpus;self.lookahead=lookahead;self.context=context;self.frames=frames
        self.gain_db=gain_db
        self.events_only=events_only
        self.channel_db=channel_db;self.context_augment=context_augment
        self.random_crop=random_crop
        self.voice_rows=[row for row in corpus.rows if row["split"]=="train" and row["language"] in ("ja","en")] if context_augment else []
        self.segments=[(r,s) for r in corpus.rows if r["split"]=="train" for s in range(0,r["frames"],frames)]
    def __len__(self):return len(self.segments)
    def __getitem__(self,number):
        row,start=self.segments[number]
        base,y,br=self.corpus.arrays(row)
        if self.random_crop and len(base)>self.frames and np.random.random()<.8:start=np.random.randint(len(base)-self.frames+1)
        t=np.arange(start-self.context,start+self.frames)
        valid=(t>=0)&(t<len(base))
        raw=base[np.clip(t,0,len(base)-1)]
        if self.gain_db and (not self.events_only or row["language"] not in ("ja","en")):raw=apply_gain(raw,np.random.uniform(-self.gain_db,self.gain_db))
        if self.channel_db:
            curve=np.interp(np.arange(40),np.linspace(0,39,5),np.random.uniform(-self.channel_db,self.channel_db,5));curve-=curve.mean()
            raw=apply_channel_colour(raw,curve)
        expanded=expand_features(raw)
        if getattr(self.corpus,"cmn",None) is not None:
            extra=np.array(self.corpus.cmn[row["offset"]+np.clip(t,0,len(base)-1)],dtype=np.float32)
            if self.context_augment:
                if row["language"] in ("ja","en"):
                    if np.random.random()<.1:extra[:]=0
                elif self.voice_rows and np.random.random()<.7:
                    for _ in range(8):
                        voice=self.voice_rows[np.random.randint(len(self.voice_rows))];position=voice["offset"]+np.random.randint(voice["frames"])
                        probe=self.corpus.base[position]
                        if probe[75]>.4 and probe[68]*80>-60:
                            normalizer=CausalSpectralMean(mode=self.corpus.metadata.get("feature_mode","cmn-v1"))
                            normalizer.mean=probe[:40].astype(np.float32)-self.corpus.cmn[position].astype(np.float32);normalizer.count=300
                            extra=np.stack([normalizer.push(frame) if is_valid else np.zeros(40,dtype=np.float32) for frame,is_valid in zip(raw,valid)])
                            break
            expanded=np.c_[expanded,extra]
        x=(expanded-self.corpus.mean)/self.corpus.scale
        x[~valid]=0
        target=t-self.lookahead
        correct=valid&(target>=0)&(target<len(base))&(np.arange(len(t))>=self.context)
        yy=np.full(len(t),-100,dtype=np.int64);bb=np.full(len(t),-1,dtype=np.float32)
        yy[correct]=y[target[correct]];bb[correct]=br[target[correct]]
        return torch.from_numpy(x),torch.from_numpy(yy),torch.from_numpy(bb),torch.from_numpy(valid),{"ja":0,"en":1}.get(row["language"],-100)

def evaluate(network,corpus,lookahead,split="validation"):
    truths=[];preds=[];languages=[];breath_clips=[]
    device=next(network.parameters()).device
    network.eval()
    with torch.inference_mode():
        for row in corpus.rows:
            if row["split"]!=split:continue
            base,y,br=corpus.arrays(row)
            x=(sequence_features(base,corpus.metadata.get("feature_mode","base"))-corpus.mean)/corpus.scale
            logits,breath=infer_sequence(network,x,device)
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
    parser.add_argument("--training-frames",type=int,default=128)
    parser.add_argument("--random-crop",action="store_true")
    parser.add_argument("--channels",type=int,default=64)
    parser.add_argument("--dilations",default="1,2,4,8")
    parser.add_argument("--lookahead",type=int,default=3)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    parser.add_argument("--gain-augment-db",type=float,default=0)
    parser.add_argument("--gain-augment-events-only",action="store_true")
    parser.add_argument("--channel-augment-db",type=float,default=0)
    parser.add_argument("--cmn-context-augment",action="store_true")
    parser.add_argument("--skip-lags",default="")
    parser.add_argument("--skip-hidden",type=int,default=64)
    parser.add_argument("--architecture",choices=["tcn","gru","attention"],default="tcn")
    parser.add_argument("--language-head",action="store_true")
    parser.add_argument("--feature-mode",choices=["base","cmn-v1","cmn-v2"],default="base")
    parser.add_argument("--spectral-frontend",action="store_true")
    parser.add_argument("--gru-layers",type=int,default=2)
    parser.add_argument("--attention-layers",type=int,default=3)
    parser.add_argument("--attention-heads",type=int,default=4)
    parser.add_argument("--attention-context",type=int,default=32)
    parser.add_argument("--english-weight",type=float,default=1.)
    parser.add_argument("--breath-selection-weight",type=float,default=.05)
    parser.add_argument("--breath-viseme-loss-weight",type=float,default=.25,help="Keep nonlexical breathing from dominating phonetic /h/ supervision")
    parser.add_argument("--patience",type=int,default=18)
    parser.add_argument("--resume")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--selection-language",choices=["all","ja","en"],default="all")
    parser.add_argument("--selection-metric",choices=["balanced","accuracy"],default="balanced")
    parser.add_argument("--learning-rate",type=float,default=.001)
    parser.add_argument("--class-weight-exponent",type=float,default=.2)
    args=parser.parse_args()
    if args.language_head and args.architecture!="gru":parser.error("--language-head currently requires --architecture gru")
    if args.feature_mode!="base" and args.architecture not in ("gru","attention"):parser.error("CMN features require a recurrent or attention model")
    if args.spectral_frontend and args.architecture not in ("gru","attention"):parser.error("Spectral frontend requires a recurrent or attention model")
    if args.architecture=="attention" and args.skip_lags:parser.error("Attention uses its own bounded temporal context; omit skip lags")
    torch.set_num_threads(args.threads);threadpool_limits(limits=args.threads)
    torch.manual_seed(args.seed);np.random.seed(args.seed)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA build/device unavailable")
    folder=ROOT/"cache/tcn_experiments"/args.run;folder.mkdir(parents=True,exist_ok=True)
    corpus=pack_corpus(ROOT/args.index,ROOT/args.corpus,ROOT/args.breath_manifest if args.breath_manifest else None,args.feature_mode)
    audit({"records":corpus.rows})
    skip_lags=tuple(int(d) for d in args.skip_lags.split(",") if d)
    if args.architecture=="attention":network=AttentionNetwork(args.channels,args.attention_layers,args.attention_heads,args.attention_context,features=len(corpus.mean),spectral_frontend=args.spectral_frontend)
    elif args.architecture=="gru":network=RecurrentNetwork(args.channels,args.gru_layers,features=len(corpus.mean),skip_lags=skip_lags,skip_hidden=args.skip_hidden,language_head=args.language_head,spectral_frontend=args.spectral_frontend)
    else:network=TemporalNetwork(args.channels,tuple(int(d) for d in args.dilations.split(",")),features=len(corpus.mean),skip_lags=skip_lags,skip_hidden=args.skip_hidden)
    if args.resume:
        checkpoint=torch.load(ROOT/args.resume,map_location="cpu",weights_only=True)
        missing=network.load_state_dict(checkpoint["model"],strict=False)
        if missing.unexpected_keys or any(not key.startswith(("skip_","language.","delta_","conv1.","conv2.","spectral_project.")) for key in missing.missing_keys):raise ValueError("Checkpoint architecture mismatch")
        corpus.mean=checkpoint["mean"].numpy();corpus.scale=checkpoint["scale"].numpy()
        if args.architecture=="gru" and len(corpus.mean)!=network.project.in_features:raise ValueError("Checkpoint acoustic feature dimensions do not match")
    network.to(device)
    (folder/"run_config.json").write_text(json.dumps({**vars(args),"actual_device":str(device),"torch_version":str(torch.__version__),"corpus_signature":corpus.metadata["signature"],"resume_kind":"weights and feature normalization; optimizer starts fresh"},indent=2),encoding="utf-8")
    dataset=SequenceDataset(corpus,args.lookahead,network.receptive_frames-1,frames=args.training_frames,gain_db=args.gain_augment_db,events_only=args.gain_augment_events_only,channel_db=args.channel_augment_db,context_augment=args.cmn_context_augment,random_crop=args.random_crop)
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
    optimizer=torch.optim.AdamW(network.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=3)
    counts=np.asarray(corpus.metadata["counts"])
    class_weight=np.minimum((counts.sum()/(16*np.maximum(counts,1)))**args.class_weight_exponent,3)
    class_weight=torch.from_numpy(class_weight.astype(np.float32)).to(device)
    best=-1.;best_epoch=0;history=[];started=time.monotonic()
    print(f"{args.architecture.upper()} device={device}, parameters={sum(p.numel() for p in network.parameters())}, training_context={network.receptive_frames}, segments={len(dataset)}",flush=True)
    for epoch in range(1,args.epochs+1):
        network.train();losses=[]
        for step,(x,y,b,valid,language) in enumerate(loader,1):
            x,y,b,valid=(v.to(device) for v in (x,y,b,valid))
            optimizer.zero_grad(set_to_none=True)
            logits,breath=network(x,valid)
            frame_loss=F.cross_entropy(logits.transpose(1,2),y,weight=class_weight,ignore_index=-100,label_smoothing=.03,reduction="none")
            importance=torch.where(b>0,args.breath_viseme_loss_weight,1.)*(y!=-100)
            normalizer=(importance*class_weight[y.clamp(min=0)]).sum().clamp(min=1)
            loss=(frame_loss*importance).sum()/normalizer
            if args.language_head:
                language=language.to(device)[:,None].expand_as(y).clone();language[y<=0]=-100
                if (language>=0).any():loss+=.1*F.cross_entropy(network.last_language_logits.transpose(1,2),language,ignore_index=-100)
            mask=b>=0
            if args.breath_manifest and mask.any():loss=loss+.3*F.binary_cross_entropy_with_logits(breath[mask],b[mask],pos_weight=torch.tensor(3.,device=device))
            loss.backward();torch.nn.utils.clip_grad_norm_(network.parameters(),5);optimizer.step()
            losses.append(float(loss.detach()))
            if step%200==0:print(f"epoch={epoch} step={step}/{args.steps_per_epoch} loss={np.mean(losses):.4f} elapsed={time.monotonic()-started:.0f}s",flush=True)
        validation=evaluate(network,corpus,args.lookahead)
        s=validation[args.selection_language]
        metric=s["accuracy"] if args.selection_metric=="accuracy" else .5*s["speech_accuracy"]+.5*s["macro_f1"]
        if "natural_breath" in validation:metric=(1-args.breath_selection_weight)*metric+args.breath_selection_weight*validation["natural_breath"]["f1"]
        scheduler.step(metric)
        record={"epoch":epoch,"loss":float(np.mean(losses)),"validation":validation,"selection_metric":metric,"elapsed_seconds":time.monotonic()-started}
        history.append(record)
        state={"model":network.state_dict(),"mean":torch.from_numpy(corpus.mean),"scale":torch.from_numpy(corpus.scale),"epoch":epoch}
        save_checkpoint(state,folder/"last.pt")
        if metric>best:
            best,best_epoch=metric,epoch;save_checkpoint(state,folder/"best.pt")
            from kotoba_lipsync.recurrent import export_recurrent
            from kotoba_lipsync.attention import export_attention
            exporter=export_attention if args.architecture=="attention" else export_recurrent if args.architecture=="gru" else export_temporal
            exporter(folder/"best_int8.npz",{k:v.detach().cpu().numpy() for k,v in network.state_dict().items()},corpus.mean,corpus.scale,model_feature_config(args.lookahead),{"channels":args.channels,"gru_layers":args.gru_layers,"attention_layers":args.attention_layers,"attention_heads":args.attention_heads,"attention_context":args.attention_context,"architecture":args.architecture,"language_head":args.language_head,"feature_mode":args.feature_mode,"spectral_frontend":args.spectral_frontend,"selection_language":args.selection_language,"selection_metric":args.selection_metric,"english_sampling_weight":args.english_weight,"dilations":list(network.dilations),"skip_lags":list(skip_lags),"skip_hidden":args.skip_hidden,"gain_augmentation_db":args.gain_augment_db,"channel_augmentation_db":args.channel_augment_db,"cmn_context_augment":args.cmn_context_augment,"has_breath_head":bool(args.breath_manifest),"best_epoch":epoch,"seed":args.seed,"pipeline_version":"ctc-midpoint-v2","receptive_frames":network.receptive_frames if args.architecture=="tcn" else None,"training_warmup_frames":network.receptive_frames-1,"labels":["sil","PP","FF","TH","DD","kk","CH","SS","nn","RR","aa","E","ih","oh","ou","HH"],"recommended_smoothing_ms":10,"training_device":str(device),"torch_version":str(torch.__version__)})
        temporary=folder/"history.tmp.json";temporary.write_text(json.dumps(history,indent=2),encoding="utf-8");temporary.replace(folder/"history.json")
        print(f"EPOCH {epoch}: validation={s['accuracy']:.4f} speech={s['speech_accuracy']:.4f} macroF1={s['macro_f1']:.4f} breath={validation.get('natural_breath',{}).get('f1')} best={best_epoch}",flush=True)
        if epoch-best_epoch>=args.patience:break
        if (folder/"STOP").exists():
            print("Cooperative stop requested; checkpoints retained.",flush=True)
            break
    print(f"Completed. Best epoch={best_epoch}. Candidate saved in {folder}. No test set used and production model unchanged.",flush=True)

if __name__=="__main__":main()
