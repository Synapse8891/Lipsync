"""Reproducible standard/five-hour recipes; never opens a microphone."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def plan_commands(profile_name,profile,*,skip_download=False,skip_prepare=False,started=None,hours=None):
    started=started if started is not None else time.time()
    extended=profile_name=="five-hour"
    deadline=started+(hours if hours is not None else profile.get("hours",5))*3600 if extended else None
    commands=[]
    if not skip_download:
        command=["scripts/download_training.py","--ja-shards",str(profile["ja_shards"])]
        if deadline is not None:command += ["--deadline",str(deadline-180)]
        commands.append(command)
    if not skip_prepare:
        command=["scripts/prepare_training.py"]
        for key in ("ja_train","en_train","ja_eval","en_eval"):
            command += ["--"+key.replace("_","-"),str(profile[key])]
        if extended:
            command += ["--balanced-order","--allow-fewer","--index-name",profile["index_name"],"--deadline",str(started+(deadline-started)*profile["preparation_fraction"])]
        commands.append(command)
    if extended:
        commands.append(["scripts/train_search.py","--index","cache/features_v2/"+profile["index_name"],"--deadline",str(deadline-120)])
    else:
        commands.append(["scripts/train.py","--epochs","100","--hidden","192","96"])
    return commands,deadline

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--profile",choices=["standard","five-hour"],default="standard")
    parser.add_argument("--hours",type=float,help="Overall computation budget for five-hour profile")
    parser.add_argument("--skip-download",action="store_true")
    parser.add_argument("--skip-prepare",action="store_true")
    parser.add_argument("--dry-run",action="store_true",help="Print plan without downloading, preparing or training")
    args=parser.parse_args()
    if args.hours is not None and (not math.isfinite(args.hours) or args.hours<=0 or args.profile!="five-hour"):
        parser.error("--hours must be positive and requires --profile five-hour")
    profiles=json.loads((ROOT/"configs/training_profiles.json").read_text(encoding="utf-8"))
    profile=profiles[args.profile]
    started=time.time()
    commands,deadline=plan_commands(args.profile,profile,skip_download=args.skip_download,skip_prepare=args.skip_prepare,started=started,hours=args.hours)
    if args.dry_run:
        print(json.dumps({"profile":args.profile,"budget_hours":(deadline-started)/3600 if deadline else None,"targets":profile,"commands":commands,"note":"時間枠が余っても待機はしません。処理中のI/O・録音・最終評価の分だけ期限を超える場合があります。"},ensure_ascii=False,indent=2))
        return
    status={"profile":args.profile,"started_unix":started,"deadline_unix":deadline,"state":"running"}
    path=ROOT/"reports/run_status.json";path.parent.mkdir(exist_ok=True)
    def save():
        status["elapsed_seconds"]=time.time()-started
        path.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    try:
        for command in commands:
            if deadline is not None and time.time()>=deadline:
                raise TimeoutError("全体の計算枠が終了しました。完了済みキャッシュは保持しています。")
            status["stage"]=command[0];save()
            subprocess.run([sys.executable,"-u",*command],cwd=ROOT,check=True)
        demos=sorted((ROOT/"data/demo").glob("*.wav"))
        if demos:
            status["stage"]="benchmark/report";save()
            subprocess.run([sys.executable,"-m","kotoba_lipsync","benchmark",str(demos[0])],cwd=ROOT,check=True)
            subprocess.run([sys.executable,"scripts/write_results.py"],cwd=ROOT,check=True)
        status["state"]="complete"
    except (subprocess.CalledProcessError,TimeoutError,KeyboardInterrupt) as error:
        status["state"]="interrupted" if isinstance(error,KeyboardInterrupt) else "stopped"
        status["reason"]=str(error);save()
        raise SystemExit(130 if isinstance(error,KeyboardInterrupt) else 1)
    save()

if __name__=="__main__":main()
