"""Promote a frozen, evaluated candidate while retaining the previous release."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("candidate",type=Path)
    parser.add_argument("report",type=Path)
    parser.add_argument("--minimum-japanese",type=float,default=.85)
    parser.add_argument("--minimum-breath-f1",type=float,default=.70)
    parser.add_argument("--dry-run",action="store_true")
    args=parser.parse_args()
    candidate=args.candidate.resolve();evaluation=args.report.resolve()
    candidate_relative=str(candidate.relative_to(ROOT));evaluation_relative=str(evaluation.relative_to(ROOT))
    report=json.loads(evaluation.read_text(encoding="utf8"))
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest!=report["sha256"]:raise ValueError("Candidate differs from the frozen evaluation")
    if report["test"]["ja"]["accuracy"]<args.minimum_japanese:raise ValueError("Japanese agreement is below the user criterion")
    if report["natural_breath"]["runtime_probability"]["true_positive_clips"]<1:raise ValueError("No independently evaluated natural breath detection")
    if report["natural_breath"]["runtime_probability"]["f1"]<args.minimum_breath_f1:raise ValueError("Breath F1 is below the retained-ability criterion")
    if args.dry_run:print(json.dumps({"validated":str(candidate),"sha256":digest,"minimum_japanese":args.minimum_japanese}));return
    backup=ROOT/"reports/baseline";backup.mkdir(exist_ok=True)
    for name in ("training_report.json","benchmark.json","runtime_validation.json","RESULTS.md"):
        source=ROOT/"reports"/name;target=backup/name
        if source.exists() and not target.exists():shutil.copy2(source,target)
    destination=ROOT/"models/kotoba_int8.npz"
    temporary=destination.with_suffix(".tmp.npz");shutil.copy2(candidate,temporary);os.replace(temporary,destination)
    report.update({"model":"models/kotoba_int8.npz","candidate":candidate_relative,"parameter_count":report["parameters"],"frozen_evaluation":evaluation_relative,"phonetic_class_15":"HH (/h/), called breath by the legacy score label list; independent natural breathing is reported separately","completion_criterion":{"scope":"Japanese agreement with automatic CTC labels, including silence","minimum":args.minimum_japanese,"observed":report["test"]["ja"]["accuracy"],"english":"auxiliary; no 75 percent requirement","natural_breath":"independent output evaluated with weak clip labels; false positives disclosed"}})
    (ROOT/"reports/training_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({"promoted":str(destination),"sha256":digest,"japanese_agreement":report["test"]["ja"]["accuracy"]},indent=2))


if __name__=="__main__":main()
