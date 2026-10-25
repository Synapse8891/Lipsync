"""Retire a completed test attempt as development data while preserving history."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument("--base",default="cache/features_v2/index_five_hour.json");p.add_argument("--retired",default="cache/features_fresh_v2/index.json");p.add_argument("--output",default="cache/features_v2/index_development_v3.json")
    args=p.parse_args();base=json.loads((ROOT/args.base).read_text());retired=json.loads((ROOT/args.retired).read_text())
    if base["annotation_signatures"]!=retired["annotation_signatures"]:raise ValueError("Annotation signatures differ")
    existing={r["id"] for r in base["records"]}
    if existing&{r["id"] for r in retired["records"]}:raise ValueError("Repeated recording in retired test")
    base["records"].extend({**r,"split":"validation","feature_root":Path(args.retired).parent.as_posix()} for r in retired["records"])
    base["development_note"]="Completed failed test attempt is now development data and must not be reported as untouched test again. Reserve other speakers/texts for the next final test."
    (ROOT/args.output).write_text(json.dumps(base,indent=2),encoding="utf-8")

if __name__=="__main__":main()
