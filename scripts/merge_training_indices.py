"""Freeze completed source records into a new immutable experiment index."""
import argparse
import hashlib
import json
from pathlib import Path


def merge_documents(documents):
    result={**documents[0]};records={}
    for document in documents:
        for field in ("annotation_signatures","feature_config","pipeline_version"):
            if document[field]!=result[field]:raise ValueError("Index pipeline differs: "+field)
        for row in document["records"]:
            if row["id"] in records and row!=records[row["id"]]:raise ValueError("Conflicting metadata for recording "+row["id"])
            records[row["id"]]=row
        for field in ("speech_extension","waveform_augmentation"):
            if field in document:result[field]=document[field]
    result["records"]=list(records.values());return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("indices",nargs="+",type=Path);parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError("Choose a new output path; frozen indices are never overwritten")
    documents=[];sources=[]
    for path in args.indices:
        raw=path.read_bytes();documents.append(json.loads(raw));sources.append({"path":str(path),"sha256":hashlib.sha256(raw).hexdigest()})
    result=merge_documents(documents);result["merged_sources"]=sources
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(f"Frozen {len(result['records'])} records -> {args.output}")


if __name__=="__main__":main()
