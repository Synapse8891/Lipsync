"""One-command reproducible recipe. Never runs a microphone or sends OSC."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def run(*args):
    subprocess.run([sys.executable,*args],cwd=ROOT,check=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--skip-download",action="store_true")
    parser.add_argument("--skip-prepare",action="store_true")
    args=parser.parse_args()
    if not args.skip_download:run("scripts/download_training.py","--ja-shards","3")
    if not args.skip_prepare:run("scripts/prepare_training.py","--ja-train","1800","--en-train","200","--ja-eval","180","--en-eval","40")
    run("scripts/train.py","--epochs","100","--hidden","192","96")
    demos=sorted((ROOT/"data/demo").glob("*.wav"))
    if demos:
        run("-m","kotoba_lipsync","benchmark",str(demos[0]))
        run("scripts/write_results.py")

if __name__=="__main__":main()
