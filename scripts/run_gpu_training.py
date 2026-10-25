"""Use the optional project-local CUDA torch without changing the runtime venv."""
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
cuda=root/"cache/torch_cuda"
if cuda.exists():sys.path.insert(0,str(cuda))
from scripts.train_temporal import main

if __name__=="__main__":main()
