"""Full-batch multinomial ridge fitting without a float64 copy of feature caches."""
from dataclasses import dataclass
import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class LinearFit:
    coef_:np.ndarray
    intercept_:np.ndarray
    classes_:np.ndarray

    def predict(self,x):
        return self.classes_[(x@self.coef_.T+self.intercept_).argmax(axis=1)]


def fit_linear(x,y,c,*,sample_weight=None,device="cpu",max_iter=150):
    """Minimize weighted CE + ||W||²/(2*C); intercept is unpenalized."""
    classes,inverse=np.unique(y,return_inverse=True)
    weights=np.ones(len(y),dtype=np.float32) if sample_weight is None else np.asarray(sample_weight,dtype=np.float32)
    if c<=0 or (weights<0).any() or weights.sum()<=0:raise ValueError("Invalid regularization or sample weights")
    tx=torch.as_tensor(x,dtype=torch.float32,device=device)
    ty=torch.as_tensor(inverse,dtype=torch.int64,device=device)
    tw=torch.as_tensor(weights,device=device);total=tw.sum()
    w=torch.zeros((len(classes),x.shape[1]),dtype=torch.float32,device=device,requires_grad=True)
    counts=np.bincount(inverse,weights=weights,minlength=len(classes))
    b=torch.tensor(np.log(np.maximum(counts/counts.sum(),1e-8)),dtype=torch.float32,device=device,requires_grad=True)
    optimizer=torch.optim.LBFGS([w,b],lr=1.,max_iter=max_iter,history_size=20,tolerance_grad=1e-6,tolerance_change=1e-9,line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss=(F.cross_entropy(F.linear(tx,w,b),ty,reduction="none")*tw).sum()/total+w.square().sum()/(2*c*total)
        loss.backward();return loss
    optimizer.step(closure)
    return LinearFit(w.detach().cpu().numpy().copy(),b.detach().cpu().numpy().copy(),classes)
