"""Fixed-depth numeric residual trees; NumPy only at inference time."""
import numpy as np

PREFIX="decoder_tree_"


class TreeResidual:
    def __init__(self,arrays,config):
        if config["format"]!="scalar16-f32-v1":raise ValueError("Unsupported residual trees")
        self.depth=int(config["max_depth"]);self.parameters=int(config["nodes"])
        if not 0<=self.depth<=8:raise ValueError("Residual tree depth exceeds the runtime bound")
        self.feature=arrays[PREFIX+"feature"].copy();self.condition=arrays[PREFIX+"condition"].copy()
        self.left=arrays[PREFIX+"left"].copy();self.right=arrays[PREFIX+"right"].copy();self.default_left=arrays[PREFIX+"default_left"].copy()
        if self.feature.ndim!=2 or self.feature.shape[0]%16 or any(a.shape!=self.feature.shape for a in (self.condition,self.left,self.right,self.default_left)):raise ValueError("Invalid tree layout")
        self.indices=np.arange(len(self.feature))[None,:]
        self.float_weights_bytes=self.condition.nbytes

    def predict(self,features):
        source=np.asarray(features,dtype=np.float32);single=source.ndim==1
        x=source[None] if single else source
        output=np.empty((len(x),16),dtype=np.float32)
        for start in range(0,len(x),64):
            batch=x[start:start+64];position=np.zeros((len(batch),len(self.feature)),dtype=np.int32)
            for _ in range(self.depth):
                left=self.left[self.indices,position];right=self.right[self.indices,position]
                values=np.take_along_axis(batch,self.feature[self.indices,position],axis=1)
                choose_left=np.where(np.isnan(values),self.default_left[self.indices,position],values<self.condition[self.indices,position])
                position=np.where(left<0,position,np.where(choose_left,left,right))
            values=self.condition[self.indices,position]
            output[start:start+len(batch)]=values.reshape(len(batch),-1,16).sum(axis=1)
        return output[0] if single else output


def pack_numeric_trees(document):
    """Read the numeric scalar-tree JSON representation, without native code."""
    learner=document["learner"]
    if int(learner["learner_model_param"]["num_class"])!=16:raise ValueError("Expected 16 classes")
    booster=learner["gradient_booster"]
    if booster["name"]!="gbtree":raise ValueError("Only additive numeric trees are supported")
    model=booster["model"];trees=model["trees"];count=len(trees)
    if count%16 or model["tree_info"]!=list(range(16))*(count//16):raise ValueError("Unexpected multiclass tree order")
    width=max(len(t["left_children"]) for t in trees)
    feature=np.zeros((count,width),np.int16);condition=np.zeros((count,width),np.float32)
    left=np.full((count,width),-1,np.int16);right=left.copy();defaults=np.zeros((count,width),bool)
    maximum_depth=0;nodes=0
    for i,tree in enumerate(trees):
        n=len(tree["left_children"]);nodes+=n
        if n>32767 or any(tree["split_type"]):raise ValueError("Unsupported tree size or categorical split")
        if max(tree["split_indices"],default=0)>32767:raise ValueError("Too many decoder features")
        feature[i,:n]=tree["split_indices"];condition[i,:n]=tree["split_conditions"]
        left[i,:n]=tree["left_children"];right[i,:n]=tree["right_children"];defaults[i,:n]=tree["default_left"]
        pending=[(0,0)];seen=set()
        while pending:
            node,depth=pending.pop()
            if node in seen or not 0<=node<n:raise ValueError("Invalid tree graph")
            seen.add(node);maximum_depth=max(maximum_depth,depth)
            if left[i,node]>=0:pending.extend(((int(left[i,node]),depth+1),(int(right[i,node]),depth+1)))
        if len(seen)!=n:raise ValueError("Unreachable tree nodes")
    if maximum_depth>8 or not np.isfinite(condition).all():raise ValueError("Invalid residual depth or values")
    return {PREFIX+"feature":feature,PREFIX+"condition":condition,PREFIX+"left":left,PREFIX+"right":right,PREFIX+"default_left":defaults},{"format":"scalar16-f32-v1","max_depth":maximum_depth,"nodes":nodes,"trees":count}
