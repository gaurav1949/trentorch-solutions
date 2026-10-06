import numpy as np


def bce_loss(p: np.ndarray, y: np.ndarray) -> float:
    
   
    psafe=np.clip(p,1e-12,1-1e-12)
    loss=-np.mean((y*np.log(psafe))+(1-y)*np.log(1-psafe))
    return float(loss)
