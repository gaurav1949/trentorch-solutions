import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    
    zsafe=np.clip(z,-500,500)
    sigmoid=(1)/(1+np.exp(-zsafe))
    return sigmoid
