import numpy as np


def bce_gradient(
    input: np.ndarray,
    p: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
   
    error=p-target
    n_samples=input.shape[0]
    grad_weight=(error.T @ input)/n_samples
    grad_bias=np.sum(error,axis=0)/n_samples
    return grad_weight,grad_bias
