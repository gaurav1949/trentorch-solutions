import numpy as np


def huber_loss(
    input: np.ndarray, target: np.ndarray, delta: float = 1.0, reduction: str = "mean"
) -> float | np.ndarray:
   
    error=input-target
    abs_error=np.abs(error)
    
    answer=np.where(abs_error<=delta,0.5*np.square(error),delta*(abs_error-0.5*delta))
    if reduction=="mean":
        return float(np.mean(answer))
    elif reduction=="sum":
        return float(np.sum(answer))
    elif reduction=="none":
        return answer
    else:
        raise ValueError("Invalid option")
