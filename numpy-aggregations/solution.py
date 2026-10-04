import numpy as np


def compute_summary(arr: np.ndarray) -> dict:
    
    return{
        "sum":np.sum(arr),
        "mean":np.mean(arr),
        "std":np.std(arr),
        "min":np.min(arr),
        "max":np.max(arr),
        "argmin":np.argmin(arr),
        "argmax":np.argmax(arr)
    }


def sum_with_loop(arr: np.ndarray) -> float:
    
    sum=0;
    for i in range(len(arr)):
        sum+=arr[i]
    return sum
