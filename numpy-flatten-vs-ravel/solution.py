import numpy as np


def flatten_safe(arr: np.ndarray) -> np.ndarray:
    
    result=arr.flatten()
    return result


def flatten_efficient(arr: np.ndarray) -> np.ndarray:
   
    result=arr.ravel()
    return result


def compare_flatten_ravel(arr: np.ndarray) -> dict:
    
    f = arr.flatten()
    r = arr.ravel()
    return{
       "flatten_result": f,
        "ravel_result": r,
        "flatten_shares_memory": np.shares_memory(f,arr),
        "ravel_shares_memory": np.shares_memory(r,arr)
    }
