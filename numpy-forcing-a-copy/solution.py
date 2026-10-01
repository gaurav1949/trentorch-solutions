import numpy as np


def get_independent_slice(arr: np.ndarray, start: int, stop: int) -> np.ndarray:
    
    b=arr[start:stop]
    independent=b.copy()
    return independent


def safe_modify_first_n(arr: np.ndarray, n: int, new_value) -> np.ndarray:
    
    b=arr.copy()
    b[:n]=new_value
    return b
