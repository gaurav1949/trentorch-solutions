import numpy as np


def describe_ndarray_basics(arr: np.ndarray) -> dict:
    
    a={"dtype":str(arr.dtype),"itemsize":arr.itemsize}
    return a
