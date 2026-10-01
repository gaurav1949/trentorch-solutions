import numpy as np


def get_dtype_name(arr: np.ndarray) -> str:
   
    return str(arr.dtype)


def create_with_dtype(values: list, dtype) -> np.ndarray:
   
    a=np.array(values,dtype=dtype)
    return a


def convert_dtype(arr: np.ndarray, new_dtype) -> np.ndarray:
    
    return arr.astype(new_dtype)
