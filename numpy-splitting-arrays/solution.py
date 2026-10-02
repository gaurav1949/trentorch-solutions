import numpy as np


def split_into_n_parts(arr: np.ndarray, n: int, axis: int) -> list:
    

    return np.split(arr,n,axis=axis)


def split_columns(arr: np.ndarray, n: int) -> list:
    
    return np.hsplit(arr,n)


def split_result_shares_memory(arr: np.ndarray, n: int, axis: int) -> bool:
   
    result=np.split(arr,n,axis=axis)
    return np.shares_memory(result[0],arr)
