import numpy as np


def select_above_threshold(arr: np.ndarray, threshold: float) -> np.ndarray:
    
    mask=arr>threshold
    return arr[mask]


def select_in_range(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    
    return arr[(arr>low) & (arr<high)]


def zero_out_negatives(arr: np.ndarray) -> None:
   
    mask=arr<0
    arr[mask]=0
