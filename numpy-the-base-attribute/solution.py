import numpy as np


def owns_its_data(arr: np.ndarray) -> bool:
   
    
    return arr.base is None


def find_ultimate_owner(arr: np.ndarray) -> np.ndarray:
   
    owner=arr
    while owner.base is not None:
        owner=owner.base
    return owner
    if arr.base is None:
        return arr
