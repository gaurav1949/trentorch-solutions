import numpy as np


def select_by_indices(arr: np.ndarray, indices: list) -> np.ndarray:
    
    return arr[indices]  


def select_paired_2d(arr: np.ndarray, rows: list, cols: list) -> np.ndarray:
    
    return  arr[rows,cols]


def fancy_index_is_copy(arr: np.ndarray, indices: list) -> dict:
    """
    Select elements at `indices` from `arr` using fancy indexing,
    then mutate the first element of the result to -1.

    Return a dictionary:
      {
        "selected": <the fancy-indexed result, after mutation>,
        "original_unaffected": <True if arr itself was NOT
                                  changed by the mutation above>
      }
    """
    a=arr[indices]
    a[0]=-1
    return {"selected":a,"original_unaffected":bool(arr[0]!=-1)}
