import numpy as np


def list_to_array(values: list) -> np.ndarray:
    """
    Convert a flat Python list `values` into a 1-dimensional
    ndarray using np.array(), and return it.
    """
    return np.array(values)

def nested_list_to_array(rows: list) -> np.ndarray:
    """
    Convert a list of equal-length lists `rows` into a
    2-dimensional ndarray using np.array(), and return it.
    Example: [[1, 2], [3, 4]] -> a 2x2 array.
    """
    return np.array(rows)
