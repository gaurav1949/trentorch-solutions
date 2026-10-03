import numpy as np


def add_scalar(matrix: np.ndarray, scalar: float) -> np.ndarray:
  
    return matrix+scalar


def add_row_vector(matrix: np.ndarray, row_vector: np.ndarray) -> np.ndarray:
    
    return matrix+row_vector
    

def add_column_vector(matrix: np.ndarray, col_values: np.ndarray) -> np.ndarray:
    
    result=matrix+col_values[:,np.newaxis]
    return result


def outer_sum(row_values: np.ndarray, col_values: np.ndarray) -> np.ndarray:
    
    result=row_values[np.newaxis,:]+col_values[:,np.newaxis]
    return result
