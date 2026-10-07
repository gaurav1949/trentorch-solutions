import numpy as np


def standardize(x: np.ndarray, mean: np.ndarray | None = None, std: np.ndarray | None = None):
    """
    Standardization ("z-scoring"): rescale each column to have mean 0
    and std 1.

        standardized = (x - mean) / std

    If mean/std aren't supplied, compute them from x itself (per
    column, axis=0). If they ARE supplied (the "apply training set
    statistics to new data" case, see Theory), use them as-is rather
    than recomputing from x.

    Returns (standardized_x, mean, std) so a caller can reuse the same
    mean/std on new data later.
    """
    if mean is None:
        mean=np.mean(x,axis=0)
    if std is None:
        std=np.std(x,axis=0)
    standardized_x=(x-mean)/(std)
    return standardized_x,mean,std


def min_max_normalize(
    x: np.ndarray, min_val: np.ndarray | None = None, max_val: np.ndarray | None = None
):
    """
    Min-max normalization: rescale each column into [0, 1].

        normalized = (x - min) / (max - min)

    Same fit-vs-reuse pattern as standardize: compute min_val/max_val
    from x if not supplied, otherwise reuse the given ones.

    Returns (normalized_x, min_val, max_val).
    """
    if min_val is None:
       min_val=np.min(x,axis=0) 
    if max_val is None:
       max_val=np.max(x,axis=0)
    normalized_x=(x-min_val)/(max_val-min_val)
    return normalized_x,min_val,max_val
