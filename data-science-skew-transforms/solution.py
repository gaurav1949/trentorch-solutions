import numpy as np


def skewness(x: np.ndarray) -> float:
    """
    x: 1D float array

    Returns:
        mean(((x - mean) / std) ** 3) with std the population standard
        deviation, as a float. Returns 0.0 when std is zero.
    """
    x=np.asarray(x,dtype=float)
    # TODO: Implement the formula from Theory.
    std=np.std(x,ddof=0)
    if std==0:
        return 0.0
    mean=np.mean(x)
    return float(np.mean(((x - mean) / std) ** 3))


def log1p_transform(x: np.ndarray) -> np.ndarray:
    """
    Returns log(1 + x) for a 1D array with no negative values.

    Raises:
        ValueError: if any value is negative.
    """
    # TODO: Use np.log1p.
    x=np.asarray(x,dtype=float)
    if np.any(x<0):
        raise ValueError("x is negative")
    return np.log1p(x)
    
        
        
def inverse_log1p(z: np.ndarray) -> np.ndarray:
    """Returns exp(z) - 1, the inverse of log1p_transform."""
    # TODO: Use np.expm1.
    z=np.asarray(z,dtype=float)
    return np.expm1(z)


def box_cox(x: np.ndarray, lam: float) -> np.ndarray:
    """
    x: 1D float array of strictly positive values
    lam: the Box-Cox parameter

    Returns:
        (x ** lam - 1) / lam when lam != 0, and log(x) when lam == 0.

    Raises:
        ValueError: if any value is not strictly positive.
    """
    # TODO: Implement both branches from Theory.
    x=np.asarray(x,dtype=float)
    if np.any(x<=0):
        raise ValueError("x is Negative")
    if lam==0:
        return np.log(x)
    return (x**lam - 1)/lam
    


def best_box_cox_lambda(x: np.ndarray, candidates: list) -> float:
    """
    Returns the value in `candidates` whose Box-Cox transform of x has
    the smallest absolute skewness. Ties go to the earliest candidate.
    """
    # TODO: Compare the skewness of each transform.
    x=np.asarray(x,dtype=float)
    result=float("inf")
    best_lambda=None
    for lm in candidates:
        transformed=box_cox(x,lm)
        s=abs(skewness(transformed))
        if result>s:
            result=s
            best_lambda=lm
    return best_lambda
