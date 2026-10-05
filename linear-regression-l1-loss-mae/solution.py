import numpy as np


def l1_loss(input: np.ndarray, target: np.ndarray, reduction: str = "mean") -> float | np.ndarray:
    """
    input, target: same shape, any shape.
    reduction: "mean" (default), "sum", or "none".

    Mirrors torch.nn.functional.l1_loss: mean absolute error, not
    squared error. See Theory for why this changes MSE Loss's
    behavior around outliers.
    """
    l1_loss=np.abs(input-target)
    if reduction=="mean":
        return np.mean(l1_loss)
    elif reduction=="sum":
        return np.sum(l1_loss)
    elif reduction=="none":
        return l1_loss
    else:
        raise ValueError("Invalid input")
