import numpy as np


def softmax(Z: np.ndarray) -> np.ndarray:
    """Z: shape (n_samples, n_classes). Rows sum to 1."""
    # TODO: Subtract each row's max before exponentiating for numerical
    # stability, then normalize each row so it sums to 1.
    Z_shift=Z-np.max(Z,axis=1,keepdims=True)
    P=(np.exp(Z_shift))/(np.sum(np.exp(Z_shift),axis=1,keepdims=True))
    return P


def cce_loss(P: np.ndarray, y_indices: np.ndarray) -> float:
    """P: shape (n_samples, n_classes) from softmax. y_indices: integer class index per sample."""
    # TODO: Pick out each sample's predicted probability for its true
    # class, clip it away from 0, and return the mean negative log as
    # a plain Python float.
    p_correct=P[np.arange(P.shape[0]),y_indices]
    p_correct=np.clip(p_correct,1e-12,1.0)
    return -np.mean(np.log(p_correct))
