import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from _load import load_solution  # noqa: E402


def train_val_split(
    input: np.ndarray, target: np.ndarray, val_fraction: float = 0.2, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    rng=np.random.default_rng(seed)
    indices=rng.permutation(len(input))
    val_size=round(val_fraction*len(input))
    train_ind=indices[val_size:]
    val_ind=indices[:val_size]
    input_train=input[train_ind]
    target_train=target[train_ind]
    input_val=input[val_ind]
    target_val=target[val_ind]
    return(input_train, target_train, input_val, target_val)


def generalization_gap(train_loss: float, val_loss: float) -> float:
    """
    The generalization gap: how much worse the model performs on data
    it never trained on, compared to the data it did. A large positive
    gap is the numerical signature of overfitting.
    """
    return val_loss-train_loss
