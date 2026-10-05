import numpy as np

def mse_gradient(input,weight,bias,target):
    prediction=input@weight.T
    if bias is not None:
        prediction+=bias
    N=prediction.size
    grad_prediction=(2/N)*(prediction-target)
    grad_weight=grad_prediction.T@input
    grad_bias=None if bias is None else grad_prediction.sum(axis=0)
    return grad_weight,grad_bias
def gd_step(weight,bias,grad_weight,grad_bias,lr):
    updated_weight=weight-lr*grad_weight
    updated_bias=None if bias is None else bias-lr*grad_bias
    return updated_weight,updated_bias
def train_linear_regression(
    input: np.ndarray,
    target: np.ndarray,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    input:  shape (batch_size, in_features)
    target: shape (batch_size,), one target value per sample
    lr: learning rate
    epochs: number of full-batch gradient-descent steps

    Returns:
        weight: shape (1, in_features)
        bias: shape (1,)
    """
    batch_size,in_features=input.shape
    weight=np.zeros((1, in_features))
    bias=np.zeros((1,))
    target=target.reshape((batch_size, 1))
    for _ in range(epochs):
        grad_weight,grad_bias=mse_gradient(input,weight,bias,target)
        weight,bias=gd_step(weight,bias,grad_weight,grad_bias,lr)
    return weight,bias
    # TODO: Initialize weight to zeros (1, in_features), bias to zeros (1,).
    # Reshape target to (batch_size, 1) once, up front.
    # Repeat for `epochs` iterations:
    #   1. Compute gradients with mse_gradient()
    #   2. Update weight, bias with gd_step()
    # Reuse those functions -- don't reimplement their logic here.
