import numpy as np
def linear(input,weight,bias):
    z=input@weight.T
    if bias is not None:
        z+=bias
    return z
def sigmoid(z):
    zsafe=np.clip(z,-500,500)
    result=(1)/(1+np.exp(-zsafe))
    return result
def bce_gradient(input,p,target_2d):
    error=p-target_2d
    n_samples=input.shape[0]
    grad_weight=(error.T @ input)/n_samples
    grad_bias=np.sum(error,axis=0)/n_samples
    return grad_weight,grad_bias
def gd_step(weight,bias,grad_weight,grad_bias,lr):
    updated_weight=weight-grad_weight*lr
    updated_bias=bias-grad_bias*lr
    return updated_weight,updated_bias
def train_logistic_regression(
    input: np.ndarray,
    target: np.ndarray,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    input:  shape (batch_size, in_features)
    target: shape (batch_size,), 0 or 1 per sample

    Returns:
        weight: shape (1, in_features)
        bias: shape (1,)
    """
    # TODO: Initialize weight to zeros (1, in_features), bias to zeros (1,).
    # Reshape target to (batch_size, 1) once, up front.
    # Repeat for `epochs` iterations:
    #   1. p = sigmoid(linear(input, weight, bias))
    #   2. Compute gradients with bce_gradient()
    #   3. Update weight, bias with gd_step()
    # Reuse those functions -- don't reimplement their logic here.
    batch_size,in_features=input.shape
    weight=np.zeros((1, in_features))
    bias=np.zeros((1,))
    
    target_2d=target.reshape((batch_size, 1))
    for _ in range(epochs):
         p = sigmoid(linear(input, weight, bias))
         grad_weight,grad_bias=bce_gradient(input,p,target_2d)
         weight,bias=gd_step(weight,bias,grad_weight,grad_bias,lr)
    return weight,bias
