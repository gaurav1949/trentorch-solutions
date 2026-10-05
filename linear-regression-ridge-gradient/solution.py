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
    
def ridge_grad(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    target: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Compute the gradient of the MSE loss with L2 regularization.
    """
    # TODO: Start from mse_gradient()'s result, then add the L2 penalty
    # term to grad_weight only. Leave grad_bias unchanged -- bias is
    # never regularized.
    grad_weight_mse,grad_bias_mse=mse_gradient(input,weight,bias,target)
    grad_weight=(grad_weight_mse)+(2*lam*weight)
    grad_bias=grad_bias_mse
    return grad_weight,grad_bias
