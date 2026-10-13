# Functions
import numpy as np


# Logistic Regression implementation
def sigmoid(t):
    """apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """

    return 1 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.
    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
    Returns:
        a non-negative loss
    """

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    z = tx @ w

    # log(1 + exp(z)) - y * z, written so that it never evaluates log(0)
    return np.mean(np.logaddexp(0, z) - y * z)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.
    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
    Returns:
        a vector of shape (D, )
    """

    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    sigma = sigmoid(tx @ w)

    return tx.T @ (sigma - y) / y.shape[0]

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the updated w.
    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
        gamma: float
    Returns:
        w: shape=(D, )
    """

    gradient = calculate_gradient(y, tx, w)
    w = w - gamma * gradient

    return w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.
    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, )
        max_iters: int
        gamma: float
    Returns:
        w: last weight vector, shape=(D, )
        loss: loss at the last w, scalar
    """

    w = initial_w
    for i in range(max_iters):
        w = learning_by_gradient_descent(y, tx, w, gamma)

    loss = calculate_loss(y, tx, w)

    return w, loss

# Regularized Logistic Regression implementation

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the gradient of the penalized loss.
    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
        lambda_: scalar
    Returns:
        gradient: shape=(D, )
    """

    gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w

    return gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression. Return the updated w.
    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, )
        gamma: scalar
        lambda_: scalar
    Returns:
        w: shape=(D, )
    """
    
    gradient = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * gradient

    return w

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.
    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        lambda_: scalar
        initial_w: shape=(D, )
        max_iters: int
        gamma: float
    Returns:
        w: last weight vector, shape=(D, )
        loss: loss at the last w, scalar
    """

    w = initial_w
    for i in range(max_iters):
        w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)

    loss = calculate_loss(y, tx, w)

    return w, loss
