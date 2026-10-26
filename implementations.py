# Functions
import numpy as np

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)

    e = y - tx @ w
    mse = 1 / (2 * len(y)) * np.dot(e, e)
    
    return w, mse

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e)) # MEAN SQUARE ERROR
    
    return w, loss

#######

# Helper functions for Logistic Regression
def sigmoid(t):
    """apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """

    return 1 / (1 + np.exp(-t))

def calculate_loss_log(y, tx, w):
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

def calculate_gradient_log(y, tx, w):
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

# Logistic Regression implementation

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
        gradient = calculate_gradient_log(y, tx, w)  # Gradient
        w = w - gamma * gradient  # Gradient descent step

    loss = calculate_loss_log(y, tx, w)  # Compute loss

    return w, loss

# Regularized Logistic Regression implementation

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
        gradient = calculate_gradient_log(y, tx, w) + 2 * lambda_ * w  # Penalized gradient
        w = w - gamma * gradient  # Gradient descent step

    loss = calculate_loss_log(y, tx, w)  # Compute loss

    return w, loss

######