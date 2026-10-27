"""The six methods required by project 1.

Every method returns (w, loss): the last weight vector and the loss at that
vector. For the regularized methods the loss does not include the penalty term.
The mean squared error has a factor 1/2, as in the lecture notes.
"""

import numpy as np

# ---------- Linear regression ----------


def compute_loss(y, tx, w):
    """Mean squared error, with a factor 1/2.

    Args:
        y: labels, shape=(N, )
        tx: features, shape=(N, D)
        w: weights, shape=(D, )

    Returns:
        loss: scalar, 1/(2N) * ||y - tx w||^2
    """
    e = y - tx @ w
    return np.mean(e**2) / 2


def compute_gradient(y, tx, w):
    """Gradient of the mean squared error at w.

    Args:
        y: labels, shape=(N, )
        tx: features, shape=(N, D)
        w: weights, shape=(D, )

    Returns:
        gradient: shape=(D, )
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: labels, shape=(N, )
        tx: features, shape=(N, D)
        initial_w: initial weights, shape=(D, )
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: last weights, shape=(D, )
        loss: mean squared error at w, scalar
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * compute_gradient(y, tx, w)

    loss = compute_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (mini-batch size 1).

    Each step uses the gradient of a single data point drawn at random.

    Args:
        y: labels, shape=(N, )
        tx: features, shape=(N, D)
        initial_w: initial weights, shape=(D, )
        max_iters: number of stochastic gradient steps
        gamma: step size

    Returns:
        w: last weights, shape=(D, )
        loss: mean squared error at w on all the data, scalar
    """
    w = initial_w
    for _ in range(max_iters):
        n = np.random.randint(len(y), size=1)  # index of one random data point
        w = w - gamma * compute_gradient(y[n], tx[n], w)

    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Args:
        y: labels, shape=(N, )
        tx: features, shape=(N, D)

    Returns:
        w: optimal weights, shape=(D, )
        loss: mean squared error at w, scalar

    >>> least_squares(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1.0, 0.1]]))[0]
    array([ 0.21212121, -0.12121212])
    """
    # The pseudo-inverse gives the solution of tx.T @ tx w = tx.T @ y, and also
    # works when tx.T @ tx is singular (duplicate or constant columns)
    w = np.linalg.pinv(tx) @ y

    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Minimizes the mean squared error + lambda_ * ||w||^2.

    Args:
        y: labels, shape=(N, )
        tx: features, shape=(N, D)
        lambda_: regularization strength

    Returns:
        w: optimal weights, shape=(D, )
        loss: mean squared error at w (without the penalty term), scalar

    >>> ridge_regression(np.array([0.1, 0.2]), np.array([[2.3, 3.2], [1.0, 0.1]]), 1)[0]
    array([0.03947092, 0.00319628])
    """
    n, d = tx.shape
    # Setting the gradient to 0 gives (tx.T tx + 2 N lambda_ I) w = tx.T y
    a = tx.T @ tx + 2 * n * lambda_ * np.eye(d)
    w = np.linalg.solve(a, tx.T @ y)

    loss = compute_loss(y, tx, w)
    return w, loss


# ---------- Logistic regression (labels in {0, 1}) ----------


def sigmoid(t):
    """Sigmoid function 1 / (1 + exp(-t)).

    Args:
        t: scalar or array

    Returns:
        sigmoid of t, same shape as t
    """
    # Same value as 1 / (1 + exp(-t)), but does not overflow for large negative t
    return np.exp(-np.logaddexp(0, -t))


def calculate_loss_log(y, tx, w):
    """Logistic loss (mean negative log-likelihood).

    Args:
        y: labels in {0, 1}, shape=(N, )
        tx: features, shape=(N, D)
        w: weights, shape=(D, )

    Returns:
        loss: non-negative scalar
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    z = tx @ w
    # log(1 + exp(z)) - y * z, written so that it never evaluates log(0)
    return np.mean(np.logaddexp(0, z) - y * z)


def calculate_gradient_log(y, tx, w):
    """Gradient of the logistic loss at w.

    Args:
        y: labels in {0, 1}, shape=(N, )
        tx: features, shape=(N, D)
        w: weights, shape=(D, )

    Returns:
        gradient: shape=(D, )
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent.

    Args:
        y: labels in {0, 1}, shape=(N, )
        tx: features, shape=(N, D)
        initial_w: initial weights, shape=(D, )
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: last weights, shape=(D, )
        loss: logistic loss at w, scalar
    """
    w = initial_w
    for _ in range(max_iters):
        w = w - gamma * calculate_gradient_log(y, tx, w)

    loss = calculate_loss_log(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent.

    Minimizes the logistic loss + lambda_ * ||w||^2.

    Args:
        y: labels in {0, 1}, shape=(N, )
        tx: features, shape=(N, D)
        lambda_: regularization strength
        initial_w: initial weights, shape=(D, )
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: last weights, shape=(D, )
        loss: logistic loss at w (without the penalty term), scalar
    """
    w = initial_w
    for _ in range(max_iters):
        # Gradient of the loss plus gradient of the penalty lambda_ * ||w||^2
        gradient = calculate_gradient_log(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = calculate_loss_log(y, tx, w)
    return w, loss
