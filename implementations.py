# Functions
import numpy as np


# Helper functions for linear regression using gradient descent
def compute_loss(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        w: shape=(D, ). The vector of model parameters.

    Returns:
        An array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    return compute_gradient(y, tx, w)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


# Mean Squared Error (MSE) using Gradient Descent (GD)
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N, D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: last weight vector of shape (D, ). The final model parameters after max_iters iterations of GD
        loss: corresponding loss after max_iters iterations of GD
    """
    # Define w
    w = initial_w

    # gradient descent
    for i in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient  # update w by gradient

    # compute loss
    loss = compute_loss(y, tx, w)

    return w, loss


# Mean Squared Error (MSE) using Stochastic Gradient Descent (SGD)
def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD (one data point per step)
        gamma: a scalar denoting the stepsize

    Returns:
        w: last weight vector of shape (D, ). The final model parameters after max_iters iterations of SGD
        loss: corresponding loss after max_iters iterations of SGD
    """
    # Define w
    w = initial_w

    # stochastic gradient descent
    for i in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=1, num_batches=1):
            gradient = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * gradient  # update w by stochastic gradient

    # compute loss
    loss = compute_loss(y, tx, w)

    return w, loss


# Linear Regression using Normal Equations
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns optimal weights and mse.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))[0]
    array([ 0.21212121, -0.12121212])
    """

    # Pseudo-inverse instead of solving tx.T @ tx w = tx.T @ y directly, so that
    # duplicate or constant columns (singular tx.T @ tx) do not raise an error
    w = np.linalg.pinv(tx) @ y

    e = y - tx @ w
    mse = 1 / (2 * len(y)) * np.dot(e, e)

    return w, mse


# Ridge Regression using Normal Equations
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: mse at w (without the penalty term), scalar.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)[0]
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)[0]
    array([0.03947092, 0.00319628])
    """

    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.eye(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    e = y - tx.dot(w)
    loss = e.dot(e) / (2 * len(e))  # MEAN SQUARE ERROR

    return w, loss


# Helper functions for Logistic Regression
def sigmoid(t):
    """apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """

    # Equal to 1 / (1 + exp(-t)), but does not overflow for large negative t
    return np.exp(-np.logaddexp(0, -t))


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
        gradient = (
            calculate_gradient_log(y, tx, w) + 2 * lambda_ * w
        )  # Penalized gradient
        w = w - gamma * gradient  # Gradient descent step

    loss = calculate_loss_log(y, tx, w)  # Compute loss

    return w, loss
