"""Extension of the methods in implementations.py, tested as an improvement.

Class weighting gave no gain once the decision threshold is tuned (see
"python experiments.py improvements"), so the final model in run.py does not
use it.
"""

import numpy as np

from implementations import calculate_loss_log, sigmoid


def weighted_reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iters, gamma, pos_weight=1.0
):
    """Regularized logistic regression with class weights, using gradient descent.

    Only ~9% of the labels are 1, so the plain loss is dominated by the negative
    examples. Each positive example is given pos_weight times the weight of a
    negative one. The weights are normalized to have mean 1, so the size of the
    gradient (and thus a good step size gamma) stays comparable to the unweighted
    method. pos_weight=1 gives exactly reg_logistic_regression.

    Args:
        y:  shape=(N, ), labels in {0, 1}
        tx: shape=(N, D)
        lambda_: scalar, regularization strength (penalty lambda_ * ||w||^2)
        initial_w: shape=(D, )
        max_iters: int
        gamma: float, step size
        pos_weight: float, weight of a positive example relative to a negative one

    Returns:
        w: last weight vector, shape=(D, )
        loss: unweighted logistic loss at the last w (without the penalty), scalar
    """
    weights = np.where(y == 1, pos_weight, 1.0)
    weights = weights / weights.mean()

    w = initial_w
    for _ in range(max_iters):
        sigma = sigmoid(tx @ w)
        gradient = tx.T @ (weights * (sigma - y)) / len(y) + 2 * lambda_ * w
        w = w - gamma * gradient

    loss = calculate_loss_log(y, tx, w)

    return w, loss
