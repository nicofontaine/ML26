# -*- coding: utf-8 -*-
"""Train/test split and k-fold cross-validation (from Lab 3 and Lab 4 notebooks)."""

import numpy as np

from build_polynomial import build_poly
from costs import compute_mse
from ridge_regression import ridge_regression


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing. If ratio times the number of samples is not round
    you can use np.floor. Also check the documentation for np.random.permutation,
    it could be useful.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        y: numpy array of shape (N,).
        ratio: scalar in [0,1]
        seed: integer.

    Returns:
        x_tr: numpy array containing the train data.
        x_te: numpy array containing the test data.
        y_tr: numpy array containing the train labels.
        y_te: numpy array containing the test labels.

    >>> split_data(np.arange(13), np.arange(13), 0.8, 1)
    (array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]), array([ 2,  3,  4, 10,  1,  6,  0,  7, 12,  9]), array([ 8, 11,  5]))
    """
    # set seed
    np.random.seed(seed)

    # shuffle the indices
    indices = np.random.permutation(len(x))

    # split the data
    split_idx = int(np.floor(ratio * len(x)))
    x_tr = x[indices[:split_idx]]
    x_te = x[indices[split_idx:]]
    y_tr = y[indices[:split_idx]]
    y_te = y[indices[split_idx:]]

    return x_tr, x_te, y_tr, y_te


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527596753, 0.33555914361295197)
    """

    tr_idx = k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)
    te_idx = k_indices[k]
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_te, y_te = x[te_idx], y[te_idx]
    
    tx = build_poly(x_tr, degree)

    w = ridge_regression(y_tr, tx, lambda_)

    e_tr = y_tr - tx @ w
    e_te = y_te - build_poly(x_te, degree) @ w

    mse_tr = compute_mse(y_tr, tx, w)
    mse_te = compute_mse(y_te, build_poly(x_te, degree), w)

    loss_tr = np.sqrt(2 * mse_tr)
    loss_te = np.sqrt(2 * mse_te)

    return loss_tr, loss_te
