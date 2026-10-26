"""K-fold cross-validation for project 1 (adapted from the Lab 4 cross-validation).

For each fold, the preprocessing is fitted on the training part only, every
model is trained on it, the decision threshold is chosen to maximize F1 on the
training part, and F1 is then measured on the held-out part. Choosing the
threshold on the training part keeps the validation score an honest estimate.

Typical use:
    models = {
        "reg_logistic": lambda y, tx: reg_logistic_regression(
            y, tx, 1e-4, np.zeros(tx.shape[1]), 500, 0.3
        )[0],
    }
    results = cross_validate(x_train, y_train, names, models)
"""

import numpy as np

from preprocessing import fit_preprocessing, transform


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


def f1_score(y_true, y_pred):
    """F1 score of the positive class.

    Args:
        y_true: true labels in {0, 1}, shape=(N,)
        y_pred: predicted labels in {0, 1}, shape=(N,)

    Returns:
        F1 = 2 TP / (2 TP + FP + FN), scalar
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    return 2 * tp / (np.sum(y_pred == 1) + np.sum(y_true == 1))


def best_threshold(y, scores):
    """Find the threshold on the model scores that maximizes F1.

    The candidates are quantiles of the scores, so this works for any model
    (probabilities from logistic regression or raw outputs from least squares).

    Args:
        y: true labels in {0, 1}, shape=(N,)
        scores: model outputs tx @ w, shape=(N,)

    Returns:
        threshold: scalar, predict 1 when score >= threshold
        f1: F1 score obtained with this threshold
    """
    # Between 0.5% and 50% of the points predicted positive
    candidates = np.quantile(scores, np.linspace(0.5, 0.995, 100))
    f1s = [f1_score(y, (scores >= t).astype(float)) for t in candidates]
    best = int(np.argmax(f1s))
    return candidates[best], f1s[best]


def predict_labels(tx, w, threshold):
    """Predict labels in {-1, 1}, the format required for the submission.

    Args:
        tx: preprocessed features, shape=(N, D)
        w: model weights, shape=(D,)
        threshold: scalar, from best_threshold

    Returns:
        predictions in {-1, 1}, shape=(N,)
    """
    return np.where(tx @ w >= threshold, 1, -1)


def cross_validate(x, y, names, models, k_fold=5, seed=1, prep_kwargs=None):
    """K-fold cross-validation of several models on the same folds.

    The preprocessing is fitted once per fold and shared by all models, so
    comparing many models costs little more than comparing one.

    Args:
        x: raw training data, shape=(N, 321)
        y: labels in {0, 1}, shape=(N,)
        names: list of the 321 column names
        models: dict {model name: function (y, tx) -> w}
        k_fold: number of folds
        seed: random seed for the folds
        prep_kwargs: dict of arguments for fit_preprocessing

    Returns:
        dict {model name: {"f1": array of the k validation F1 scores,
                           "threshold": array of the k chosen thresholds}}
    """
    prep_kwargs = prep_kwargs or {}
    k_indices = build_k_indices(y, k_fold, seed)
    results = {name: {"f1": [], "threshold": []} for name in models}

    for k in range(k_fold):
        te_idx = k_indices[k]
        tr_idx = k_indices[np.arange(k_fold) != k].reshape(-1)

        # Preprocessing statistics come from the training part of this fold only
        params = fit_preprocessing(x[tr_idx], names, **prep_kwargs)
        tx_tr = transform(x[tr_idx], names, params)
        tx_te = transform(x[te_idx], names, params)

        for name, train in models.items():
            w = train(y[tr_idx], tx_tr)
            threshold, _ = best_threshold(y[tr_idx], tx_tr @ w)
            y_pred = (tx_te @ w >= threshold).astype(float)
            results[name]["f1"].append(f1_score(y[te_idx], y_pred))
            results[name]["threshold"].append(threshold)

    for name in models:
        results[name] = {key: np.array(val) for key, val in results[name].items()}
    return results


def print_results(results):
    """Print mean and standard deviation of the validation F1 of each model."""
    for name, res in results.items():
        print(f"{name:45s} F1 = {res['f1'].mean():.4f} +/- {res['f1'].std():.4f}")
