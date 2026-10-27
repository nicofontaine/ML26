"""K-fold cross-validation for project 1 (adapted from the Lab 4 cross-validation).

For each fold, preprocessing and model fitting use only the training rows. The
decision threshold is selected from the model's training scores, then F1 is
measured on the held-out rows. The held-out fold is never used to choose the
threshold, although an in-sample threshold can be less well calibrated than
one selected from separate calibration data.

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
    """Build reproducible, stratified folds that include every sample.

    Args:
        y:      Labels for N samples, shape=(N,)
        k_fold: Number of folds.
        seed:   Random seed used to shuffle samples.

    Returns:
        A list containing one 1D integer-index array per fold. Every sample is
        included, and each class is spread as evenly as possible across folds.
    """
    y = np.asarray(y)

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array of labels")

    if k_fold < 2 or k_fold > len(y):
        raise ValueError("k_fold must be between 2 and the number of samples")

    # Stratification can only put every class in every fold when each class
    # has at least k_fold samples. Fail clearly instead of making empty or
    # unrepresentative folds for a class.
    for label in np.unique(y):
        class_count = np.sum(y == label)
        if class_count < k_fold:
            raise ValueError(
                f"Each class needs at least k_fold samples; class {label} "
                f"has {class_count}"
            )

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k_fold)]

    # Split each class across all folds so the class proportions stay similar.
    # np.array_split distributes any remainder instead of dropping samples.
    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        rng.shuffle(class_indices)

        for fold_id, class_part in enumerate(np.array_split(class_indices, k_fold)):
            folds[fold_id].extend(class_part.tolist())

    # Shuffle within each fold so examples are not grouped by label.
    result = []
    for fold in folds:
        fold = np.asarray(fold, dtype=int)
        rng.shuffle(fold)
        result.append(fold)

    return result


def f1_score(y_true, y_pred):
    """F1 score of the positive class.

    Args:
        y_true: true labels in {0, 1}, shape=(N,)
        y_pred: predicted labels in {0, 1}, shape=(N,)

    Returns:
        F1 = 2 TP / (2 TP + FP + FN), scalar
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    denominator = np.sum(y_pred == 1) + np.sum(y_true == 1)

    # F1 is undefined when there are no positive labels or predictions.
    if denominator == 0:
        return 0.0

    return float(2 * tp / denominator)


def best_threshold(y, scores):
    """Search score quantiles for a threshold with a high F1 score.

    Checking 101 quantiles is a simple approximation to checking every
    possible threshold. It works with probabilities or raw model scores.

    Args:
        y: true labels in {0, 1}, shape=(N,)
        scores: model outputs tx @ w, shape=(N,)

    Returns:
        threshold: scalar, predict 1 when score >= threshold
        f1: F1 score obtained with this threshold
    """
    # Sample thresholds across the full score range. Including the minimum
    # lets the search consider predicting every sample as positive.
    candidates = np.quantile(scores, np.linspace(0.0, 1.0, 101))
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
        # Combine all other folds; this also works when fold sizes differ.
        tr_idx = np.concatenate(
            [fold for fold_id, fold in enumerate(k_indices) if fold_id != k]
        )

        # Preprocessing statistics come from the training part of this fold only
        params = fit_preprocessing(x[tr_idx], names, **prep_kwargs)
        tx_tr = transform(x[tr_idx], names, params)
        tx_te = transform(x[te_idx], names, params)

        for name, train in models.items():
            w = train(y[tr_idx], tx_tr)

            # Choose the threshold from training-fold scores only. Keep the
            # held-out fold untouched until the final F1 evaluation below.
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
