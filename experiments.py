"""Cross-validation experiments reported in the report.

Run from the repository root, with the data in dataset/:
    python experiments.py methods        # compare the six basic methods
    python experiments.py improvements   # class weighting and interaction features
    python experiments.py groups         # separate models per sex / age group
    python experiments.py features       # ablation of the feature options

All experiments use the same 5 folds (seed 1), so their numbers are comparable.
"""

import sys

import numpy as np

from cross_validation import cross_validate, print_results
from helpers import load_csv_data
from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)
from models import weighted_reg_logistic_regression
from preprocessing import INTERACTION_COLUMNS, load_feature_names

DATA_PATH = "dataset"
K_FOLD = 5
SEED = 1


def load_data():
    """Load the training data, with labels converted to {0, 1}."""
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    return x_train, (y_train == 1).astype(float), load_feature_names(DATA_PATH)


def zeros(tx):
    """Initial weights for the iterative methods."""
    return np.zeros(tx.shape[1])


def compare_methods(x, y, names):
    """5-fold F1 of the six methods of implementations.py."""
    np.random.seed(0)  # SGD samples its data points with np.random
    models = {
        "least_squares": lambda y, tx: least_squares(y, tx)[0],
        "ridge lambda=1e-4": lambda y, tx: ridge_regression(y, tx, 1e-4)[0],
        "ridge lambda=1e-2": lambda y, tx: ridge_regression(y, tx, 1e-2)[0],
        "mse_gd gamma=0.05 iters=500": lambda y, tx: mean_squared_error_gd(
            y, tx, zeros(tx), 500, 0.05
        )[0],
        "mse_sgd gamma=0.001 iters=100000": lambda y, tx: mean_squared_error_sgd(
            y, tx, zeros(tx), 100000, 0.001
        )[0],
        "logistic gamma=0.3 iters=500": lambda y, tx: logistic_regression(
            y, tx, zeros(tx), 500, 0.3
        )[0],
        "logistic gamma=0.3 iters=2000": lambda y, tx: logistic_regression(
            y, tx, zeros(tx), 2000, 0.3
        )[0],
    }
    for lambda_ in [1e-5, 1e-4, 1e-3, 1e-2]:
        models[f"reg_logistic lambda={lambda_:g} gamma=0.3 iters=500"] = (
            lambda l: lambda y, tx: reg_logistic_regression(
                y, tx, l, zeros(tx), 500, 0.3
            )[0]
        )(lambda_)
    print_results(cross_validate(x, y, names, models, K_FOLD, SEED))


def compare_improvements(x, y, names):
    """5-fold F1 of class weighting, with and without interaction features."""
    balanced = (1 - y.mean()) / y.mean()  # makes both classes weigh the same in total
    models = {
        f"pos_weight={pw:.3g}": (
            lambda p: lambda y, tx: weighted_reg_logistic_regression(
                y, tx, 1e-4, zeros(tx), 500, 0.3, p
            )[0]
        )(pw)
        for pw in [1.0, 3.0, balanced]
    }
    for label, prep in [
        ("no interactions", {}),
        ("with interactions", {"interaction_columns": INTERACTION_COLUMNS}),
    ]:
        print(f"--- {label}")
        print_results(cross_validate(x, y, names, models, K_FOLD, SEED, prep))


def compare_groups(x, y, names):
    """5-fold F1 of one model for everyone vs. separate models per group."""
    models = {
        "reg_logistic lambda=1e-4": lambda y, tx: reg_logistic_regression(
            y, tx, 1e-4, zeros(tx), 500, 0.3
        )[0]
    }
    for label, group in [
        ("one model (current)", None),
        ("one model per sex", "SEX"),
        ("one model per age group (under / over 65)", "_AGE65YR"),
    ]:
        prep = {"interaction_columns": INTERACTION_COLUMNS, "group_column": group}
        print(f"--- {label}")
        print_results(cross_validate(x, y, names, models, K_FOLD, SEED, prep))


def compare_features(x, y, names):
    """Ablation of the feature options of fit_preprocessing, on the same folds.

    Every variant is the current model (reg_logistic, lambda=1e-4, gamma=0.3,
    500 iterations, interaction features) with one option changed. The "best
    combination" combines every option whose mean F1 is above the baseline's;
    it is chosen on these same folds, so its estimate is slightly optimistic.
    """
    models = {
        "reg_logistic": lambda y, tx: reg_logistic_regression(
            y, tx, 1e-4, zeros(tx), 500, 0.3
        )[0]
    }
    base = {"interaction_columns": INTERACTION_COLUMNS}
    variants = {
        "baseline": {},
        "age fix": {"age_fix": True},
        "flags for dropped columns": {"dropped_flags": True},
        "max_missing=0.95": {"max_missing": 0.95},
        "comorbidity count": {"comorbidity": True},
        "_AGE80 features": {"age80": True},
    }

    f1 = {}
    for label, options in variants.items():
        res = cross_validate(x, y, names, models, K_FOLD, SEED, {**base, **options})
        f1[label] = res["reg_logistic"]["f1"]
        print(f"done: {label} ({f1[label].mean():.4f})", flush=True)

    baseline = f1["baseline"]
    better = [
        label
        for label in variants
        if label != "baseline" and f1[label].mean() > baseline.mean()
    ]
    if better:
        combined = {}
        for label in better:
            combined.update(variants[label])
        label = "best combination: " + " + ".join(better)
        res = cross_validate(x, y, names, models, K_FOLD, SEED, {**base, **combined})
        f1[label] = res["reg_logistic"]["f1"]
    else:
        print("no option beat the baseline, so there is no combination to test")

    # Results table: change from baseline, and on how many folds it was better
    print()
    print(f"{'variant':60s} {'F1 (mean +/- std)':18s} {'change':>8s}  better folds")
    for label, scores in f1.items():
        diff = scores - baseline
        print(
            f"{label:60s} {scores.mean():.4f} +/- {scores.std():.4f}"
            f" {diff.mean():+8.4f}  {np.sum(diff > 0)}/{len(diff)}"
        )
    print(f"\nthreshold for changing the defaults: change > {baseline.std():.4f}")


if __name__ == "__main__":
    experiment = sys.argv[1] if len(sys.argv) > 1 else "improvements"
    x, y, names = load_data()
    experiments = {
        "methods": compare_methods,
        "improvements": compare_improvements,
        "groups": compare_groups,
        "features": compare_features,
    }
    experiments[experiment](x, y, names)
