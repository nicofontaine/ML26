"""Cross-validation experiments behind the tables of the report.

Run from the repository root, with the data in dataset/:
    python experiments.py methods        # Table I: the six basic methods
    python experiments.py features       # ablation of the preprocessing options
    python experiments.py improvements   # class weighting
    python experiments.py final          # plain vs previous vs final model

All experiments use the same 5 stratified folds (seed 1), so their numbers are
comparable. The saved outputs are in results/.
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
from run import FINAL_PREPROCESSING

DATA_PATH = "dataset"
K_FOLD = 5
SEED = 1

# Preprocessing of the previous final model: interaction features only
INTERACTIONS = {"interaction_columns": INTERACTION_COLUMNS}


def load_data():
    """Load the training data, with labels converted to {0, 1}."""
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    return x_train, (y_train == 1).astype(float), load_feature_names(DATA_PATH)


def zeros(tx):
    """Initial weights for the iterative methods."""
    return np.zeros(tx.shape[1])


def reg_logistic_model():
    """The model of run.py: reg_logistic, lambda=1e-4, gamma=0.3, 500 iterations."""
    return {
        "reg_logistic": lambda y, tx: reg_logistic_regression(
            y, tx, 1e-4, zeros(tx), 500, 0.3
        )[0]
    }


def print_differences(f1, pairs):
    """Per-fold differences between pairs of F1 arrays: mean, std, folds better."""
    for new, old in pairs:
        diff = f1[new] - f1[old]
        print(
            f"{new} vs {old}: mean difference {diff.mean():+.4f},"
            f" std of the per-fold differences {diff.std():.4f},"
            f" better on {np.sum(diff > 0)}/{len(diff)} folds"
        )


def compare_methods(x, y, names):
    """Table I: 5-fold F1 of the six methods of implementations.py."""
    np.random.seed(0)  # SGD samples its data points with np.random
    models = {
        "least_squares": lambda y, tx: least_squares(y, tx)[0],
        "ridge lambda=1e-4": lambda y, tx: ridge_regression(y, tx, 1e-4)[0],
        "mse_gd gamma=0.05 iters=500": lambda y, tx: mean_squared_error_gd(
            y, tx, zeros(tx), 500, 0.05
        )[0],
        "mse_sgd gamma=0.001 iters=100000": lambda y, tx: mean_squared_error_sgd(
            y, tx, zeros(tx), 100000, 0.001
        )[0],
        "logistic gamma=0.3 iters=500": lambda y, tx: logistic_regression(
            y, tx, zeros(tx), 500, 0.3
        )[0],
        "reg_logistic lambda=1e-4 gamma=0.3 iters=500": lambda y, tx: (
            reg_logistic_regression(y, tx, 1e-4, zeros(tx), 500, 0.3)[0]
        ),
    }
    print_results(cross_validate(x, y, names, models, K_FOLD, SEED))


def compare_features(x, y, names):
    """Ablation of the preprocessing options of the final model.

    Every variant is the previous final model (reg_logistic with interaction
    features, the "baseline") with one option added. "final (all options)" uses
    FINAL_PREPROCESSING, the settings of run.py.
    """
    variants = {
        "baseline": {},
        "flags for dropped columns": {"dropped_flags": True},
        "max_missing=0.95": {"max_missing": 0.95},
        "comorbidity count": {"comorbidity": True},
        "_AGE80 features": {"age80": True},
        "final (all options)": FINAL_PREPROCESSING,
    }
    f1 = {}
    for label, options in variants.items():
        prep = {**INTERACTIONS, **options}
        res = cross_validate(x, y, names, reg_logistic_model(), K_FOLD, SEED, prep)
        f1[label] = res["reg_logistic"]["f1"]
        print(f"done: {label} ({f1[label].mean():.4f})", flush=True)

    # Results table: change from baseline, and on how many folds it was better
    baseline = f1["baseline"]
    print()
    print(f"{'variant':28s} {'F1 (mean +/- std)':18s} {'change':>8s}  better folds")
    for label, scores in f1.items():
        diff = scores - baseline
        print(
            f"{label:28s} {scores.mean():.4f} +/- {scores.std():.4f}"
            f" {diff.mean():+8.4f}  {np.sum(diff > 0)}/{len(diff)}"
        )


def compare_improvements(x, y, names):
    """Class weighting (positive weight 1, 3 and balanced) with interaction features.

    pos_weight=1 is exactly the previous final model.
    """
    balanced = (1 - y.mean()) / y.mean()  # makes both classes weigh the same in total
    models = {
        f"pos_weight={pw:.3g}": (
            lambda p: lambda y, tx: weighted_reg_logistic_regression(
                y, tx, 1e-4, zeros(tx), 500, 0.3, p
            )[0]
        )(pw)
        for pw in [1.0, 3.0, balanced]
    }
    res = cross_validate(x, y, names, models, K_FOLD, SEED, INTERACTIONS)
    print_results(res)
    print()
    f1 = {name: r["f1"] for name, r in res.items()}
    unweighted, *weighted = f1
    print_differences(f1, [(name, unweighted) for name in weighted])


def compare_final(x, y, names):
    """Per-fold F1 of plain reg_logistic, the previous and the final model."""
    setups = {
        "(a) reg_logistic, no interactions": {},
        "(b) previous final (interactions)": INTERACTIONS,
        "(c) new final (FINAL_PREPROCESSING)": FINAL_PREPROCESSING,
    }
    f1 = {}
    for label, prep in setups.items():
        res = cross_validate(x, y, names, reg_logistic_model(), K_FOLD, SEED, prep)
        f1[label] = res["reg_logistic"]["f1"]
        folds = " ".join(f"{f:.4f}" for f in f1[label])
        print(
            f"{label:38s} F1 = {f1[label].mean():.4f} +/- {f1[label].std():.4f}"
            f"   folds: {folds}",
            flush=True,
        )
    a, b, c = setups
    print()
    print_differences(f1, [(c, b), (c, a), (b, a)])


if __name__ == "__main__":
    experiments = {
        "methods": compare_methods,
        "features": compare_features,
        "improvements": compare_improvements,
        "final": compare_final,
    }
    if len(sys.argv) != 2 or sys.argv[1] not in experiments:
        raise SystemExit(f"usage: python experiments.py [{'|'.join(experiments)}]")
    x, y, names = load_data()
    experiments[sys.argv[1]](x, y, names)
