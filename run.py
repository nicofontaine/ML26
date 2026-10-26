"""Reproduce our best submission.

Run from the repository root, with x_train.csv, y_train.csv and x_test.csv in dataset/:
    python run.py

Model: regularized logistic regression (lambda = 1e-4, gamma = 0.3, 500 gradient
descent steps) on the preprocessed features plus degree-2 interaction features.
The decision threshold is the one that maximizes F1 on the training set.
How these choices were made: see experiments.py and the report.

Output: submission.csv, with labels in {-1, 1}.
"""

import numpy as np

from cross_validation import best_threshold, f1_score, predict_labels
from helpers import create_csv_submission, load_csv_data
from implementations import reg_logistic_regression
from preprocessing import (
    INTERACTION_COLUMNS,
    fit_preprocessing,
    load_feature_names,
    transform,
)

DATA_PATH = "dataset"
OUTPUT_PATH = "submission.csv"

LAMBDA = 1e-4
GAMMA = 0.3
MAX_ITERS = 500


def main():
    # 1. Load the data; the logistic methods expect labels in {0, 1}
    print("Loading data...")
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    y_train = (y_train == 1).astype(float)
    names = load_feature_names(DATA_PATH)

    # 2. Preprocessing: statistics from the training set, applied to both sets
    print("Preprocessing...")
    params = fit_preprocessing(x_train, names, interaction_columns=INTERACTION_COLUMNS)
    tx_train = transform(x_train, names, params)
    tx_test = transform(x_test, names, params)
    print(f"  {tx_train.shape[1] - 1} features (+ bias)")

    # 3. Train the model
    print("Training...")
    w, loss = reg_logistic_regression(
        y_train, tx_train, LAMBDA, np.zeros(tx_train.shape[1]), MAX_ITERS, GAMMA
    )

    # 4. Decision threshold that maximizes F1 on the training set
    threshold, train_f1 = best_threshold(y_train, tx_train @ w)
    print(f"  training loss {loss:.4f}, training F1 {train_f1:.4f}")

    # 5. Predict the test set and write the submission
    y_pred = predict_labels(tx_test, w, threshold)
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}: {np.mean(y_pred == 1):.1%} predicted positive")


if __name__ == "__main__":
    main()
