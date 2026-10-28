"""Reproduce our final submission.

Run from the repository root, with x_train.csv, y_train.csv and x_test.csv in dataset/:
    python run.py

Model: regularized logistic regression (lambda = 1e-4, gamma = 0.3, 500 gradient
descent steps, w0 = 0) on the features of preprocessing.py with the options in
FINAL_PREPROCESSING:
    - degree-2 interaction features of 11 strong predictors (INTERACTION_COLUMNS)
    - columns are kept up to 95% missing values (max_missing=0.95)
    - a 0/1 missing-value flag for each column dropped as mostly missing
    - COMORBIDITY: the number of chronic conditions answered "yes" (8 questions)
    - exact age (_AGE80) instead of 5-year age groups in the interactions, plus
      the square of the centered age
The decision threshold is the one that maximizes F1 on the training set.
How these choices were made: see experiments.py, results/ and the report.

Output: submission.csv, with labels in {-1, 1}.
"""

import numpy as np

from cross_validation import best_threshold, predict_labels
from helpers import create_csv_submission, load_csv_data
from implementations import reg_logistic_regression, sigmoid
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

# Options of fit_preprocessing used by the final model (also imported by
# experiments.py and plots.py, so all three use exactly the same features)
FINAL_PREPROCESSING = {
    "interaction_columns": INTERACTION_COLUMNS,
    "dropped_flags": True,
    "max_missing": 0.95,
    "comorbidity": True,
    "age80": True,
}


def main():
    # 1. Load the data; the logistic methods expect labels in {0, 1}
    print("Loading data...")
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    y_train = (y_train == 1).astype(float)
    names = load_feature_names(DATA_PATH)

    # 2. Preprocessing: statistics from the training set, applied to both sets
    print("Preprocessing...")
    params = fit_preprocessing(x_train, names, **FINAL_PREPROCESSING)
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
    print(
        f"  threshold: score {threshold:.4f}"
        f" = predicted probability {sigmoid(threshold):.4f}"
    )

    # 5. Predict the test set and write the submission
    y_pred = predict_labels(tx_test, w, threshold)
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}: {np.mean(y_pred == 1):.1%} predicted positive")


if __name__ == "__main__":
    main()
