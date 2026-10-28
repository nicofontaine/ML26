# ML Project 1: Predicting coronary heart disease (MICHD) from BRFSS survey data

EPFL CS-433 Machine Learning, Project 1. We predict whether a respondent of the
2015 Behavioral Risk Factor Surveillance System (BRFSS) has been diagnosed with
coronary heart disease or a heart attack (`_MICHD`), from 321 survey answers.
Only 8.8% of the 328,135 training respondents are positive, so models are
evaluated with the F1 score of the positive class.

Everything is implemented with NumPy only (plus matplotlib for the figures).

## Quick start

1. Download `x_train.csv`, `y_train.csv` and `x_test.csv` from the
   [AIcrowd challenge](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1)
   and put them, with these names, in a folder `dataset/` at the root of the
   repository.
2. With Python 3 and NumPy installed, run from the repository root:

   ```
   python run.py
   ```

   This trains the final model on the full training set and writes
   `submission.csv` (labels in {-1, 1}), the file we submitted to AIcrowd. It
   takes about two minutes and is deterministic: every run gives the same file.

## Files

| File | Content |
|---|---|
| `implementations.py` | The six required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression`. Each returns `(w, loss)`. |
| `preprocessing.py` | Turns the 321 raw columns into standardized features. `fit_preprocessing` computes every statistic on the training data; `transform` applies it unchanged to any data set. |
| `cross_validation.py` | Stratified k-fold cross-validation, F1 score, choice of the decision threshold. |
| `run.py` | Reproduces our final submission. `FINAL_PREPROCESSING` holds the preprocessing options of the final model. |
| `experiments.py` | The cross-validation experiments behind the tables of the report. |
| `plots.py` | The figures of the report. |
| `models.py` | Class-weighted logistic regression, tested as an improvement but not used in the final model. |
| `helpers.py` | Data loading and submission writing, provided by the course. |
| `results/` | Saved output of every experiment (see below). |
| `figures/` | The figures of the report (PDF and PNG). |
| `submission.csv` | Our submission, as written by `run.py`. |

## Final model

**Preprocessing** (`preprocessing.py`). All statistics come from the training
data only, so no information leaks from the test set (in cross-validation they
are recomputed on each training fold).

1. Drop 48 columns that describe the interview rather than the person (IDs,
   dates, phone-survey fields, sampling weights) or duplicate a cleaner computed
   column, and `HAREHAB1`, which is only asked after a heart attack (part of the
   label), so even whether it was answered reveals the label.
2. Decode the survey codes: "don't know" / "refused" (7/9, 77/99, 777/999)
   become missing; "none" (88, 555, 888) becomes 0; `ALCDAY5` is converted to
   drinking days per week.
3. Drop columns with more than 95% missing values (`max_missing=0.95`) and
   constant columns, but keep a 0/1 "was missing" flag for each dropped column
   (`dropped_flags=True`): many questions are only asked to some respondents,
   so whether a question was answered carries information.
4. Impute missing values with the training median (mode for categorical
   columns), and add a "was missing" flag for each kept column with missing
   values.
5. Clip continuous columns to their 1st-99th percentiles and log-transform the
   skewed ones; one-hot encode the categorical columns (3 to 10 values).
6. Add `COMORBIDITY`, the number of "yes" answers to 8 chronic-condition
   questions (high blood pressure, high cholesterol, diabetes, stroke, kidney
   disease, COPD, arthritis, depression) (`comorbidity=True`).
7. Add degree-2 interaction features (squares and pairwise products) of 11
   strong predictors and `COMORBIDITY` (`INTERACTION_COLUMNS`), with exact age
   `_AGE80` instead of 5-year age groups, plus the square of the centered age
   (`age80=True`).
8. Standardize, drop features that are constant or correlated above 0.95 with
   an earlier feature, and add a bias column.

**Model** (`run.py`). Regularized logistic regression (`reg_logistic_regression`)
with lambda = 1e-4, step size gamma = 0.3, 500 gradient descent steps from
w = 0. A respondent is predicted positive when the predicted probability is
above the threshold that maximizes F1 on the training set (about 0.2 instead of
0.5, because positives are rare).

## Reproducing the results of the report

All experiments use the same 5 stratified folds (seed 1), so their numbers are
directly comparable. Each prints the mean and standard deviation of the
validation F1 and the F1 of every fold. The decision threshold is always chosen
on the training folds, never on the validation fold.

| Command | What it compares | Output | Time |
|---|---|---|---|
| `python experiments.py methods` | The six methods of `implementations.py`, several lambdas and iteration counts | `results/methods.txt` | ~15 min |
| `python experiments.py improvements` | Class weighting (positive weight 1, 3, balanced), with and without interaction features | `results/improvements.txt` | ~12 min |
| `python experiments.py groups` | One model vs. separate models per sex / per age group | `results/groups.txt` | ~15 min |
| `python experiments.py features` | Each preprocessing option of the final model, one at a time and all together | `results/features.txt` | ~40 min |
| `python experiments.py final` | Plain reg. logistic vs. previous final model vs. final model, per fold | `results/final_comparison.txt` | ~10 min |
| `python plots.py all` | Figures: F1 against lambda, against the decision threshold, and training loss against iterations | `figures/`, `results/plots.txt` | ~30 min |
| `python run.py` | Final model on the full training set | `submission.csv`, `results/run.txt` | ~2 min |

To save an output, redirect it, e.g. `python experiments.py final > results/final_comparison.txt`.

## Results

5-fold stratified cross-validation F1 (mean ± std over folds):

| Model | CV F1 |
|---|---|
| Least squares / ridge regression / MSE gradient descent | 0.416 |
| Logistic regression, regularized logistic regression | 0.4278 ± 0.0050 |
| + interaction features (previous final model) | 0.4292 ± 0.0048 |
| + missing flags, max_missing 0.95, comorbidity count, exact age (**final model**) | **0.4318 ± 0.0043** |

The final model is better than the previous one on all 5 folds (+0.0026 on
average). The previous final model scored F1 = 0.441 on the AIcrowd test set.

What mattered most is the decision threshold: with the default threshold 0.5
the final model reaches only F1 = 0.236, against 0.432 with the threshold that
maximizes F1 (a probability of about 0.21). Class weighting, separate models per
sex or age group, and treating the unknown age group as missing did not improve
the F1 (see `results/`).
