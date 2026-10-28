"""Figure and convergence numbers of the report.

Run from the repository root, with the data in dataset/:
    python plots.py [figure|convergence|all]     (default: all)

    figure       Fig. 1 of the report (figures/lambda_threshold.pdf/.png):
                 (a) train and validation F1 against lambda (5-fold CV),
                 (b) validation F1, precision and recall against the decision
                 threshold on the predicted probability
    convergence  training loss against gradient descent iterations for several
                 step sizes gamma (numbers quoted in the text, no figure)
    all          both

Same setup as run.py (FINAL_PREPROCESSING, lambda = 1e-4, gamma = 0.3, 500
iterations, w0 = 0, labels in {0, 1}) and the same 5 stratified folds as
experiments.py (seed 1). The preprocessing is fitted once per fold, on its
training part only. The numbers behind the figure are printed.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # write files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cross_validation import best_threshold, build_k_indices, f1_score  # noqa: E402
from helpers import load_csv_data  # noqa: E402
from implementations import (  # noqa: E402
    calculate_gradient_log,
    calculate_loss_log,
    reg_logistic_regression,
    sigmoid,
)
from preprocessing import fit_preprocessing, load_feature_names, transform  # noqa: E402
from run import FINAL_PREPROCESSING  # noqa: E402

DATA_PATH = "dataset"
FIGURE_DIR = "figures"
K_FOLD = 5
SEED = 1

# Settings of run.py
LAMBDA = 1e-4
GAMMA = 0.3
MAX_ITERS = 500

LAMBDAS = np.logspace(-6, -1, 11)  # Fig. 1a
THRESHOLDS = np.linspace(0.02, 0.7, 69)  # Fig. 1b, on the predicted probability
GAMMAS = [0.01, 0.05, 0.1, 0.3, 1.0]  # convergence
CONVERGENCE_ITERS = 1000  # convergence

FULL_SIZE = (7.0, 2.5)  # both IEEE columns, in inches
plt.rcParams.update(
    {
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    }
)


# ---------- Data and models ----------


def prepare_folds(x, y, names, which):
    """Preprocess the requested folds: fit on the training part, apply to both.

    Args:
        x: raw training data, shape=(N, 321)
        y: labels in {0, 1}, shape=(N,)
        names: list of the 321 column names
        which: fold numbers to prepare

    Returns:
        dict {fold number: {"tx_tr", "y_tr", "tx_va", "y_va"}}
    """
    k_indices = build_k_indices(y, K_FOLD, SEED)
    folds = {}
    for k in which:
        va = k_indices[k]
        tr = np.concatenate([f for j, f in enumerate(k_indices) if j != k])
        params = fit_preprocessing(x[tr], names, **FINAL_PREPROCESSING)
        folds[k] = {
            "tx_tr": transform(x[tr], names, params),
            "y_tr": y[tr],
            "tx_va": transform(x[va], names, params),
            "y_va": y[va],
        }
        print(f"  fold {k + 1}/{K_FOLD} preprocessed", flush=True)
    return folds


def trained_weights(folds, k, lambda_, cache):
    """Weights of reg_logistic_regression on the training part of fold k.

    Trained once per (lambda_, fold) and cached, so Fig. 1a and 1b share them.
    """
    key = (float(lambda_), k)
    if key not in cache:
        fold = folds[k]
        w0 = np.zeros(fold["tx_tr"].shape[1])
        cache[key], _ = reg_logistic_regression(
            fold["y_tr"], fold["tx_tr"], lambda_, w0, MAX_ITERS, GAMMA
        )
    return cache[key]


def precision_recall(y_true, y_pred):
    """Precision and recall of the positive class (0 when undefined)."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    n_pred, n_true = np.sum(y_pred == 1), np.sum(y_true == 1)
    precision = tp / n_pred if n_pred > 0 else 0.0
    recall = tp / n_true if n_true > 0 else 0.0
    return precision, recall


def save(fig, name):
    """Save a figure as PDF and PNG in FIGURE_DIR and return the file paths."""
    os.makedirs(FIGURE_DIR, exist_ok=True)
    paths = [os.path.join(FIGURE_DIR, f"{name}.{ext}") for ext in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


# ---------- Fig. 1a: lambda ----------


def lambda_curve(folds, cache):
    """Train and validation F1 and loss for every lambda and fold.

    As in cross_validate, the threshold is chosen on the training scores.

    Returns:
        dict of arrays of shape (len(LAMBDAS), K_FOLD)
    """
    shape = (len(LAMBDAS), K_FOLD)
    res = {key: np.zeros(shape) for key in ["f1_tr", "f1_va", "loss_tr", "loss_va"]}
    for i, lambda_ in enumerate(LAMBDAS):
        for k, fold in folds.items():
            w = trained_weights(folds, k, lambda_, cache)
            threshold, res["f1_tr"][i, k] = best_threshold(
                fold["y_tr"], fold["tx_tr"] @ w
            )
            y_pred = (fold["tx_va"] @ w >= threshold).astype(float)
            res["f1_va"][i, k] = f1_score(fold["y_va"], y_pred)
            res["loss_tr"][i, k] = calculate_loss_log(fold["y_tr"], fold["tx_tr"], w)
            res["loss_va"][i, k] = calculate_loss_log(fold["y_va"], fold["tx_va"], w)
        print(f"  lambda {lambda_:.1e} done", flush=True)

    print("\nFig. 1a: F1 and loss (without penalty) against lambda, 5 folds")
    print(
        f"{'lambda':>9s}  {'train F1':>15s}  {'validation F1':>15s}"
        f"  {'train loss':>10s}  {'val. loss':>10s}"
    )
    for i, lambda_ in enumerate(LAMBDAS):
        print(
            f"{lambda_:9.1e}"
            f"  {res['f1_tr'][i].mean():.4f} +/- {res['f1_tr'][i].std():.4f}"
            f"  {res['f1_va'][i].mean():.4f} +/- {res['f1_va'][i].std():.4f}"
            f"  {res['loss_tr'][i].mean():10.4f}  {res['loss_va'][i].mean():10.4f}"
        )
    best = int(np.argmax(res["f1_va"].mean(axis=1)))
    print(
        f"highest mean validation F1: lambda = {LAMBDAS[best]:.1e}"
        f" ({res['f1_va'][best].mean():.4f})"
    )
    return res


def draw_lambda(ax, res):
    """Draw Fig. 1a on the axes ax."""
    for key, label, color in [
        ("f1_tr", "train", "tab:blue"),
        ("f1_va", "validation", "tab:red"),
    ]:
        mean, std = res[key].mean(axis=1), res[key].std(axis=1)
        ax.semilogx(LAMBDAS, mean, marker=".", color=color, label=label)
        ax.fill_between(LAMBDAS, mean - std, mean + std, color=color, alpha=0.2)
    ax.axvline(
        LAMBDA, color="gray", linestyle="--", label=f"run.py ($\\lambda$={LAMBDA:g})"
    )
    ax.set_xlabel("regularization $\\lambda$")
    ax.set_ylabel("F1 score")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)


# ---------- Fig. 1b: decision threshold ----------


def threshold_curve(folds, cache):
    """Validation F1, precision and recall against the probability threshold.

    Uses the out-of-fold probabilities sigmoid(tx_va @ w) of each fold
    (lambda = LAMBDA). The best threshold here is chosen on the validation
    folds themselves, so the figure describes the tradeoff; it is not an
    unbiased estimate (cross_validate chooses the threshold on training data).

    Returns:
        dict with arrays of shape (len(THRESHOLDS), K_FOLD) and summary numbers
    """
    shape = (len(THRESHOLDS), K_FOLD)
    res = {key: np.zeros(shape) for key in ["f1", "precision", "recall"]}
    probs, f1_default, train_thresholds = {}, [], []
    for k, fold in folds.items():
        w = trained_weights(folds, k, LAMBDA, cache)
        probs[k] = sigmoid(fold["tx_va"] @ w)
        for i, t in enumerate(THRESHOLDS):
            y_pred = (probs[k] >= t).astype(float)
            res["f1"][i, k] = f1_score(fold["y_va"], y_pred)
            res["precision"][i, k], res["recall"][i, k] = precision_recall(
                fold["y_va"], y_pred
            )
        f1_default.append(f1_score(fold["y_va"], (probs[k] >= 0.5).astype(float)))
        # The threshold the pipeline itself would use (chosen on training scores)
        score_threshold, _ = best_threshold(fold["y_tr"], fold["tx_tr"] @ w)
        train_thresholds.append(sigmoid(score_threshold))

    mean_f1 = res["f1"].mean(axis=1)
    best = int(np.argmax(mean_f1))
    res["best"] = THRESHOLDS[best]
    res["best_f1"] = mean_f1[best]
    res["default_f1"] = np.mean(f1_default)
    share = [np.mean(probs[k] >= res["best"]) for k in folds]

    print("\nFig. 1b: validation scores against the probability threshold")
    print(
        f"threshold 0.5 (default): F1 = {res['default_f1']:.4f}"
        f" +/- {np.std(f1_default):.4f}"
    )
    print(
        f"best threshold {res['best']:.2f}: F1 = {res['best_f1']:.4f}"
        f" +/- {res['f1'][best].std():.4f},"
        f" precision {res['precision'][best].mean():.3f},"
        f" recall {res['recall'][best].mean():.3f}"
    )
    print(f"share predicted positive at {res['best']:.2f}: {np.mean(share):.1%}")
    print(
        f"threshold chosen on the training folds (as in run.py):"
        f" {np.mean(train_thresholds):.3f} +/- {np.std(train_thresholds):.3f}"
    )
    return res


def draw_threshold(ax, res):
    """Draw Fig. 1b on the axes ax."""
    mean, std = res["f1"].mean(axis=1), res["f1"].std(axis=1)
    ax.plot(THRESHOLDS, mean, color="tab:red", label="F1")
    ax.fill_between(THRESHOLDS, mean - std, mean + std, color="tab:red", alpha=0.2)
    ax.plot(
        THRESHOLDS,
        res["precision"].mean(axis=1),
        color="tab:green",
        linewidth=0.8,
        label="precision",
    )
    ax.plot(
        THRESHOLDS,
        res["recall"].mean(axis=1),
        color="tab:purple",
        linewidth=0.8,
        label="recall",
    )
    ax.axvline(
        0.5,
        color="gray",
        linestyle=":",
        label=f"0.5 (default), F1={res['default_f1']:.3f}",
    )
    ax.axvline(
        res["best"],
        color="black",
        linestyle="--",
        label=f"{res['best']:.2f} (best), F1={res['best_f1']:.3f}",
    )
    ax.set_xlabel("decision threshold on predicted probability")
    ax.set_ylabel("validation score")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)


# ---------- Convergence (numbers in the text) ----------


def convergence_curve(fold):
    """Training loss after every gradient descent step, for each gamma in GAMMAS.

    The same update as reg_logistic_regression, written out so that the loss
    can be recorded at every iteration.

    Returns:
        dict {gamma: losses}, losses[i] = loss after i steps, i = 0..CONVERGENCE_ITERS
    """
    y, tx = fold["y_tr"], fold["tx_tr"]
    res = {}
    for gamma in GAMMAS:
        w = np.zeros(tx.shape[1])
        losses = [calculate_loss_log(y, tx, w)]
        for _ in range(CONVERGENCE_ITERS):
            gradient = calculate_gradient_log(y, tx, w) + 2 * LAMBDA * w
            w = w - gamma * gradient
            losses.append(calculate_loss_log(y, tx, w))
        res[gamma] = np.array(losses)
        print(f"  gamma {gamma} done", flush=True)

    # The manual loop must match reg_logistic_regression (run.py's settings)
    _, loss = reg_logistic_regression(y, tx, LAMBDA, np.zeros(tx.shape[1]), 500, GAMMA)
    assert abs(res[GAMMA][MAX_ITERS] - loss) < 1e-10, (res[GAMMA][MAX_ITERS], loss)

    print("\nConvergence: training loss (fold 1) after n gradient descent steps")
    steps = [100, 250, 500, 1000]
    print(f"{'gamma':>6s}" + "".join(f"  {f'n={n}':>8s}" for n in steps))
    for gamma in GAMMAS:
        print(f"{gamma:6g}" + "".join(f"  {res[gamma][n]:8.5f}" for n in steps))
    change = (res[GAMMA][1000] - res[GAMMA][500]) / res[GAMMA][500]
    print(
        f"gamma={GAMMA}: loss after 500 steps matches reg_logistic_regression"
        f" ({loss:.6f}); relative change 500 -> 1000 steps: {change:+.2%}"
    )
    return res


# ---------- Main ----------


def main(mode):
    """Compute the numbers of the given mode, and save the figure for "figure"."""
    modes = ["figure", "convergence", "all"]
    if mode not in modes:
        raise SystemExit(f"usage: python plots.py [{'|'.join(modes)}]")

    print("Loading data...")
    x, _, y, _, _ = load_csv_data(DATA_PATH)
    y = (y == 1).astype(float)
    names = load_feature_names(DATA_PATH)

    print("Preprocessing...")
    which = [0] if mode == "convergence" else list(range(K_FOLD))
    folds = prepare_folds(x, y, names, which)
    del x  # only the preprocessed folds are needed from here on

    if mode in ("figure", "all"):
        cache = {}
        lambda_res = lambda_curve(folds, cache)
        threshold_res = threshold_curve(folds, cache)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FULL_SIZE)
        draw_lambda(ax1, lambda_res)
        draw_threshold(ax2, threshold_res)
        ax1.set_title("(a) F1 against $\\lambda$")
        ax2.set_title("(b) F1 against the decision threshold")
        fig.tight_layout()
        print("\nFiles created:")
        for path in save(fig, "lambda_threshold"):
            print(f"  {path}")
    if mode in ("convergence", "all"):
        convergence_curve(folds[0])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
