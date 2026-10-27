"""Preprocessing of the BRFSS heart disease data for project 1.

All statistics (which columns to keep, missing codes, fill values, categories,
clipping bounds, which columns to log-transform, mean and std, which features
to drop as redundant) are computed on the training set only by
'fit_preprocessing' and then applied unchanged to any data set by 'transform',
so train and test are processed identically and no information leaks from the
test set. When cross-validating, call 'fit_preprocessing' on each training fold.

Typical use:
    names = load_feature_names("dataset")
    params = fit_preprocessing(x_train, names)
    tx_train = transform(x_train, names, params)
    tx_test = transform(x_test, names, params)
"""

import os

import numpy as np

# Survey administration, identifiers, interview dates and sampling weights:
# they describe how the interview was done, not the person's health
ADMIN_COLUMNS = [
    "_STATE",
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
    "CTELENUM",
    "PVTRESD1",
    "COLGHOUS",
    "STATERES",
    "CELLFON3",
    "LADULT",
    "CTELNUM1",
    "CELLFON2",
    "CADULT",
    "PVTRESD2",
    "CCLGHOUS",
    "CSTATE",
    "LANDLINE",
    "QSTVER",
    "QSTLANG",
    "_STSTR",
    "_STRWT",
    "_RAWRAKE",
    "_WT2RAKE",
    "_CLLCPWT",
    "_DUALUSE",
    "_DUALCOR",
    "_LLCPWT",
    "FLSHTMY2",
    "HIVTSTD3",
]

# Raw answers in awkward units that also exist as a cleaner computed column
# (e.g. WEIGHT2 in pounds or kg -> WTKG3, FRUIT1 coded per day/week/month -> FRUTDA1_)
REDUNDANT_COLUMNS = [
    "WEIGHT2",
    "HEIGHT3",
    "HTIN4",
    "FRUITJU1",
    "FRUIT1",
    "FVBEANS",
    "FVGREEN",
    "FVORANG",
    "VEGETAB1",
    "EXEROFT1",
    "EXEROFT2",
    "EXERHMM1",
    "EXERHMM2",
    "STRENGTH",
]

# Strongest predictors of the plain logistic model, plus BMI and smoking
INTERACTION_COLUMNS = [
    "_AGEG5YR",
    "GENHLTH",
    "SEX",
    "BPHIGH4",
    "TOLDHI2",
    "CVDSTRK3",
    "CHCCOPD1",
    "CHCKIDNY",
    "DIABETE3",
    "_BMI5",
    "SMOKE100",
]

# Computed (non-integer) columns whose missing value is a single sentinel
SENTINELS = {"DROCDY3_": 900, "_DRNKWEK": 99900, "MAXVO2_": 999, "FC60_": 999}


def load_feature_names(data_path):
    """Read the feature names from the header of x_train.csv.

    Args:
        data_path: folder containing x_train.csv

    Returns:
        list of the 321 feature names, in the same order as the columns of x_train
    """
    with open(os.path.join(data_path, "x_train.csv")) as f:
        header = f.readline().strip().split(",")
    return header[1:]  # the first column is Id, which load_csv_data removes


def find_missing_codes(x, names):
    """Find, for each column, which values are survey codes rather than answers.

    BRFSS codes "don't know" and "refused" with the largest values of the answer
    range, and "none" with 8s. The code length depends on the column's range:
        max 9      -> 7 (if 6 is not a valid answer) and 9 are missing
        max 99     -> 77, 99 are missing, 88 means none (0)
        max 999    -> 777, 999 are missing, 555 and 888 mean none (0)
        max 9999   -> 7777, 9999 are missing

    Args:
        x: training data, shape=(N, D)
        names: list of the D column names

    Returns:
        dict {name: (codes to set to NaN, codes to set to 0)}
    """
    codes = {}
    for j, name in enumerate(names):
        if name in SENTINELS:
            codes[name] = ([SENTINELS[name]], [])
            continue

        values = x[:, j][~np.isnan(x[:, j])]
        if values.size == 0 or np.any(values != np.round(values)):
            continue  # empty or continuous column: no codes

        top = values.max()
        if top == 9:
            nan_codes = [9] if np.any(values == 6) else [7, 9]
            codes[name] = (nan_codes, [])
        elif top == 99:
            codes[name] = ([77, 99], [88])
        elif top == 999:
            codes[name] = ([777, 999], [555, 888])
        elif top == 9999:
            codes[name] = ([7777, 9999], [])
    return codes


def clean(x, names, codes):
    """Replace the survey codes by NaN or 0 and decode ALCDAY5.

    Args:
        x: data, shape=(N, D) (not modified)
        names: list of the D column names
        codes: output of find_missing_codes

    Returns:
        cleaned copy of x, shape=(N, D)
    """
    x = x.copy()
    for j, name in enumerate(names):
        if name not in codes:
            continue
        nan_codes, zero_codes = codes[name]
        col = x[:, j]  # view: assignments below modify x
        col[np.isin(col, nan_codes)] = np.nan
        col[np.isin(col, zero_codes)] = 0

    # ALCDAY5: 1xx = days per week, 2xx = days in the past 30 days -> days per week
    if "ALCDAY5" in names:
        col = x[:, names.index("ALCDAY5")]
        per_week = (col >= 100) & (col < 200)
        per_month = (col >= 200) & (col < 300)
        col[per_week] = col[per_week] - 100
        col[per_month] = (col[per_month] - 200) * 7 / 30
    return x


def signed_log1p(a):
    """log(1 + |a|) keeping the sign: compresses long tails, also for negative values."""
    return np.sign(a) * np.log1p(np.abs(a))


def skewness(a):
    """Skewness of each column of a: mean of ((a - mean) / std)^3, 0 for constant columns."""
    std = a.std(axis=0)
    safe_std = np.where(std > 0, std, 1.0)
    return np.where(std > 0, (((a - a.mean(axis=0)) / safe_std) ** 3).mean(axis=0), 0.0)


def encode(x, names, params):
    """Clean, flag, impute, clip, log-transform and one-hot encode, without standardizing.

    Args:
        x: data with all original columns, shape=(N, 321)
        names: list of the original column names
        params: output of fit_preprocessing

    Returns:
        x_enc: encoded data, shape=(N, F)
        feature_names: list of the F feature names
    """
    columns = params["columns"]
    x = x[:, [names.index(n) for n in columns]]
    x = clean(x, columns, params["codes"])
    missing = np.isnan(x)

    # Impute missing values with the training median (numeric) or mode (categorical)
    x[missing] = np.take(params["fill"], np.nonzero(missing)[1])

    blocks, feature_names = [], []
    for j, name in enumerate(columns):
        col = x[:, j]
        if name in params["categories"]:
            # One column per category seen in training; unseen values -> all zeros
            categories = params["categories"][name]
            blocks.append((col[:, None] == categories).astype(float))
            feature_names += [f"{name}={int(c)}" for c in categories]
            continue

        if name in params["clip"]:
            # Winsorize: limit outliers to the training percentile range
            col = np.clip(col, *params["clip"][name])
        if name in params["log"]:
            col = signed_log1p(col)
        blocks.append(col[:, None])
        feature_names.append(name)

    # Missing-value flags: answering "don't know" or skipping a question can itself
    # be informative. Taken from `missing`, i.e. before imputation.
    for name in params["flags"]:
        blocks.append(missing[:, [columns.index(name)]].astype(float))
        feature_names.append(f"{name}_missing")

    # Degree-2 features of a few strong predictors: squares and pairwise products
    # of their imputed values, so a linear model can capture e.g. risk that rises
    # faster with age when blood pressure is high
    inter = params["interactions"]
    for a in range(len(inter)):
        for b in range(a, len(inter)):
            col_a = x[:, columns.index(inter[a])]
            col_b = x[:, columns.index(inter[b])]
            blocks.append((col_a * col_b)[:, None])
            feature_names.append(f"{inter[a]}*{inter[b]}")

    return np.hstack(blocks), feature_names


def fit_preprocessing(
    x_train,
    names,
    max_missing=0.8,
    max_categories=10,
    one_hot=True,
    missing_flags=True,
    clip_pct=(1, 99),
    log_transform=True,
    skew_thresh=1.0,
    corr_thresh=0.95,
    corr_sample=50000,
    seed=0,
    interaction_columns=None,
):
    """Compute all preprocessing statistics on the training set.

    Steps:
        1. drop administrative and redundant columns
        2. find survey codes for "don't know" / "refused" / "none"
        3. drop columns with more than max_missing missing values, and constant columns
        4. split the columns into categorical (integer, 3 to max_categories values)
           and numeric; numeric columns with more than max_categories values are continuous
        5. fill values: mode for categorical columns, median for the others
        6. continuous columns only: clipping bounds and which ones to log-transform
        7. missing-value flags for the columns that have missing values
        8. squares and pairwise products of the interaction_columns
        9. mean and std of the encoded features, for standardization
        10. drop constant features, and features with |correlation| above corr_thresh
            with an earlier feature (flags and products come last, so they are
            dropped first)

    Args:
        x_train: training data, shape=(N, 321)
        names: list of the 321 column names
        max_missing: maximum share of missing values to keep a column
        max_categories: maximum number of distinct values to treat a column as categorical
        one_hot: whether to one-hot encode categorical columns
        missing_flags: whether to add a 0/1 "was missing" feature per column with missing values
        clip_pct: (low, high) percentiles to clip continuous columns to, or None for no clipping
        log_transform: whether to log-transform skewed continuous columns
        skew_thresh: continuous columns with |skewness| above this are log-transformed
        corr_thresh: correlation above which a feature is dropped, or None to keep all
        corr_sample: number of rows used to compute the correlations (for speed)
        seed: random seed for the correlation sample
        interaction_columns: list of column names to build degree-2 features from,
            or None for no interaction features

    Returns:
        params: dict used by transform
    """
    names = list(names)
    dropped = set(ADMIN_COLUMNS) | set(REDUNDANT_COLUMNS)
    columns = [n for n in names if n not in dropped]

    x = x_train[:, [names.index(n) for n in columns]]
    codes = find_missing_codes(x, columns)
    x = clean(x, columns, codes)

    params = {
        "codes": codes,
        "columns": [],
        "fill": [],
        "categories": {},
        "clip": {},
        "log": set(),
        "flags": [],
    }
    continuous = []
    for j, name in enumerate(columns):
        col = x[:, j]
        values = col[~np.isnan(col)]
        if np.isnan(col).mean() > max_missing or np.unique(values).size < 2:
            continue

        unique, counts = np.unique(values, return_counts=True)
        is_integer = np.all(unique == np.round(unique))
        is_categorical = is_integer and 3 <= unique.size <= max_categories

        params["columns"].append(name)
        if missing_flags and np.isnan(col).any():
            params["flags"].append(name)
        if is_categorical:
            params["fill"].append(unique[np.argmax(counts)])  # mode
            if one_hot:
                params["categories"][name] = unique
        else:
            params["fill"].append(np.median(values))
            if unique.size > max_categories:
                continuous.append(name)
    params["fill"] = np.array(params["fill"])

    # Clipping bounds and skewness, for continuous columns only
    for name in continuous:
        col = x[:, columns.index(name)]
        fill = params["fill"][params["columns"].index(name)]
        col = np.where(np.isnan(col), fill, col)
        if clip_pct is not None:
            params["clip"][name] = tuple(np.percentile(col, clip_pct))
            col = np.clip(col, *params["clip"][name])
        if log_transform and abs(skewness(col[:, None])[0]) > skew_thresh:
            params["log"].add(name)

    # Only columns that survived the selection above can be used for products
    params["interactions"] = [
        n for n in (interaction_columns or []) if n in params["columns"]
    ]

    x_enc, feature_names = encode(x_train, names, params)
    params["mean"] = x_enc.mean(axis=0)
    std = x_enc.std(axis=0)
    params["std"] = np.where(std > 0, std, 1.0)  # avoid dividing by 0

    # Keep non-constant features, then greedily drop near-duplicates: walk through
    # the features in order and keep one only if it is not highly correlated with
    # a feature already kept
    selected = np.nonzero(std > 1e-12)[0]
    if corr_thresh is not None:
        rng = np.random.default_rng(seed)
        rows = rng.choice(len(x_enc), min(corr_sample, len(x_enc)), replace=False)
        corr = np.abs(np.corrcoef(x_enc[rows][:, selected], rowvar=False))
        # A feature constant on the sample has no correlation
        corr = np.nan_to_num(corr)
        kept = []
        for j in range(len(selected)):
            if not kept or corr[j, kept].max() < corr_thresh:
                kept.append(j)
        selected = selected[kept]

    params["selected"] = selected
    params["feature_names"] = [feature_names[j] for j in selected]
    return params


def transform(x, names, params, add_bias=True):
    """Apply the preprocessing fitted on the training set to any data set.

    Args:
        x: data with all original columns, shape=(N, 321)
        names: list of the original column names
        params: output of fit_preprocessing
        add_bias: whether to add a first column of ones

    Returns:
        tx: standardized features, shape=(N, F) or (N, F + 1) with the bias
    """
    x_enc, _ = encode(x, list(names), params)
    selected = params["selected"]
    tx = (x_enc[:, selected] - params["mean"][selected]) / params["std"][selected]
    if add_bias:
        tx = np.c_[np.ones(tx.shape[0]), tx]
    return tx
