from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "advanced_ensemble"
MODEL_DIR = ROOT / "models" / "advanced_ensemble"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


FILES = {
    "Bayesian_Ridge":
        DATA_DIR / "bayesian_predictions.csv",

    "SMC":
        DATA_DIR / "smc_predictions.csv",

    "Chronos":
        DATA_DIR / "chronos_predictions.csv",

    "TimesFM":
        DATA_DIR / "timesfm_predictions.csv",

    "Bayesian_Deep_Learning":
        DATA_DIR / "bayesian_deep_learning_predictions.csv",
}


# ============================================================
# CONFIG
# ============================================================

EPS = 1e-6

VALIDATION_FRACTION = 0.40

RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_probability(x):
    """
    Force probabilities into a safe numerical range.
    """

    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    return np.clip(
        x.astype(float),
        EPS,
        1.0 - EPS
    )


def direction_from_return(x):
    """
    Canonical financial direction.

    Positive return:
        UP = 1

    Zero or negative:
        DOWN = 0
    """

    x = pd.to_numeric(
        x,
        errors="coerce"
    )

    return (
        x > 0
    ).astype(int)


def normalize_direction(series):
    """
    Normalize different direction formats
    into 0 / 1.
    """

    s = pd.Series(series)

    if pd.api.types.is_numeric_dtype(s):

        numeric = pd.to_numeric(
            s,
            errors="coerce"
        )

        unique = set(
            numeric.dropna().unique().tolist()
        )

        if unique.issubset({0, 1}):

            return numeric.astype("Int64")

        if unique.issubset({-1, 1}):

            return (
                numeric > 0
            ).astype("Int64")

    text = (
        s.astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "up": 1,
        "down": 0,
        "positive": 1,
        "negative": 0,
        "bullish": 1,
        "bearish": 0,
        "rise": 1,
        "fall": 0,
        "rising": 1,
        "falling": 0,
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
        "yes": 1,
        "no": 0,
    }

    return (
        text.map(mapping)
        .astype("Int64")
    )


def find_first_existing(
    df,
    candidates
):

    for col in candidates:

        if col in df.columns:

            return col

    return None


# ============================================================
# STANDARDIZE MODEL
# ============================================================

def standardize_model(
    model_name,
    path
):

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    df = pd.read_csv(path)

    print(
        f"File: {path}"
    )

    print(
        f"Rows: {len(df)}"
    )

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_col = find_first_existing(
        df,
        [
            "prediction_for_date",
            "forecast_for_date",
            "date",
            "origin_date",
        ]
    )

    if date_col is None:

        raise ValueError(
            f"{model_name}: "
            f"No usable date column found."
        )

    df["date"] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    probability_col = find_first_existing(
        df,
        [
            "probability_positive",
            "probability_up",
        ]
    )

    if probability_col is None:

        raise ValueError(
            f"{model_name}: "
            f"No probability column found."
        )

    df["probability_positive"] = (
        safe_probability(
            df[probability_col]
        )
    )

    # --------------------------------------------------------
    # PREDICTED DIRECTION
    # --------------------------------------------------------

    predicted_col = find_first_existing(
        df,
        [
            "predicted_direction",
            "predicted_direction_label",
        ]
    )

    if predicted_col is None:

        df["predicted_direction"] = (
            df["probability_positive"] >= 0.5
        ).astype(int)

    else:

        normalized_pred = normalize_direction(
            df[predicted_col]
        )

        df["predicted_direction"] = (
            normalized_pred
        )

        missing_pred = (
            df["predicted_direction"].isna()
        )

        df.loc[
            missing_pred,
            "predicted_direction"
        ] = (
            df.loc[
                missing_pred,
                "probability_positive"
            ] >= 0.5
        ).astype(int)

    df["predicted_direction"] = (
        pd.to_numeric(
            df["predicted_direction"],
            errors="coerce"
        )
        .astype(int)
    )

    # --------------------------------------------------------
    # ACTUAL RETURN
    # --------------------------------------------------------

    actual_return_col = find_first_existing(
        df,
        [
            "actual_next_day_return",
            "daily_return",
        ]
    )

    if actual_return_col is not None:

        df["actual_return"] = pd.to_numeric(
            df[actual_return_col],
            errors="coerce"
        )

    else:

        df["actual_return"] = np.nan

    # --------------------------------------------------------
    # ACTUAL DIRECTION LABEL
    # --------------------------------------------------------

    actual_direction_col = find_first_existing(
        df,
        [
            "actual_direction",
            "actual_next_day_direction",
            "actual_direction_label",
        ]
    )

    if actual_direction_col is not None:

        raw_actual = normalize_direction(
            df[actual_direction_col]
        )

    else:

        raw_actual = pd.Series(
            pd.NA,
            index=df.index,
            dtype="Int64"
        )

    # --------------------------------------------------------
    # RETURN-BASED ACTUAL DIRECTION
    # --------------------------------------------------------

    return_available = (
        df["actual_return"].notna()
    )

    return_direction = pd.Series(
        pd.NA,
        index=df.index,
        dtype="Int64"
    )

    return_direction.loc[
        return_available
    ] = (
        direction_from_return(
            df.loc[
                return_available,
                "actual_return"
            ]
        )
        .astype("Int64")
    )

    # --------------------------------------------------------
    # AUDIT LABEL VS RETURN
    # --------------------------------------------------------

    label_available = (
        raw_actual.notna()
    )

    mismatch_mask = (
        return_available
        & label_available
        & (
            raw_actual.astype("Int64")
            != return_direction.astype("Int64")
        )
    )

    mismatch_count = int(
        mismatch_mask.sum()
    )

    print(
        f"Actual return column: "
        f"{actual_return_col}"
    )

    print(
        f"Actual direction column: "
        f"{actual_direction_col}"
    )

    print(
        f"Return-vs-label mismatches: "
        f"{mismatch_count}"
    )

    # --------------------------------------------------------
    # CANONICAL ACTUAL DIRECTION
    # --------------------------------------------------------

    canonical_actual = raw_actual.copy()

    canonical_actual.loc[
        return_available
    ] = return_direction.loc[
        return_available
    ]

    missing_actual = (
        canonical_actual.isna()
    )

    if missing_actual.any():

        raise ValueError(
            f"{model_name}: "
            f"{missing_actual.sum()} rows have no "
            f"usable actual direction."
        )

    df["actual_direction"] = (
        canonical_actual.astype(int)
    )

    # --------------------------------------------------------
    # STANDARDIZED OUTPUT
    # --------------------------------------------------------

    out = df[
        [
            "date",
            "probability_positive",
            "predicted_direction",
            "actual_direction",
            "actual_return",
        ]
    ].copy()

    out = out.dropna(
        subset=[
            "date",
            "probability_positive",
            "actual_direction",
        ]
    )

    out = (
        out
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last"
        )
        .reset_index(drop=True)
    )

    print(
        f"Standardized rows: "
        f"{len(out)}"
    )

    return (
        out,
        mismatch_count,
        actual_return_col is not None,
    )


# ============================================================
# LOAD ALL MODELS
# ============================================================

def load_all_models():

    standardized = {}

    mismatch_report = {}

    return_availability = {}

    for model_name, path in FILES.items():

        if not path.exists():

            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

        (
            data,
            mismatches,
            has_return
        ) = standardize_model(
            model_name,
            path
        )

        standardized[
            model_name
        ] = data

        mismatch_report[
            model_name
        ] = mismatches

        return_availability[
            model_name
        ] = has_return

    return (
        standardized,
        mismatch_report,
        return_availability,
    )


# ============================================================
# ALIGN MODELS
# ============================================================

def align_models(models):

    print()
    print("=" * 70)
    print("ALIGNING MODELS")
    print("=" * 70)

    common_dates = None

    for name, df in models.items():

        dates = set(
            df["date"].dt.strftime(
                "%Y-%m-%d"
            )
        )

        if common_dates is None:

            common_dates = dates

        else:

            common_dates &= dates

    common_dates = sorted(
        common_dates
    )

    print(
        f"Common dates: "
        f"{len(common_dates)}"
    )

    if len(common_dates) < 100:

        raise ValueError(
            "Too few common forecast dates."
        )

    master = pd.DataFrame(
        {
            "date": pd.to_datetime(
                common_dates
            )
        }
    )

    master = (
        master
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Add model probabilities
    # --------------------------------------------------------

    for model_name, df in models.items():

        temp = df[
            [
                "date",
                "probability_positive",
                "predicted_direction",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "probability_positive":
                    f"{model_name}__probability",

                "predicted_direction":
                    f"{model_name}__prediction",
            }
        )

        master = master.merge(
            temp,
            on="date",
            how="left"
        )

    # --------------------------------------------------------
    # Reference actual information
    # --------------------------------------------------------

    reference_name = (
        list(models.keys())[0]
    )

    reference = models[
        reference_name
    ][
        [
            "date",
            "actual_direction",
            "actual_return",
        ]
    ].copy()

    master = master.merge(
        reference,
        on="date",
        how="left"
    )

    # --------------------------------------------------------
    # Compare model actual labels
    # --------------------------------------------------------

    consistency_issues = []

    for model_name, df in models.items():

        temp = df[
            [
                "date",
                "actual_direction",
            ]
        ].copy()

        temp = temp.rename(
            columns={
                "actual_direction":
                    f"{model_name}__actual"
            }
        )

        master = master.merge(
            temp,
            on="date",
            how="left"
        )

        mismatch = (
            master[
                f"{model_name}__actual"
            ].astype(int)
            != master[
                "actual_direction"
            ].astype(int)
        )

        count = int(
            mismatch.sum()
        )

        consistency_issues.append(
            (
                model_name,
                count
            )
        )

    print()
    print(
        "Actual-direction consistency:"
    )

    for model_name, count in (
        consistency_issues
    ):

        print(
            f"  {model_name}: "
            f"{count} mismatches"
        )

    # --------------------------------------------------------
    # Canonical actual direction
    # --------------------------------------------------------

    master[
        "canonical_actual_direction"
    ] = direction_from_return(
        master["actual_return"]
    )

    return (
        master,
        common_dates,
        consistency_issues,
    )


# ============================================================
# METRICS
# ============================================================

def evaluate_probabilities(
    y_true,
    probabilities
):

    probabilities = safe_probability(
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = {}

    result["accuracy"] = (
        accuracy_score(
            y_true,
            predictions
        )
    )

    result["brier_score"] = (
        brier_score_loss(
            y_true,
            probabilities
        )
    )

    result["log_loss"] = (
        log_loss(
            y_true,
            probabilities,
            labels=[0, 1]
        )
    )

    try:

        result["roc_auc"] = (
            roc_auc_score(
                y_true,
                probabilities
            )
        )

    except ValueError:

        result["roc_auc"] = np.nan

    return result


# ============================================================
# WEIGHT HELPERS
# ============================================================

def normalize_weights(weights):

    weights = np.asarray(
        weights,
        dtype=float
    )

    weights = np.maximum(
        weights,
        0
    )

    total = weights.sum()

    if total <= 0:

        return (
            np.ones_like(weights)
            / len(weights)
        )

    return weights / total


def weighted_probability(
    probability_matrix,
    weights
):

    weights = normalize_weights(
        weights
    )

    probability_matrix = np.asarray(
        probability_matrix,
        dtype=float
    )

    return np.clip(
        probability_matrix @ weights,
        EPS,
        1 - EPS
    )


# ============================================================
# CONSTRAINED PROBABILITY STACKING
# ============================================================

def optimize_log_score(
    probability_matrix,
    y_true
):

    n_models = (
        probability_matrix.shape[1]
    )

    initial = (
        np.ones(n_models)
        / n_models
    )

    def objective(weights):

        weights = normalize_weights(
            weights
        )

        probability = (
            weighted_probability(
                probability_matrix,
                weights
            )
        )

        return log_loss(
            y_true,
            probability,
            labels=[0, 1]
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[
            (0.0, 1.0)
            for _ in range(n_models)
        ],
        constraints={
            "type": "eq",
            "fun":
                lambda w:
                np.sum(w) - 1
        },
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
        }
    )

    if not result.success:

        print(
            "WARNING: Probability stacking "
            "optimization did not fully converge."
        )

    return normalize_weights(
        result.x
    )


# ============================================================
# BMA-STYLE WEIGHTS
# ============================================================

def calculate_bma_style_weights(
    probability_matrix,
    y_true
):

    """
    BMA-style predictive weighting.

    This is NOT exact Bayesian model posterior
    probability.

    It uses validation predictive log score
    softmax because common pointwise posterior
    log-likelihood draws are unavailable for
    every model.
    """

    log_scores = []

    for i in range(
        probability_matrix.shape[1]
    ):

        probability = safe_probability(
            probability_matrix[:, i]
        )

        score = -log_loss(
            y_true,
            probability,
            labels=[0, 1]
        )

        log_scores.append(
            score
        )

    log_scores = np.asarray(
        log_scores
    )

    shifted = (
        log_scores
        - np.max(log_scores)
    )

    exp_scores = np.exp(
        shifted
    )

    weights = (
        exp_scores
        / exp_scores.sum()
    )

    return (
        weights,
        log_scores
    )


# ============================================================
# LOGIT STACKING
# ============================================================

def optimize_logit_stacking(
    probability_matrix,
    y_true
):

    probability_matrix = np.clip(
        probability_matrix,
        EPS,
        1 - EPS
    )

    logits = logit(
        probability_matrix
    )

    n_models = (
        logits.shape[1]
    )

    initial = (
        np.ones(n_models)
        / n_models
    )

    def objective(weights):

        weights = normalize_weights(
            weights
        )

        combined_logit = (
            logits @ weights
        )

        probability = np.clip(
            expit(
                combined_logit
            ),
            EPS,
            1 - EPS
        )

        return log_loss(
            y_true,
            probability,
            labels=[0, 1]
        )

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[
            (0.0, 1.0)
            for _ in range(n_models)
        ],
        constraints={
            "type": "eq",
            "fun":
                lambda w:
                np.sum(w) - 1
        },
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
        }
    )

    if not result.success:

        print(
            "WARNING: Logit stacking "
            "optimization did not fully converge."
        )

    weights = normalize_weights(
        result.x
    )

    combined_logit = (
        logits @ weights
    )

    probabilities = np.clip(
        expit(
            combined_logit
        ),
        EPS,
        1 - EPS
    )

    return (
        weights,
        probabilities
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("STEP 25 - ADVANCED BAYESIAN ENSEMBLE")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    (
        models,
        mismatch_report,
        return_availability
    ) = load_all_models()

    # --------------------------------------------------------
    # ALIGN
    # --------------------------------------------------------

    (
        master,
        common_dates,
        consistency
    ) = align_models(
        models
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We only use return-derived canonical direction when
    # actual return exists for the reference model.
    #
    # Reference model is Bayesian Ridge and contains
    # actual_next_day_return.
    # --------------------------------------------------------

    if master["actual_return"].isna().any():

        raise ValueError(
            "Reference actual return contains missing values."
        )

    master[
        "actual_direction"
    ] = (
        master[
            "canonical_actual_direction"
        ].astype(int)
    )

    print()
    print(
        "Canonical actual direction "
        "derived from reference actual return."
    )

    # --------------------------------------------------------
    # MODEL NAMES
    # --------------------------------------------------------

    model_names = list(
        models.keys()
    )

    probability_columns = [
        f"{name}__probability"
        for name in model_names
    ]

    # --------------------------------------------------------
    # PROBABILITY MATRIX
    # --------------------------------------------------------

    probability_matrix = (
        master[
            probability_columns
        ]
        .astype(float)
        .values
    )

    y = (
        master[
            "actual_direction"
        ]
        .astype(int)
        .values
    )

    # --------------------------------------------------------
    # NaN CHECK
    # --------------------------------------------------------

    if np.isnan(
        probability_matrix
    ).any():

        raise ValueError(
            "Probability matrix contains NaN."
        )

    # --------------------------------------------------------
    # CHRONOLOGICAL SPLIT
    # --------------------------------------------------------

    n = len(master)

    validation_start = int(
        n * (
            1.0
            - VALIDATION_FRACTION
        )
    )

    train_idx = np.arange(
        0,
        validation_start
    )

    test_idx = np.arange(
        validation_start,
        n
    )

    X_val = probability_matrix[
        train_idx
    ]

    y_val = y[
        train_idx
    ]

    X_test = probability_matrix[
        test_idx
    ]

    y_test = y[
        test_idx
    ]

    print()
    print(
        f"Total aligned observations: {n}"
    )

    print(
        f"Validation observations: "
        f"{len(train_idx)}"
    )

    print(
        f"Test observations: "
        f"{len(test_idx)}"
    )

    print(
        "Split type: chronological"
    )

    # --------------------------------------------------------
    # EQUAL WEIGHTS
    # --------------------------------------------------------

    equal_weights = (
        np.ones(
            len(model_names)
        )
        / len(model_names)
    )

    # --------------------------------------------------------
    # BMA-STYLE
    # --------------------------------------------------------

    (
        bma_weights,
        validation_log_scores
    ) = calculate_bma_style_weights(
        X_val,
        y_val
    )

    # --------------------------------------------------------
    # PROBABILITY STACKING
    # --------------------------------------------------------

    stacking_weights = (
        optimize_log_score(
            X_val,
            y_val
        )
    )

    # --------------------------------------------------------
    # LOGIT STACKING
    # --------------------------------------------------------

    (
        logit_stacking_weights,
        _
    ) = optimize_logit_stacking(
        X_val,
        y_val
    )

    # --------------------------------------------------------
    # TEST PROBABILITIES
    # --------------------------------------------------------

    equal_probability = (
        weighted_probability(
            X_test,
            equal_weights
        )
    )

    bma_probability = (
        weighted_probability(
            X_test,
            bma_weights
        )
    )

    stacking_probability = (
        weighted_probability(
            X_test,
            stacking_weights
        )
    )

    logit_probability = np.clip(
        expit(
            logit(
                np.clip(
                    X_test,
                    EPS,
                    1 - EPS
                )
            )
            @ logit_stacking_weights
        ),
        EPS,
        1 - EPS
    )

    # --------------------------------------------------------
    # INDIVIDUAL MODEL METRICS
    # --------------------------------------------------------

    metric_rows = []

    for i, model_name in enumerate(
        model_names
    ):

        metrics = (
            evaluate_probabilities(
                y_test,
                X_test[:, i]
            )
        )

        metric_rows.append(
            {
                "model":
                    model_name,

                "ensemble_type":
                    "individual",

                **metrics,
            }
        )

    # --------------------------------------------------------
    # ENSEMBLES
    # --------------------------------------------------------

    ensemble_predictions = {

        "Equal_Weight_Ensemble":
            equal_probability,

        "BMA_Style_Ensemble":
            bma_probability,

        "Constrained_Stacking":
            stacking_probability,

        "Logit_Constrained_Stacking":
            logit_probability,
    }

    for (
        name,
        probability
    ) in ensemble_predictions.items():

        metrics = (
            evaluate_probabilities(
                y_test,
                probability
            )
        )

        metric_rows.append(
            {
                "model":
                    name,

                "ensemble_type":
                    "ensemble",

                **metrics,
            }
        )

    metrics_df = pd.DataFrame(
        metric_rows
    )

    # --------------------------------------------------------
    # TEST DATA
    # --------------------------------------------------------

    test_master = master.iloc[
        test_idx
    ].copy()

    test_master[
        "equal_weight_probability"
    ] = equal_probability

    test_master[
        "bma_style_probability"
    ] = bma_probability

    test_master[
        "constrained_stacking_probability"
    ] = stacking_probability

    test_master[
        "logit_stacking_probability"
    ] = logit_probability

    test_master[
        "equal_weight_prediction"
    ] = (
        equal_probability >= 0.5
    ).astype(int)

    test_master[
        "bma_style_prediction"
    ] = (
        bma_probability >= 0.5
    ).astype(int)

    test_master[
        "constrained_stacking_prediction"
    ] = (
        stacking_probability >= 0.5
    ).astype(int)

    test_master[
        "logit_stacking_prediction"
    ] = (
        logit_probability >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # WEIGHTS REPORT
    # --------------------------------------------------------

    weight_rows = []

    for i, model_name in enumerate(
        model_names
    ):

        weight_rows.append(
            {
                "model":
                    model_name,

                "equal_weight":
                    equal_weights[i],

                "bma_style_weight":
                    bma_weights[i],

                "probability_stacking_weight":
                    stacking_weights[i],

                "logit_stacking_weight":
                    logit_stacking_weights[i],

                "validation_log_predictive_score":
                    validation_log_scores[i],
            }
        )

    weights_df = pd.DataFrame(
        weight_rows
    )

    # --------------------------------------------------------
    # PROBABILITY COMPARISON
    # --------------------------------------------------------

    probability_comparison = (
        test_master[
            [
                "date",
                "actual_direction",
                *probability_columns,
                "equal_weight_probability",
                "bma_style_probability",
                "constrained_stacking_probability",
                "logit_stacking_probability",
            ]
        ].copy()
    )

    # --------------------------------------------------------
    # STACKING COEFFICIENTS
    # --------------------------------------------------------

    stacking_coefficients = (
        pd.DataFrame(
            {
                "model":
                    model_names,

                "probability_stacking_weight":
                    stacking_weights,

                "logit_stacking_weight":
                    logit_stacking_weights,
            }
        )
    )

    # --------------------------------------------------------
    # RECONCILIATION REPORT
    # --------------------------------------------------------

    mismatch_rows = []

    for (
        model_name,
        mismatch_count
    ) in mismatch_report.items():

        mismatch_rows.append(
            {
                "model":
                    model_name,

                "actual_return_vs_label_mismatches":
                    mismatch_count,

                "has_actual_return":
                    return_availability[
                        model_name
                    ],
            }
        )

    mismatch_df = pd.DataFrame(
        mismatch_rows
    )

    consistency_df = pd.DataFrame(
        consistency,
        columns=[
            "model",
            "canonical_actual_direction_mismatches",
        ]
    )

    reconciliation_df = (
        mismatch_df.merge(
            consistency_df,
            on="model",
            how="left"
        )
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics_path = (
        REPORT_DIR
        / "metrics.csv"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE WEIGHTS
    # --------------------------------------------------------

    weights_path = (
        REPORT_DIR
        / "weights.csv"
    )

    weights_df.to_csv(
        weights_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE PROBABILITY COMPARISON
    # --------------------------------------------------------

    probability_path = (
        REPORT_DIR
        / "probability_comparison.csv"
    )

    probability_comparison.to_csv(
        probability_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE STACKING COEFFICIENTS
    # --------------------------------------------------------

    stacking_path = (
        REPORT_DIR
        / "stacking_coefficients.csv"
    )

    stacking_coefficients.to_csv(
        stacking_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE RECONCILIATION
    # --------------------------------------------------------

    reconciliation_path = (
        REPORT_DIR
        / "actual_direction_reconciliation.csv"
    )

    reconciliation_df.to_csv(
        reconciliation_path,
        index=False
    )

    # --------------------------------------------------------
    # SAVE FINAL PREDICTIONS
    # --------------------------------------------------------

    predictions_path = (
        DATA_DIR
        / "advanced_ensemble_predictions.csv"
    )

    save_columns = [
        "date",
        "actual_return",
        "actual_direction",
        *probability_columns,

        "equal_weight_probability",
        "equal_weight_prediction",

        "bma_style_probability",
        "bma_style_prediction",

        "constrained_stacking_probability",
        "constrained_stacking_prediction",

        "logit_stacking_probability",
        "logit_stacking_prediction",
    ]

    test_master[
        save_columns
    ].to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_path = (
        REPORT_DIR
        / "summary.txt"
    )

    best_row = metrics_df.loc[
        metrics_df[
            "brier_score"
        ].idxmin()
    ]

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 25 - ADVANCED ENSEMBLE SUMMARY\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Total aligned observations: {n}\n"
        )

        f.write(
            f"Validation observations: "
            f"{len(train_idx)}\n"
        )

        f.write(
            f"Test observations: "
            f"{len(test_idx)}\n"
        )

        f.write(
            "Validation split: chronological\n\n"
        )

        f.write(
            "Models:\n"
        )

        for model_name in model_names:

            f.write(
                f"  - {model_name}\n"
            )

        f.write(
            "\nActual direction policy:\n"
        )

        f.write(
            "Reference actual direction is derived "
            "from actual realized return.\n"
        )

        f.write(
            "Positive return = UP (1).\n"
        )

        f.write(
            "Zero or negative return = DOWN (0).\n\n"
        )

        f.write(
            "BMA-style weighting:\n"
        )

        f.write(
            "Validation predictive log score "
            "softmax.\n"
        )

        f.write(
            "This is not exact Bayesian model "
            "posterior probability.\n\n"
        )

        f.write(
            "Exact WAIC/LOO:\n"
        )

        f.write(
            "Not computed because common pointwise "
            "posterior log-likelihood draws are not "
            "available for all ensemble members.\n\n"
        )

        f.write(
            "TEST METRICS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            metrics_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nBEST MODEL BY BRIER SCORE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            best_row.to_string()
        )

        f.write(
            "\n\nACTUAL DIRECTION RECONCILIATION\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            reconciliation_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    config = {

        "step": 25,

        "project":
            "Bayesian Regime Detection Engine",

        "models":
            model_names,

        "validation_fraction":
            VALIDATION_FRACTION,

        "random_state":
            RANDOM_STATE,

        "probability_floor":
            EPS,

        "bma_style": {

            "method":
                "validation log predictive "
                "score softmax",

            "exact_bayesian_model_probability":
                False,
        },

        "stacking": {

            "probability_space":
                True,

            "logit_space":
                True,

            "nonnegative_weights":
                True,

            "weights_sum_to_one":
                True,
        },

        "actual_direction": {

            "reference":
                "Bayesian Ridge actual return",

            "positive_return":
                1,

            "zero_or_negative_return":
                0,
        },
    }

    config_path = (
        MODEL_DIR
        / "ensemble_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_results = {}

    validation_results[
        "metrics_exists"
    ] = metrics_path.exists()

    validation_results[
        "weights_exists"
    ] = weights_path.exists()

    validation_results[
        "predictions_exists"
    ] = predictions_path.exists()

    validation_results[
        "summary_exists"
    ] = summary_path.exists()

    validation_results[
        "config_exists"
    ] = config_path.exists()

    validation_results[
        "common_dates_positive"
    ] = (
        len(common_dates) > 0
    )

    validation_probability_columns = [
        *probability_columns,

        "equal_weight_probability",

        "bma_style_probability",

        "constrained_stacking_probability",

        "logit_stacking_probability",
    ]

    validation_probability_array = (
        probability_comparison[
            validation_probability_columns
        ].values
    )

    validation_results[
        "probabilities_valid"
    ] = bool(
        np.all(
            validation_probability_array
            >= 0
        )
        and
        np.all(
            validation_probability_array
            <= 1
        )
        and
        np.isfinite(
            validation_probability_array
        ).all()
    )

    validation_results[
        "chronological_order"
    ] = bool(
        master[
            "date"
        ].is_monotonic_increasing
    )

    validation_results[
        "duplicate_dates"
    ] = bool(
        master[
            "date"
        ].duplicated().sum()
        == 0
    )

    validation_results[
        "weights_sum_to_one"
    ] = bool(
        np.isclose(
            bma_weights.sum(),
            1.0
        )
        and
        np.isclose(
            stacking_weights.sum(),
            1.0
        )
        and
        np.isclose(
            logit_stacking_weights.sum(),
            1.0
        )
    )

    validation_results[
        "weights_nonnegative"
    ] = bool(
        np.all(
            bma_weights >= 0
        )
        and
        np.all(
            stacking_weights >= 0
        )
        and
        np.all(
            logit_stacking_weights >= 0
        )
    )

    validation_results[
        "no_test_training_overlap"
    ] = bool(
        set(
            master.iloc[
                train_idx
            ]["date"]
        ).isdisjoint(
            set(
                master.iloc[
                    test_idx
                ]["date"]
            )
        )
    )

    validation_results[
        "reference_actual_return_complete"
    ] = bool(
        master[
            "actual_return"
        ].notna().all()
    )

    validation_results[
        "canonical_direction_valid"
    ] = bool(
        set(
            master[
                "actual_direction"
            ].unique()
        ).issubset(
            {0, 1}
        )
    )

    validation_results[
        "test_predictions_valid"
    ] = bool(
        set(
            test_master[
                "equal_weight_prediction"
            ].unique()
        ).issubset(
            {0, 1}
        )
        and
        set(
            test_master[
                "bma_style_prediction"
            ].unique()
        ).issubset(
            {0, 1}
        )
        and
        set(
            test_master[
                "constrained_stacking_prediction"
            ].unique()
        ).issubset(
            {0, 1}
        )
        and
        set(
            test_master[
                "logit_stacking_prediction"
            ].unique()
        ).issubset(
            {0, 1}
        )
    )

    validation_results[
        "output_files_complete"
    ] = all(
        [
            metrics_path.exists(),
            weights_path.exists(),
            probability_path.exists(),
            stacking_path.exists(),
            reconciliation_path.exists(),
            predictions_path.exists(),
            summary_path.exists(),
            config_path.exists(),
        ]
    )

    all_pass = all(
        validation_results.values()
    )

    print()
    print("=" * 70)
    print("STEP 25 VALIDATION")
    print("=" * 70)

    for key, value in (
        validation_results.items()
    ):

        print(
            f"{key}: {value}"
        )

    if all_pass:

        print()
        print(
            "STEP 25 VALIDATION: PASSED"
        )

    else:

        print()
        print(
            "STEP 25 VALIDATION: FAILED"
        )

        raise RuntimeError(
            "Step 25 validation failed."
        )

    print()
    print("=" * 70)
    print("STEP 25 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        "Generated files:"
    )

    print(
        predictions_path
    )

    print(
        metrics_path
    )

    print(
        weights_path
    )

    print(
        probability_path
    )

    print(
        stacking_path
    )

    print(
        reconciliation_path
    )

    print(
        summary_path
    )

    print(
        config_path
    )


if __name__ == "__main__":
    main()