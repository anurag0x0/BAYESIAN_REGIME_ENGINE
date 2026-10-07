# ============================================================
# PROJECT 1A
# BAYESIAN REGIME DETECTION ENGINE
#
# STEP 23
# TimesFM Foundation Model Benchmark
# ============================================================
#
# Model:
#   Google TimesFM 2.5 - 200M PyTorch
#
# Objective:
#   Forecast next-day NIFTY 50 return using a pretrained
#   time-series foundation model and convert the forecast
#   into probabilistic direction estimates.
#
# Method:
#   - Chronological train/test split
#   - No future information in context
#   - Rolling one-step-ahead forecasting
#   - TimesFM point forecast
#   - TimesFM quantile forecast
#   - Probability of positive return
#   - Predictive uncertainty
#   - Direction classification
#   - Brier score
#   - Accuracy
#   - ROC-AUC
#   - Log loss
#
# ============================================================

from pathlib import Path
import json
import math
import time
import warnings

import numpy as np
import pandas as pd

import torch

import timesfm

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "nifty50_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "timesfm"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "timesfm"
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "timesfm_predictions.csv"
)

METRICS_FILE = (
    REPORT_DIR
    / "metrics.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "summary.txt"
)

CONFIG_FILE = (
    MODEL_DIR
    / "timesfm_config.json"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.80

# TimesFM context length.
#
# 512 is deliberately selected for CPU execution.
# TimesFM 2.5 supports substantially longer context.
MAX_CONTEXT = 512

# One-step-ahead forecast
FORECAST_HORIZON = 1

# Process test forecasts in batches.
#
# Each example can have a different context length.
# TimesFM accepts a list of input series.
BATCH_SIZE = 16

# Current official checkpoint.
MODEL_ID = (
    "google/timesfm-2.5-200m-pytorch"
)

# Quantile configuration
USE_CONTINUOUS_QUANTILE_HEAD = True

# TimesFM 2.5 returns:
#
# index 0 = mean
# index 1 = q10
# index 2 = q20
# index 3 = q30
# index 4 = q40
# index 5 = q50
# index 6 = q60
# index 7 = q70
# index 8 = q80
# index 9 = q90
#
# This mapping follows the current TimesFM 2.5 API.
# ============================================================


# ============================================================
# 3. DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# 4. CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. RANDOM SEED
# ============================================================

def set_seed():

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


# ============================================================
# 6. LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)

    print(
        "LOADING NIFTY 50 FEATURE DATA"
    )

    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input file: {INPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_candidates = [
        "date",
        "Date",
        "DATE",
    ]

    date_column = None

    for col in date_candidates:

        if col in df.columns:

            date_column = col

            break

    if date_column is None:

        raise ValueError(
            "Date column not found."
        )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return_candidates = [
        "daily_return",
        "log_return",
        "return",
    ]

    return_column = None

    for col in return_candidates:

        if col in df.columns:

            return_column = col

            break

    if return_column is None:

        raise ValueError(
            "Return column not found."
        )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_candidates = [
        "target_next_day_direction",
        "next_day_direction",
        "target_direction",
    ]

    target_column = None

    for col in target_candidates:

        if col in df.columns:

            target_column = col

            break

    if target_column is None:

        raise ValueError(
            "Next-day target column not found."
        )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df[return_column] = pd.to_numeric(
        df[return_column],
        errors="coerce"
    )

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    df = (
        df
        .dropna(
            subset=[
                date_column,
                return_column,
                target_column,
            ]
        )
        .sort_values(
            date_column
        )
        .reset_index(
            drop=True
        )
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=[
            return_column,
            target_column,
        ]
    ).reset_index(
        drop=True
    )

    print(
        f"Date column: {date_column}"
    )

    print(
        f"Return column: {return_column}"
    )

    print(
        f"Target column: {target_column}"
    )

    print(
        f"Usable observations: {len(df)}"
    )

    return (
        df,
        date_column,
        return_column,
        target_column,
    )


# ============================================================
# 7. CHRONOLOGICAL SPLIT
# ============================================================

def chronological_split(
    df,
    date_column,
    return_column,
    target_column,
):

    print("\n" + "=" * 70)

    print(
        "CREATING CHRONOLOGICAL TRAIN / TEST SPLIT"
    )

    print("=" * 70)

    split_index = int(
        len(df)
        * TRAIN_RATIO
    )

    train = df.iloc[
        :split_index
    ].copy()

    test = df.iloc[
        split_index:
    ].copy()

    print(
        f"Total observations : {len(df)}"
    )

    print(
        f"Training           : {len(train)}"
    )

    print(
        f"Testing            : {len(test)}"
    )

    print(
        f"Train period: "
        f"{train[date_column].min().date()} -> "
        f"{train[date_column].max().date()}"
    )

    print(
        f"Test period : "
        f"{test[date_column].min().date()} -> "
        f"{test[date_column].max().date()}"
    )

    return (
        train,
        test,
    )


# ============================================================
# 8. BUILD TIMESFM MODEL
# ============================================================

def load_timesfm():

    print("\n" + "=" * 70)

    print(
        "LOADING TIMESFM 2.5 FOUNDATION MODEL"
    )

    print("=" * 70)

    print(
        f"Model: {MODEL_ID}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        "\nFirst execution may download the pretrained "
        "model weights."
    )

    # --------------------------------------------------------
    # PyTorch precision
    # --------------------------------------------------------

    try:

        torch.set_float32_matmul_precision(
            "high"
        )

    except Exception:

        pass

    # --------------------------------------------------------
    # Load pretrained model
    # --------------------------------------------------------

    model = (
        timesfm
        .TimesFM_2p5_200M_torch
        .from_pretrained(
            MODEL_ID
        )
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        timesfm.ForecastConfig(
            max_context=MAX_CONTEXT,
            max_horizon=FORECAST_HORIZON,
            normalize_inputs=True,
            use_continuous_quantile_head=(
                USE_CONTINUOUS_QUANTILE_HEAD
            ),
            force_flip_invariance=True,
            infer_is_positive=True,
            fix_quantile_crossing=True,
        )
    )

    print(
        "\nTimesFM loaded and compiled successfully."
    )

    return model


# ============================================================
# 9. CREATE FORECAST CONTEXT
# ============================================================

def create_context(
    values,
    end_position,
):
    """
    Returns observations available BEFORE the target
    forecast.

    If the target is at index i, the context ends at i-1.

    This prevents future leakage.
    """

    start_position = max(
        0,
        end_position
        - MAX_CONTEXT
    )

    context = values[
        start_position:end_position
    ]

    context = np.asarray(
        context,
        dtype=np.float32
    )

    context = context[
        np.isfinite(context)
    ]

    return context


# ============================================================
# 10. NORMAL CDF
# ============================================================

def normal_cdf(
    x,
    mean,
    std,
):
    """
    Standard normal approximation for probability
    that future return is <= x.
    """

    std = max(
        float(std),
        1e-8
    )

    z = (
        x - mean
    ) / (
        std
        * math.sqrt(2.0)
    )

    return (
        0.5
        * (
            1.0
            + math.erf(z)
        )
    )


# ============================================================
# 11. QUANTILE TO PROBABILITY
# ============================================================

def estimate_probability_positive(
    quantiles,
):
    """
    Convert TimesFM quantile forecasts into an estimated
    probability that the next return is positive.

    TimesFM 2.5 quantile output:

        [mean, q10, q20, q30, q40,
         q50, q60, q70, q80, q90]

    We estimate sigma from q10 and q90:

        sigma ≈ (q90 - q10) / (2 * 1.28155)

    Then:

        P(R > 0)
          = 1 - Phi((0 - mean) / sigma)

    This is an approximation used to obtain a directional
    probability from the continuous forecast distribution.
    """

    mean = float(
        quantiles[0]
    )

    q10 = float(
        quantiles[1]
    )

    q90 = float(
        quantiles[9]
    )

    # --------------------------------------------------------
    # Estimate standard deviation
    # --------------------------------------------------------

    std = (
        q90 - q10
    ) / (
        2.0
        * 1.2815515655446004
    )

    std = max(
        abs(std),
        1e-8
    )

    # --------------------------------------------------------
    # P(return > 0)
    # --------------------------------------------------------

    probability_positive = (
        1.0
        - normal_cdf(
            0.0,
            mean,
            std,
        )
    )

    probability_positive = float(
        np.clip(
            probability_positive,
            0.0,
            1.0
        )
    )

    return (
        mean,
        std,
        probability_positive,
    )


# ============================================================
# 12. WALK-FORWARD TIMESFM FORECAST
# ============================================================

def generate_forecasts(
    model,
    full_df,
    test_df,
    date_column,
    return_column,
    target_column,
):

    print("\n" + "=" * 70)

    print(
        "RUNNING WALK-FORWARD TIMESFM FORECAST"
    )

    print("=" * 70)

    print(
        f"Context length: {MAX_CONTEXT}"
    )

    print(
        f"Horizon: {FORECAST_HORIZON}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    # --------------------------------------------------------
    # Full return series
    # --------------------------------------------------------

    returns = (
        full_df[
            return_column
        ]
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # Find test start
    # --------------------------------------------------------

    test_start_date = (
        test_df[
            date_column
        ].iloc[0]
    )

    test_start_index = (
        full_df[
            date_column
        ]
        .searchsorted(
            test_start_date
        )
    )

    # --------------------------------------------------------
    # Prepare origins
    # --------------------------------------------------------

    origins = []

    contexts = []

    rows = []

    for i in range(
        test_start_index,
        len(full_df),
    ):

        # ----------------------------------------------------
        # We need the next day as actual target.
        # Therefore stop one row before the end.
        # ----------------------------------------------------

        if (
            i + 1
            >= len(full_df)
        ):

            break

        context = create_context(
            returns,
            i + 1,
        )

        if len(context) < 20:

            continue

        # ----------------------------------------------------
        # Current origin i
        #
        # Context includes information through current
        # observation.
        #
        # Forecast is for i+1.
        # ----------------------------------------------------

        origins.append(i)

        contexts.append(
            context
        )

        rows.append(
            {
                "origin_date":
                    full_df.loc[
                        i,
                        date_column
                    ],

                "forecast_for_date":
                    full_df.loc[
                        i + 1,
                        date_column
                    ],

                "actual_next_day_return":
                    float(
                        full_df.loc[
                            i + 1,
                            return_column
                        ]
                    ),

                "actual_next_day_direction":
                    int(
                        full_df.loc[
                            i + 1,
                            target_column
                        ]
                    ),
            }
        )

    total_forecasts = len(
        contexts
    )

    print(
        f"Forecasts to generate: "
        f"{total_forecasts}"
    )

    if total_forecasts == 0:

        raise ValueError(
            "No valid TimesFM forecast contexts."
        )

    forecast_records = []

    start_time = time.time()

    # --------------------------------------------------------
    # Batched forecasting
    # --------------------------------------------------------

    for batch_start in range(
        0,
        total_forecasts,
        BATCH_SIZE,
    ):

        batch_end = min(
            batch_start
            + BATCH_SIZE,
            total_forecasts,
        )

        batch_contexts = (
            contexts[
                batch_start:batch_end
            ]
        )

        batch_rows = (
            rows[
                batch_start:batch_end
            ]
        )

        # ----------------------------------------------------
        # TimesFM forecast
        # ----------------------------------------------------

        point_forecast, quantile_forecast = (
            model.forecast(
                horizon=FORECAST_HORIZON,
                inputs=batch_contexts,
            )
        )

        point_forecast = np.asarray(
            point_forecast
        )

        quantile_forecast = np.asarray(
            quantile_forecast
        )

        # ----------------------------------------------------
        # Validate shapes
        # ----------------------------------------------------

        if point_forecast.shape[0] != len(
            batch_rows
        ):

            raise RuntimeError(
                "TimesFM point forecast batch "
                "size does not match input batch."
            )

        if quantile_forecast.shape[0] != len(
            batch_rows
        ):

            raise RuntimeError(
                "TimesFM quantile forecast batch "
                "size does not match input batch."
            )

        # ----------------------------------------------------
        # Process each forecast
        # ----------------------------------------------------

        for j, row in enumerate(
            batch_rows
        ):

            point_value = float(
                point_forecast[
                    j,
                    0
                ]
            )

            quantiles = (
                quantile_forecast[
                    j,
                    0,
                    :
                ]
            )

            (
                distribution_mean,
                predictive_std,
                probability_positive,
            ) = estimate_probability_positive(
                quantiles
            )

            # ------------------------------------------------
            # Use TimesFM point forecast for direction
            # ------------------------------------------------

            predicted_direction = int(
                point_value >= 0
            )

            confidence = max(
                probability_positive,
                1
                - probability_positive
            )

            p = np.clip(
                probability_positive,
                1e-8,
                1 - 1e-8
            )

            entropy = -(
                p * np.log(p)
                +
                (1 - p)
                * np.log(1 - p)
            )

            forecast_records.append(
                {
                    "origin_date":
                        row[
                            "origin_date"
                        ],

                    "forecast_for_date":
                        row[
                            "forecast_for_date"
                        ],

                    "predicted_return":
                        point_value,

                    "distribution_mean":
                        distribution_mean,

                    "predictive_std":
                        predictive_std,

                    "probability_positive":
                        probability_positive,

                    "predicted_direction":
                        predicted_direction,

                    "actual_next_day_return":
                        row[
                            "actual_next_day_return"
                        ],

                    "actual_next_day_direction":
                        row[
                            "actual_next_day_direction"
                        ],

                    "confidence":
                        confidence,

                    "predictive_entropy":
                        entropy,

                    "q10":
                        float(
                            quantiles[1]
                        ),

                    "q50":
                        float(
                            quantiles[5]
                        ),

                    "q90":
                        float(
                            quantiles[9]
                        ),
                }
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        completed = (
            batch_end
        )

        elapsed = (
            time.time()
            - start_time
        )

        rate = (
            completed
            / elapsed
            if elapsed > 0
            else 0
        )

        remaining = (
            total_forecasts
            - completed
        )

        eta = (
            remaining
            / rate
            if rate > 0
            else 0
        )

        print(
            f"Progress: "
            f"{completed}/{total_forecasts} "
            f"({completed / total_forecasts * 100:.1f}%) "
            f"| Rate: {rate:.2f}/sec "
            f"| ETA: {eta/60:.1f} min"
        )

    result = pd.DataFrame(
        forecast_records
    )

    result = result.sort_values(
        "forecast_for_date"
    ).reset_index(
        drop=True
    )

    return result


# ============================================================
# 13. ADD DERIVED COLUMNS
# ============================================================

def add_prediction_columns(
    result
):

    # --------------------------------------------------------
    # Direction labels
    # --------------------------------------------------------

    result[
        "predicted_direction_label"
    ] = np.where(
        result[
            "predicted_direction"
        ] == 1,
        "UP",
        "DOWN"
    )

    result[
        "actual_direction_label"
    ] = np.where(
        result[
            "actual_next_day_direction"
        ] == 1,
        "UP",
        "DOWN"
    )

    # --------------------------------------------------------
    # Correctness
    # --------------------------------------------------------

    result[
        "prediction_correct"
    ] = (
        result[
            "predicted_direction"
        ]
        ==
        result[
            "actual_next_day_direction"
        ]
    ).astype(int)

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    result[
        "return_forecast_error"
    ] = (
        result[
            "actual_next_day_return"
        ]
        -
        result[
            "predicted_return"
        ]
    )

    # --------------------------------------------------------
    # Uncertainty state
    # --------------------------------------------------------

    result[
        "uncertainty_state"
    ] = np.select(
        [
            result[
                "predictive_std"
            ]
            <
            result[
                "predictive_std"
            ].quantile(0.33),

            result[
                "predictive_std"
            ]
            <
            result[
                "predictive_std"
            ].quantile(0.67),
        ],
        [
            "Low",
            "Medium",
        ],
        default="High",
    )

    # --------------------------------------------------------
    # Uncertainty percentile
    # --------------------------------------------------------

    result[
        "uncertainty_percentile"
    ] = (
        result[
            "predictive_std"
        ]
        .rank(
            pct=True
        )
        * 100
    )

    return result


# ============================================================
# 14. EVALUATE
# ============================================================

def evaluate_model(
    result
):

    print("\n" + "=" * 70)

    print(
        "EVALUATING TIMESFM"
    )

    print("=" * 70)

    y_true = (
        result[
            "actual_next_day_direction"
        ]
        .astype(int)
        .values
    )

    y_pred = (
        result[
            "predicted_direction"
        ]
        .astype(int)
        .values
    )

    probability = (
        result[
            "probability_positive"
        ]
        .astype(float)
        .values
    )

    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    brier = brier_score_loss(
        y_true,
        probability
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            probability
        )

    except ValueError:

        roc_auc = np.nan

    logloss = log_loss(
        y_true,
        np.column_stack(
            [
                1 - probability,
                probability,
            ]
        ),
        labels=[0, 1],
    )

    mae = np.mean(
        np.abs(
            result[
                "return_forecast_error"
            ]
        )
    )

    rmse = np.sqrt(
        np.mean(
            result[
                "return_forecast_error"
            ] ** 2
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Forecasts : {len(result)}"
    )

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1 Score  : {f1:.4f}"
    )

    print(
        f"ROC-AUC   : {roc_auc:.4f}"
    )

    print(
        f"Brier     : {brier:.4f}"
    )

    print(
        f"Log Loss  : {logloss:.4f}"
    )

    print(
        f"Return MAE: {mae:.6f}"
    )

    print(
        f"Return RMSE: {rmse:.6f}"
    )

    print(
        "\nMean predictive uncertainty: "
        f"{result['predictive_std'].mean():.6f}"
    )

    print(
        "Mean confidence: "
        f"{result['confidence'].mean():.6f}"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    metrics = {

        "model":
            "TimesFM 2.5 200M",

        "forecasts":
            len(result),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "brier_score":
            float(brier),

        "log_loss":
            float(logloss),

        "return_mae":
            float(mae),

        "return_rmse":
            float(rmse),

        "mean_predictive_std":
            float(
                result[
                    "predictive_std"
                ].mean()
            ),

        "mean_confidence":
            float(
                result[
                    "confidence"
                ].mean()
            ),

        "mean_entropy":
            float(
                result[
                    "predictive_entropy"
                ].mean()
            ),

        "context_length":
            MAX_CONTEXT,

        "horizon":
            FORECAST_HORIZON,
    }

    return metrics


# ============================================================
# 15. SAVE OUTPUTS
# ============================================================

def save_outputs(
    result,
    metrics,
    train,
    test,
    date_column,
    return_column,
    target_column,
):

    print("\n" + "=" * 70)

    print(
        "SAVING TIMESFM OUTPUTS"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Prediction CSV
    # --------------------------------------------------------

    result.to_csv(
        PREDICTION_FILE,
        index=False
    )

    print(
        f"Predictions saved:\n"
        f"{PREDICTION_FILE}"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    print(
        f"Metrics saved:\n"
        f"{METRICS_FILE}"
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = []

    summary.append(
        "TIMESFM FOUNDATION MODEL BENCHMARK"
    )

    summary.append(
        "=" * 60
    )

    summary.append(
        "Model: TimesFM 2.5 200M PyTorch"
    )

    summary.append(
        f"Model ID: {MODEL_ID}"
    )

    summary.append(
        f"Context length: {MAX_CONTEXT}"
    )

    summary.append(
        f"Forecast horizon: {FORECAST_HORIZON}"
    )

    summary.append(
        f"Training observations: {len(train)}"
    )

    summary.append(
        f"Test observations: {len(test)}"
    )

    summary.append(
        f"Forecasts generated: {len(result)}"
    )

    summary.append("")

    summary.append(
        "PERFORMANCE"
    )

    summary.append(
        f"Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    summary.append(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )

    summary.append(
        f"Recall: "
        f"{metrics['recall']:.6f}"
    )

    summary.append(
        f"F1: "
        f"{metrics['f1']:.6f}"
    )

    summary.append(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.6f}"
    )

    summary.append(
        f"Brier Score: "
        f"{metrics['brier_score']:.6f}"
    )

    summary.append(
        f"Log Loss: "
        f"{metrics['log_loss']:.6f}"
    )

    summary.append(
        f"Return MAE: "
        f"{metrics['return_mae']:.8f}"
    )

    summary.append(
        f"Return RMSE: "
        f"{metrics['return_rmse']:.8f}"
    )

    summary.append("")

    summary.append(
        "UNCERTAINTY"
    )

    summary.append(
        f"Mean predictive std: "
        f"{metrics['mean_predictive_std']:.8f}"
    )

    summary.append(
        f"Mean confidence: "
        f"{metrics['mean_confidence']:.8f}"
    )

    summary.append(
        f"Mean entropy: "
        f"{metrics['mean_entropy']:.8f}"
    )

    summary.append("")

    summary.append(
        "DATA ALIGNMENT"
    )

    summary.append(
        f"Date column: {date_column}"
    )

    summary.append(
        f"Return column: {return_column}"
    )

    summary.append(
        f"Target column: {target_column}"
    )

    summary.append(
        "Evaluation: chronological walk-forward"
    )

    summary.append(
        "Future observations are excluded from "
        "each forecast context."
    )

    SUMMARY_FILE.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print(
        f"Summary saved:\n"
        f"{SUMMARY_FILE}"
    )


# ============================================================
# 16. SAVE CONFIG
# ============================================================

def save_config():

    config = {

        "model":
            "TimesFM 2.5 200M PyTorch",

        "model_id":
            MODEL_ID,

        "context_length":
            MAX_CONTEXT,

        "forecast_horizon":
            FORECAST_HORIZON,

        "batch_size":
            BATCH_SIZE,

        "train_ratio":
            TRAIN_RATIO,

        "continuous_quantile_head":
            USE_CONTINUOUS_QUANTILE_HEAD,

        "device":
            DEVICE,

        "random_seed":
            RANDOM_SEED,

        "quantile_mapping":
            {
                "0": "mean",
                "1": "q10",
                "2": "q20",
                "3": "q30",
                "4": "q40",
                "5": "q50",
                "6": "q60",
                "7": "q70",
                "8": "q80",
                "9": "q90",
            },

        "probability_method":
            (
                "Normal approximation using q10/q90 "
                "dispersion"
            ),
    }

    CONFIG_FILE.write_text(
        json.dumps(
            config,
            indent=4
        ),
        encoding="utf-8"
    )

    print(
        f"Configuration saved:\n"
        f"{CONFIG_FILE}"
    )


# ============================================================
# 17. VALIDATION
# ============================================================

def validate_results(
    result
):

    print("\n" + "=" * 70)

    print(
        "FINAL TIMESFM VALIDATION"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "origin_date",
        "forecast_for_date",
        "predicted_return",
        "probability_positive",
        "predictive_std",
        "predicted_direction",
        "actual_next_day_return",
        "actual_next_day_direction",
    ]

    required_columns_valid = all(
        column in result.columns
        for column in required_columns
    )

    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probability_valid = (
        result[
            "probability_positive"
        ]
        .between(
            0,
            1
        )
        .all()
    )

    # --------------------------------------------------------
    # Uncertainty
    # --------------------------------------------------------

    uncertainty_valid = (
        np.isfinite(
            result[
                "predictive_std"
            ]
        ).all()
        and
        (
            result[
                "predictive_std"
            ] >= 0
        ).all()
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    forecast_dates = pd.to_datetime(
        result[
            "forecast_for_date"
        ]
    )

    dates_sorted = (
        forecast_dates
        .is_monotonic_increasing
    )

    duplicate_dates = (
        forecast_dates
        .duplicated()
        .sum()
    )

    no_duplicates = (
        duplicate_dates == 0
    )

    # --------------------------------------------------------
    # Origin before forecast
    # --------------------------------------------------------

    origin_dates = pd.to_datetime(
        result[
            "origin_date"
        ]
    )

    no_temporal_leakage = (
        (
            origin_dates
            <
            forecast_dates
        )
        .all()
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    directions_valid = (
        result[
            "predicted_direction"
        ]
        .isin(
            [0, 1]
        )
        .all()
    )

    # --------------------------------------------------------
    # Actual labels
    # --------------------------------------------------------

    actual_labels_valid = (
        result[
            "actual_next_day_direction"
        ]
        .isin(
            [0, 1]
        )
        .all()
    )

    # --------------------------------------------------------
    # Return finite
    # --------------------------------------------------------

    returns_valid = (
        np.isfinite(
            result[
                "predicted_return"
            ]
        ).all()
        and
        np.isfinite(
            result[
                "actual_next_day_return"
            ]
        ).all()
    )

    print(
        f"Required columns      : "
        f"{required_columns_valid}"
    )

    print(
        f"Probability valid     : "
        f"{probability_valid}"
    )

    print(
        f"Uncertainty valid     : "
        f"{uncertainty_valid}"
    )

    print(
        f"Dates sorted          : "
        f"{dates_sorted}"
    )

    print(
        f"No duplicate dates    : "
        f"{no_duplicates}"
    )

    print(
        f"No temporal leakage   : "
        f"{no_temporal_leakage}"
    )

    print(
        f"Predicted directions  : "
        f"{directions_valid}"
    )

    print(
        f"Actual directions     : "
        f"{actual_labels_valid}"
    )

    print(
        f"Returns finite        : "
        f"{returns_valid}"
    )

    passed = all(
        [
            required_columns_valid,
            probability_valid,
            uncertainty_valid,
            dates_sorted,
            no_duplicates,
            no_temporal_leakage,
            directions_valid,
            actual_labels_valid,
            returns_valid,
        ]
    )

    print(
        "\nSTEP 23 VALIDATION: "
        f"{'PASSED' if passed else 'FAILED'}"
    )

    return passed


# ============================================================
# 18. MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "PROJECT 1A - BAYESIAN REGIME DETECTION ENGINE"
    )

    print(
        "STEP 23 - TIMESFM FOUNDATION MODEL"
    )

    print("=" * 70)

    set_seed()

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        df,
        date_column,
        return_column,
        target_column,
    ) = load_data()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train,
        test,
    ) = chronological_split(
        df,
        date_column,
        return_column,
        target_column,
    )

    # --------------------------------------------------------
    # Load TimesFM
    # --------------------------------------------------------

    model = load_timesfm()

    # --------------------------------------------------------
    # Generate walk-forward forecasts
    # --------------------------------------------------------

    result = generate_forecasts(
        model,
        df,
        test,
        date_column,
        return_column,
        target_column,
    )

    # --------------------------------------------------------
    # Derived columns
    # --------------------------------------------------------

    result = add_prediction_columns(
        result
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_model(
        result
    )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    save_outputs(
        result,
        metrics,
        train,
        test,
        date_column,
        return_column,
        target_column,
    )

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------

    save_config()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation_passed = validate_results(
        result
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if validation_passed:

        print(
            "STEP 23 COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "STEP 23 COMPLETED WITH VALIDATION ISSUES"
        )

    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"1. {PREDICTION_FILE}"
    )

    print(
        f"2. {METRICS_FILE}"
    )

    print(
        f"3. {SUMMARY_FILE}"
    )

    print(
        f"4. {CONFIG_FILE}"
    )

    print("\nFinal TimesFM metrics:")

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Brier    : "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Log Loss : "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Mean uncertainty: "
        f"{metrics['mean_predictive_std']:.6f}"
    )

    print(
        "\nTimesFM is ready for ensemble integration."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()