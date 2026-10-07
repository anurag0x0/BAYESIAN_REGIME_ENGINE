from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "conformal"
MODEL_DIR = ROOT / "models" / "conformal"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_FILE = (
    DATA_DIR
    / "advanced_ensemble_predictions.csv"
)

OUTPUT_FILE = (
    DATA_DIR
    / "conformal_predictions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COVERAGE = 0.90

ALPHA = 1.0 - TARGET_COVERAGE

CALIBRATION_FRACTION = 0.40

ROLLING_WINDOW = 100

EPS = 1e-8


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def validate_probability_array(
    probabilities
):
    """
    Validate and clip probability values.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    if not np.isfinite(
        probabilities
    ).all():

        raise ValueError(
            "Probability array contains "
            "NaN or infinite values."
        )

    probabilities = np.clip(
        probabilities,
        EPS,
        1.0 - EPS
    )

    return probabilities


def binary_nonconformity_scores(
    probabilities,
    y_true
):
    """
    Classification nonconformity score:

        score = 1 - probability assigned
                to the true class

    For binary classification:

        if y = 1:
            score = 1 - p

        if y = 0:
            score = p
    """

    probabilities = (
        validate_probability_array(
            probabilities
        )
    )

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    scores = np.where(
        y_true == 1,
        1.0 - probabilities,
        probabilities
    )

    return scores


def conformal_quantile(
    scores,
    alpha
):
    """
    Finite-sample conformal quantile.

    Uses the conservative order-statistic index:

        ceil((n + 1) * (1-alpha))

    """

    scores = np.asarray(
        scores,
        dtype=float
    )

    scores = scores[
        np.isfinite(scores)
    ]

    if len(scores) == 0:

        raise ValueError(
            "No valid calibration scores."
        )

    n = len(scores)

    rank = int(
        np.ceil(
            (n + 1)
            * (1.0 - alpha)
        )
    )

    rank = min(
        max(rank, 1),
        n
    )

    sorted_scores = np.sort(
        scores
    )

    return float(
        sorted_scores[
            rank - 1
        ]
    )


def split_conformal_binary(
    calibration_probability,
    calibration_y,
    test_probability,
    alpha
):
    """
    Split conformal prediction for binary
    classification.

    Returns:

        threshold
        prediction_set
        set_size
        conformal_uncertainty
    """

    calibration_scores = (
        binary_nonconformity_scores(
            calibration_probability,
            calibration_y
        )
    )

    threshold = conformal_quantile(
        calibration_scores,
        alpha
    )

    test_probability = (
        validate_probability_array(
            test_probability
        )
    )

    # Probability assigned to each possible class.
    prob_class_0 = (
        1.0 - test_probability
    )

    prob_class_1 = (
        test_probability
    )

    include_0 = (
        1.0 - prob_class_0
        <= threshold
    )

    include_1 = (
        1.0 - prob_class_1
        <= threshold
    )

    prediction_set = []

    for a, b in zip(
        include_0,
        include_1
    ):

        labels = []

        if a:
            labels.append(0)

        if b:
            labels.append(1)

        prediction_set.append(
            labels
        )

    set_size = np.array(
        [
            len(x)
            for x in prediction_set
        ],
        dtype=int
    )

    # Probability uncertainty.
    uncertainty = (
        1.0
        - np.abs(
            test_probability - 0.5
        ) * 2.0
    )

    return (
        threshold,
        prediction_set,
        set_size,
        uncertainty
    )


# ============================================================
# APS
# ============================================================

def aps_calibration_scores(
    probabilities,
    y_true
):
    """
    APS-style cumulative probability score
    for binary classification.

    For each observation:

        Sort class probabilities descending.

        Accumulate probabilities until the
        true class is reached.

    A randomized tie-breaking component is
    not used here so the method remains
    deterministic/reproducible.
    """

    probabilities = (
        validate_probability_array(
            probabilities
        )
    )

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    scores = []

    for p, y in zip(
        probabilities,
        y_true
    ):

        class_probabilities = np.array(
            [
                1.0 - p,
                p
            ]
        )

        order = np.argsort(
            -class_probabilities
        )

        cumulative = 0.0

        score = None

        for class_index in order:

            cumulative += (
                class_probabilities[
                    class_index
                ]
            )

            if class_index == y:

                score = cumulative

                break

        scores.append(
            score
        )

    return np.asarray(
        scores,
        dtype=float
    )


def aps_prediction_sets(
    probabilities,
    threshold
):
    """
    Generate deterministic APS prediction sets.
    """

    probabilities = (
        validate_probability_array(
            probabilities
        )
    )

    prediction_sets = []

    set_sizes = []

    for p in probabilities:

        class_probabilities = np.array(
            [
                1.0 - p,
                p
            ]
        )

        order = np.argsort(
            -class_probabilities
        )

        cumulative = 0.0

        selected = []

        for class_index in order:

            cumulative += (
                class_probabilities[
                    class_index
                ]
            )

            selected.append(
                int(class_index)
            )

            if cumulative >= threshold:

                break

        selected = sorted(
            selected
        )

        prediction_sets.append(
            selected
        )

        set_sizes.append(
            len(selected)
        )

    return (
        prediction_sets,
        np.asarray(
            set_sizes,
            dtype=int
        )
    )


# ============================================================
# COVERAGE
# ============================================================

def prediction_set_coverage(
    prediction_sets,
    y_true
):
    """
    Empirical coverage.
    """

    y_true = np.asarray(
        y_true,
        dtype=int
    )

    covered = np.array(
        [
            int(
                y in prediction_set
            )
            for prediction_set, y
            in zip(
                prediction_sets,
                y_true
            )
        ]
    )

    coverage = float(
        covered.mean()
    )

    return (
        coverage,
        covered
    )


# ============================================================
# ROLLING COVERAGE
# ============================================================

def rolling_coverage(
    covered,
    window
):
    """
    Calculate rolling empirical coverage.
    """

    covered = np.asarray(
        covered,
        dtype=float
    )

    return (
        pd.Series(covered)
        .rolling(
            window=window,
            min_periods=1
        )
        .mean()
        .values
    )


# ============================================================
# DISTRIBUTION SHIFT MONITOR
# ============================================================

def calculate_distribution_shift(
    calibration_probability,
    test_probability
):
    """
    Simple distribution-shift diagnostics.

    Measures:

        calibration mean probability
        test mean probability
        absolute mean difference
        calibration std
        test std
    """

    calibration_probability = (
        np.asarray(
            calibration_probability,
            dtype=float
        )
    )

    test_probability = (
        np.asarray(
            test_probability,
            dtype=float
        )
    )

    calibration_mean = float(
        np.mean(
            calibration_probability
        )
    )

    test_mean = float(
        np.mean(
            test_probability
        )
    )

    calibration_std = float(
        np.std(
            calibration_probability
        )
    )

    test_std = float(
        np.std(
            test_probability
        )
    )

    mean_shift = abs(
        test_mean
        - calibration_mean
    )

    std_shift = abs(
        test_std
        - calibration_std
    )

    return {
        "calibration_probability_mean":
            calibration_mean,

        "test_probability_mean":
            test_mean,

        "absolute_mean_shift":
            mean_shift,

        "calibration_probability_std":
            calibration_std,

        "test_probability_std":
            test_std,

        "absolute_std_shift":
            std_shift,
    }


# ============================================================
# LOAD DATA
# ============================================================

def load_input_data():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Required Step 25 output "
            f"not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print()
    print(
        f"Loaded: {INPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    return df


# ============================================================
# IDENTIFY ENSEMBLE PROBABILITY
# ============================================================

def select_ensemble_probability(
    df
):
    """
    Prefer the constrained stacking probability
    from Step 25.

    Fallback order:

        constrained stacking
        logit stacking
        BMA
        equal weight
    """

    candidates = [
        "constrained_stacking_probability",
        "logit_stacking_probability",
        "bma_style_probability",
        "equal_weight_probability",
    ]

    for column in candidates:

        if column in df.columns:

            print(
                f"Using ensemble probability: "
                f"{column}"
            )

            return column

    raise ValueError(
        "No Step 25 ensemble probability "
        "column found."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STEP 26 - CONFORMAL PREDICTION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_input_data()

    required_columns = [
        "date",
        "actual_direction",
    ]

    for column in required_columns:

        if column not in df.columns:

            raise ValueError(
                f"Required column missing: "
                f"{column}"
            )

    probability_column = (
        select_ensemble_probability(
            df
        )
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    df[
        probability_column
    ] = pd.to_numeric(
        df[
            probability_column
        ],
        errors="coerce"
    )

    df[
        "actual_direction"
    ] = pd.to_numeric(
        df[
            "actual_direction"
        ],
        errors="coerce"
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if df["date"].isna().any():

        raise ValueError(
            "Invalid dates detected."
        )

    if df["date"].duplicated().any():

        raise ValueError(
            "Duplicate dates detected."
        )

    if df[
        probability_column
    ].isna().any():

        raise ValueError(
            "Missing ensemble probabilities."
        )

    if not set(
        df[
            "actual_direction"
        ].unique()
    ).issubset(
        {0, 1}
    ):

        raise ValueError(
            "Actual direction must contain "
            "only 0/1."
        )

    probabilities = (
        validate_probability_array(
            df[
                probability_column
            ].values
        )
    )

    y = (
        df[
            "actual_direction"
        ]
        .astype(int)
        .values
    )

    dates = df[
        "date"
    ].values

    # --------------------------------------------------------
    # Chronological calibration/test split
    # --------------------------------------------------------

    n = len(df)

    calibration_end = int(
        n * CALIBRATION_FRACTION
    )

    if calibration_end < 50:

        raise ValueError(
            "Calibration set is too small."
        )

    calibration_indices = np.arange(
        0,
        calibration_end
    )

    test_indices = np.arange(
        calibration_end,
        n
    )

    calibration_probability = (
        probabilities[
            calibration_indices
        ]
    )

    calibration_y = (
        y[
            calibration_indices
        ]
    )

    test_probability = (
        probabilities[
            test_indices
        ]
    )

    test_y = (
        y[
            test_indices
        ]
    )

    test_dates = (
        dates[
            test_indices
        ]
    )

    print()
    print(
        f"Total observations: {n}"
    )

    print(
        f"Calibration observations: "
        f"{len(calibration_indices)}"
    )

    print(
        f"Test observations: "
        f"{len(test_indices)}"
    )

    print(
        "Split: chronological"
    )

    # ========================================================
    # 1. SPLIT CONFORMAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "1. SPLIT CONFORMAL"
    )
    print("=" * 70)

    (
        split_threshold,
        split_prediction_sets,
        split_set_sizes,
        split_uncertainty
    ) = split_conformal_binary(
        calibration_probability,
        calibration_y,
        test_probability,
        ALPHA
    )

    (
        split_coverage,
        split_covered
    ) = prediction_set_coverage(
        split_prediction_sets,
        test_y
    )

    split_rolling = rolling_coverage(
        split_covered,
        ROLLING_WINDOW
    )

    print(
        f"Target coverage: "
        f"{TARGET_COVERAGE:.2%}"
    )

    print(
        f"Conformal threshold: "
        f"{split_threshold:.6f}"
    )

    print(
        f"Empirical coverage: "
        f"{split_coverage:.2%}"
    )

    print(
        f"Average prediction-set size: "
        f"{split_set_sizes.mean():.4f}"
    )

    # ========================================================
    # 2. APS
    # ========================================================

    print()
    print("=" * 70)
    print(
        "2. ADAPTIVE PREDICTION SETS (APS)"
    )
    print("=" * 70)

    aps_calibration_scores_values = (
        aps_calibration_scores(
            calibration_probability,
            calibration_y
        )
    )

    aps_threshold = conformal_quantile(
        aps_calibration_scores_values,
        ALPHA
    )

    (
        aps_prediction_sets_values,
        aps_set_sizes
    ) = aps_prediction_sets(
        test_probability,
        aps_threshold
    )

    (
        aps_coverage,
        aps_covered
    ) = prediction_set_coverage(
        aps_prediction_sets_values,
        test_y
    )

    aps_rolling = rolling_coverage(
        aps_covered,
        ROLLING_WINDOW
    )

    print(
        f"APS threshold: "
        f"{aps_threshold:.6f}"
    )

    print(
        f"APS empirical coverage: "
        f"{aps_coverage:.2%}"
    )

    print(
        f"APS average set size: "
        f"{aps_set_sizes.mean():.4f}"
    )

    # ========================================================
    # 3. DISTRIBUTION SHIFT
    # ========================================================

    print()
    print("=" * 70)
    print(
        "3. DISTRIBUTION-SHIFT MONITOR"
    )
    print("=" * 70)

    shift_metrics = (
        calculate_distribution_shift(
            calibration_probability,
            test_probability
        )
    )

    for key, value in (
        shift_metrics.items()
    ):

        print(
            f"{key}: "
            f"{value:.6f}"
        )

    # --------------------------------------------------------
    # Rolling probability shift
    # --------------------------------------------------------

    rolling_probability_mean = (
        pd.Series(
            test_probability
        )
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1
        )
        .mean()
        .values
    )

    rolling_probability_shift = (
        np.abs(
            rolling_probability_mean
            - shift_metrics[
                "calibration_probability_mean"
            ]
        )
    )

    # ========================================================
    # RESULT DATAFRAME
    # ========================================================

    result = pd.DataFrame(
        {
            "date":
                pd.to_datetime(
                    test_dates
                ),

            "actual_direction":
                test_y,

            "ensemble_probability":
                test_probability,

            "ensemble_prediction":
                (
                    test_probability
                    >= 0.5
                ).astype(int),

            "split_conformal_threshold":
                split_threshold,

            "split_conformal_set":
                [
                    ",".join(
                        map(
                            str,
                            s
                        )
                    )
                    for s in
                    split_prediction_sets
                ],

            "split_conformal_set_size":
                split_set_sizes,

            "split_conformal_covered":
                split_covered,

            "split_conformal_rolling_coverage":
                split_rolling,

            "conformal_uncertainty":
                split_uncertainty,

            "aps_threshold":
                aps_threshold,

            "aps_prediction_set":
                [
                    ",".join(
                        map(
                            str,
                            s
                        )
                    )
                    for s in
                    aps_prediction_sets_values
                ],

            "aps_set_size":
                aps_set_sizes,

            "aps_covered":
                aps_covered,

            "aps_rolling_coverage":
                aps_rolling,

            "rolling_probability_mean":
                rolling_probability_mean,

            "rolling_probability_shift":
                rolling_probability_shift,
        }
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = pd.DataFrame(
        [
            {
                "method":
                    "Split Conformal",

                "target_coverage":
                    TARGET_COVERAGE,

                "empirical_coverage":
                    split_coverage,

                "coverage_gap":
                    split_coverage
                    - TARGET_COVERAGE,

                "average_set_size":
                    split_set_sizes.mean(),

                "median_set_size":
                    np.median(
                        split_set_sizes
                    ),

                "threshold":
                    split_threshold,
            },

            {
                "method":
                    "APS",

                "target_coverage":
                    TARGET_COVERAGE,

                "empirical_coverage":
                    aps_coverage,

                "coverage_gap":
                    aps_coverage
                    - TARGET_COVERAGE,

                "average_set_size":
                    aps_set_sizes.mean(),

                "median_set_size":
                    np.median(
                        aps_set_sizes
                    ),

                "threshold":
                    aps_threshold,
            },
        ]
    )

    # ========================================================
    # COVERAGE SUMMARY
    # ========================================================

    coverage_summary = pd.DataFrame(
        [
            {
                "method":
                    "Split Conformal",

                "target_coverage":
                    TARGET_COVERAGE,

                "empirical_coverage":
                    split_coverage,

                "absolute_coverage_error":
                    abs(
                        split_coverage
                        - TARGET_COVERAGE
                    ),

                "average_set_size":
                    split_set_sizes.mean(),

                "single_prediction_rate":
                    np.mean(
                        split_set_sizes
                        == 1
                    ),

                "ambiguous_prediction_rate":
                    np.mean(
                        split_set_sizes
                        == 2
                    ),
            },

            {
                "method":
                    "APS",

                "target_coverage":
                    TARGET_COVERAGE,

                "empirical_coverage":
                    aps_coverage,

                "absolute_coverage_error":
                    abs(
                        aps_coverage
                        - TARGET_COVERAGE
                    ),

                "average_set_size":
                    aps_set_sizes.mean(),

                "single_prediction_rate":
                    np.mean(
                        aps_set_sizes
                        == 1
                    ),

                "ambiguous_prediction_rate":
                    np.mean(
                        aps_set_sizes
                        == 2
                    ),
            },
        ]
    )

    # ========================================================
    # ROLLING COVERAGE
    # ========================================================

    rolling_df = pd.DataFrame(
        {
            "date":
                pd.to_datetime(
                    test_dates
                ),

            "split_conformal_rolling_coverage":
                split_rolling,

            "aps_rolling_coverage":
                aps_rolling,

            "target_coverage":
                TARGET_COVERAGE,

            "rolling_probability_shift":
                rolling_probability_shift,
        }
    )

    # ========================================================
    # SAVE METRICS
    # ========================================================

    metrics_path = (
        REPORT_DIR
        / "metrics.csv"
    )

    metrics.to_csv(
        metrics_path,
        index=False
    )

    # ========================================================
    # SAVE COVERAGE
    # ========================================================

    coverage_path = (
        REPORT_DIR
        / "coverage.csv"
    )

    coverage_summary.to_csv(
        coverage_path,
        index=False
    )

    # ========================================================
    # SAVE APS
    # ========================================================

    aps_path = (
        REPORT_DIR
        / "aps_results.csv"
    )

    result[
        [
            "date",
            "actual_direction",
            "ensemble_probability",
            "aps_threshold",
            "aps_prediction_set",
            "aps_set_size",
            "aps_covered",
            "aps_rolling_coverage",
        ]
    ].to_csv(
        aps_path,
        index=False
    )

    # ========================================================
    # SAVE ROLLING COVERAGE
    # ========================================================

    rolling_path = (
        REPORT_DIR
        / "rolling_coverage.csv"
    )

    rolling_df.to_csv(
        rolling_path,
        index=False
    )

    # ========================================================
    # SAVE FINAL PREDICTIONS
    # ========================================================

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # ========================================================
    # SAVE DISTRIBUTION SHIFT
    # ========================================================

    shift_path = (
        REPORT_DIR
        / "distribution_shift.csv"
    )

    pd.DataFrame(
        [
            shift_metrics
        ]
    ).to_csv(
        shift_path,
        index=False
    )

    # ========================================================
    # PLOT COVERAGE
    # ========================================================

    coverage_plot_path = (
        REPORT_DIR
        / "coverage_plot.png"
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        rolling_df["date"],
        rolling_df[
            "split_conformal_rolling_coverage"
        ],
        label="Split Conformal"
    )

    plt.plot(
        rolling_df["date"],
        rolling_df[
            "aps_rolling_coverage"
        ],
        label="APS"
    )

    plt.axhline(
        TARGET_COVERAGE,
        linestyle="--",
        label="Target Coverage"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Rolling Coverage"
    )

    plt.title(
        "Rolling Conformal Coverage"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        coverage_plot_path,
        dpi=150
    )

    plt.close()

    # ========================================================
    # PLOT DISTRIBUTION SHIFT
    # ========================================================

    shift_plot_path = (
        REPORT_DIR
        / "distribution_shift_plot.png"
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.plot(
        rolling_df["date"],
        rolling_df[
            "rolling_probability_shift"
        ],
        label="Rolling Probability Shift"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Absolute Mean Probability Shift"
    )

    plt.title(
        "Distribution-Shift Monitoring"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        shift_plot_path,
        dpi=150
    )

    plt.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = (
        REPORT_DIR
        / "summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 26 - CONFORMAL PREDICTION\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            f"Input file: {INPUT_FILE}\n"
        )

        f.write(
            f"Total observations: {n}\n"
        )

        f.write(
            f"Calibration observations: "
            f"{len(calibration_indices)}\n"
        )

        f.write(
            f"Test observations: "
            f"{len(test_indices)}\n\n"
        )

        f.write(
            "Target coverage:\n"
        )

        f.write(
            f"{TARGET_COVERAGE:.4f}\n\n"
        )

        f.write(
            "SPLIT CONFORMAL\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{split_threshold:.8f}\n"
        )

        f.write(
            f"Empirical coverage: "
            f"{split_coverage:.6f}\n"
        )

        f.write(
            f"Coverage gap: "
            f"{split_coverage - TARGET_COVERAGE:.6f}\n"
        )

        f.write(
            f"Average set size: "
            f"{split_set_sizes.mean():.6f}\n\n"
        )

        f.write(
            "APS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Threshold: "
            f"{aps_threshold:.8f}\n"
        )

        f.write(
            f"Empirical coverage: "
            f"{aps_coverage:.6f}\n"
        )

        f.write(
            f"Coverage gap: "
            f"{aps_coverage - TARGET_COVERAGE:.6f}\n"
        )

        f.write(
            f"Average set size: "
            f"{aps_set_sizes.mean():.6f}\n\n"
        )

        f.write(
            "DISTRIBUTION SHIFT\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for key, value in (
            shift_metrics.items()
        ):

            f.write(
                f"{key}: {value:.8f}\n"
            )

        f.write(
            "\nINTERPRETATION\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Conformal prediction provides "
            "empirical coverage diagnostics "
            "for the held-out chronological test "
            "period.\n"
        )

        f.write(
            "Coverage is reported from the actual "
            "test outcomes and is not artificially "
            "forced to the target level.\n"
        )

        f.write(
            "Distribution shift is monitored using "
            "changes in the ensemble probability "
            "distribution between calibration and "
            "test periods.\n"
        )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "step":
            26,

        "project":
            "Bayesian Regime Detection Engine",

        "input_file":
            str(INPUT_FILE),

        "output_file":
            str(OUTPUT_FILE),

        "ensemble_probability_column":
            probability_column,

        "target_coverage":
            TARGET_COVERAGE,

        "alpha":
            ALPHA,

        "calibration_fraction":
            CALIBRATION_FRACTION,

        "rolling_window":
            ROLLING_WINDOW,

        "methods":
            [
                "Split Conformal",
                "Adaptive Prediction Sets",
                "Rolling Coverage Monitoring",
                "Distribution Shift Monitoring",
            ],

        "chronological_split":
            True,

        "future_leakage":
            False,
    }

    config_path = (
        MODEL_DIR
        / "conformal_config.json"
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

    validation = {}

    validation[
        "input_exists"
    ] = INPUT_FILE.exists()

    validation[
        "output_exists"
    ] = OUTPUT_FILE.exists()

    validation[
        "metrics_exists"
    ] = metrics_path.exists()

    validation[
        "coverage_exists"
    ] = coverage_path.exists()

    validation[
        "aps_results_exists"
    ] = aps_path.exists()

    validation[
        "rolling_coverage_exists"
    ] = rolling_path.exists()

    validation[
        "distribution_shift_exists"
    ] = shift_path.exists()

    validation[
        "summary_exists"
    ] = summary_path.exists()

    validation[
        "coverage_plot_exists"
    ] = coverage_plot_path.exists()

    validation[
        "shift_plot_exists"
    ] = shift_plot_path.exists()

    validation[
        "config_exists"
    ] = config_path.exists()

    validation[
        "dates_sorted"
    ] = bool(
        df[
            "date"
        ].is_monotonic_increasing
    )

    validation[
        "no_duplicate_dates"
    ] = bool(
        df[
            "date"
        ].duplicated().sum()
        == 0
    )

    validation[
        "probabilities_valid"
    ] = bool(
        np.isfinite(
            probabilities
        ).all()
        and
        np.all(
            probabilities >= 0
        )
        and
        np.all(
            probabilities <= 1
        )
    )

    validation[
        "actual_direction_valid"
    ] = bool(
        set(
            y.tolist()
        ).issubset(
            {0, 1}
        )
    )

    validation[
        "chronological_split_valid"
    ] = bool(
        calibration_indices[-1]
        <
        test_indices[0]
    )

    validation[
        "calibration_test_disjoint"
    ] = bool(
        set(
            calibration_indices.tolist()
        ).isdisjoint(
            set(
                test_indices.tolist()
            )
        )
    )

    validation[
        "split_coverage_valid"
    ] = bool(
        0.0
        <= split_coverage
        <= 1.0
    )

    validation[
        "aps_coverage_valid"
    ] = bool(
        0.0
        <= aps_coverage
        <= 1.0
    )

    validation[
        "split_set_sizes_valid"
    ] = bool(
        np.all(
            (
                split_set_sizes >= 0
            )
            &
            (
                split_set_sizes <= 2
            )
        )
    )

    validation[
        "aps_set_sizes_valid"
    ] = bool(
        np.all(
            (
                aps_set_sizes >= 1
            )
            &
            (
                aps_set_sizes <= 2
            )
        )
    )

    validation[
        "rolling_coverage_valid"
    ] = bool(
        np.isfinite(
            split_rolling
        ).all()
        and
        np.isfinite(
            aps_rolling
        ).all()
    )

    validation[
        "output_rows_match_test"
    ] = bool(
        len(result)
        == len(test_indices)
    )

    validation[
        "config_valid"
    ] = config_path.exists()

    validation[
        "all_required_outputs"
    ] = all(
        [
            OUTPUT_FILE.exists(),
            metrics_path.exists(),
            coverage_path.exists(),
            aps_path.exists(),
            rolling_path.exists(),
            shift_path.exists(),
            summary_path.exists(),
            coverage_plot_path.exists(),
            shift_plot_path.exists(),
            config_path.exists(),
        ]
    )

    all_passed = all(
        validation.values()
    )

    print()
    print("=" * 70)
    print(
        "STEP 26 VALIDATION"
    )
    print("=" * 70)

    for key, value in (
        validation.items()
    ):

        print(
            f"{key}: {value}"
        )

    if all_passed:

        print()
        print(
            "STEP 26 VALIDATION: PASSED"
        )

    else:

        print()
        print(
            "STEP 26 VALIDATION: FAILED"
        )

        failed = [
            key
            for key, value
            in validation.items()
            if not value
        ]

        print(
            "Failed checks:"
        )

        for item in failed:

            print(
                f"  - {item}"
            )

        raise RuntimeError(
            "Step 26 validation failed."
        )

    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 70)
    print(
        "STEP 26 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print(
        "Generated files:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        metrics_path
    )

    print(
        coverage_path
    )

    print(
        aps_path
    )

    print(
        rolling_path
    )

    print(
        shift_path
    )

    print(
        summary_path
    )

    print(
        coverage_plot_path
    )

    print(
        shift_plot_path
    )

    print(
        config_path
    )


if __name__ == "__main__":
    main()