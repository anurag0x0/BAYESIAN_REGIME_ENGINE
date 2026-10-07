"""
PROJECT 1A - BAYESIAN REGIME DETECTION ENGINE

STEP 24
Bayesian Online Changepoint Detection (BOCPD)
+ Online / Batch Regime Reconciliation

Purpose
-------
1. Detect structural changes in NIFTY 50 daily returns using BOCPD.
2. Produce online changepoint probabilities without using future observations.
3. Estimate posterior run length.
4. Reconcile online changepoint information with causal batch HMM regimes.
5. Produce audit-friendly outputs and diagnostics.

No future information is used for the BOCPD calculation.

Input
-----
data/processed/nifty50_features.csv
data/processed/nifty50_regimes_causal.csv

Outputs
-------
data/processed/bocpd_changepoints.csv
data/processed/regime_reconciliation.csv

reports/bocpd/metrics.csv
reports/bocpd/summary.txt
reports/bocpd/changepoint_plot.png
reports/bocpd/run_length_plot.png

models/bocpd/bocpd_config.json
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = ROOT / "data" / "processed" / "nifty50_features.csv"
CAUSAL_HMM_FILE = ROOT / "data" / "processed" / "nifty50_regimes_causal.csv"

BOCPD_OUTPUT = ROOT / "data" / "processed" / "bocpd_changepoints.csv"
RECONCILIATION_OUTPUT = ROOT / "data" / "processed" / "regime_reconciliation.csv"

REPORT_DIR = ROOT / "reports" / "bocpd"
MODEL_DIR = ROOT / "models" / "bocpd"

METRICS_FILE = REPORT_DIR / "metrics.csv"
SUMMARY_FILE = REPORT_DIR / "summary.txt"
CHANGEPOINT_PLOT = REPORT_DIR / "changepoint_plot.png"
RUN_LENGTH_PLOT = REPORT_DIR / "run_length_plot.png"

CONFIG_FILE = MODEL_DIR / "bocpd_config.json"


# ============================================================
# CONFIGURATION
# ============================================================

HAZARD_RATE = 1.0 / 60.0
MAX_RUN_LENGTH = 500

# Prior for Gaussian observations with unknown mean and variance.
PRIOR_MEAN = 0.0
PRIOR_KAPPA = 1.0
PRIOR_ALPHA = 1.0
PRIOR_BETA = 0.00005

# Practical alert thresholds.
CHANGEPOINT_ALERT_THRESHOLD = 0.20
STRONG_CHANGEPOINT_THRESHOLD = 0.50

# Number of top changepoints reported.
TOP_CHANGEPOINTS = 20

# Maximum historical observations used by the online detector.
# This prevents unnecessary growth of the run-length matrix.
MAX_OBSERVATIONS_FOR_PLOT = 4064


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ensure_directories():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    BOCPD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    RECONCILIATION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def validate_input_file(path: Path, name: str):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found:\n{path}"
        )


def find_column(df: pd.DataFrame, candidates, required=True):
    """
    Find a column using exact names first, then case-insensitive matching.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise KeyError(
            f"Could not find required column. "
            f"Tried: {candidates}. "
            f"Available columns: {list(df.columns)}"
        )

    return None


# ============================================================
# BOCPD MATHEMATICS
# ============================================================

def student_t_pdf(x, mean, scale, degrees_of_freedom):
    """
    Student-t probability density.

    Used as the predictive distribution for the
    Normal-Inverse-Gamma model.
    """

    if not np.isfinite(x):
        return 1e-300

    df = max(float(degrees_of_freedom), 1e-6)
    scale = max(float(scale), 1e-12)

    z = (x - mean) / scale

    log_pdf = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
        - ((df + 1.0) / 2.0) * math.log1p((z * z) / df)
    )

    return float(math.exp(np.clip(log_pdf, -745.0, 50.0)))


def predictive_parameters(
    mu,
    kappa,
    alpha,
    beta,
):
    """
    Predictive Student-t parameters for the
    Normal-Inverse-Gamma conjugate model.
    """

    mean = mu

    degrees_of_freedom = 2.0 * alpha

    variance = (
        beta * (kappa + 1.0)
        / (alpha * kappa)
    )

    scale = math.sqrt(max(variance, 1e-12))

    return mean, scale, degrees_of_freedom


def update_normal_inverse_gamma(
    mu,
    kappa,
    alpha,
    beta,
    x,
):
    """
    Bayesian update for Normal-Inverse-Gamma posterior.
    """

    kappa_new = kappa + 1.0

    mu_new = (
        kappa * mu + x
    ) / kappa_new

    alpha_new = alpha + 0.5

    beta_new = (
        beta
        + 0.5
        * kappa
        * (x - mu) ** 2
        / kappa_new
    )

    return (
        mu_new,
        kappa_new,
        alpha_new,
        beta_new,
    )


# ============================================================
# BOCPD ENGINE
# ============================================================

def run_bocpd(
    observations: np.ndarray,
    dates: np.ndarray,
    hazard_rate: float,
    max_run_length: int,
):
    """
    Bayesian Online Changepoint Detection.

    The algorithm processes observations sequentially.

    At time t:

        P(r_t = 0 | x_1:t)

    is the posterior probability that a new
    regime/run starts at observation t.

    No future observations are used.
    """

    n = len(observations)

    # Run-length posterior.
    # Index = run length.
    run_probability = np.zeros(max_run_length + 1)
    run_probability[0] = 1.0

    # Sufficient statistics for each possible run length.
    mu = np.full(max_run_length + 1, PRIOR_MEAN)
    kappa = np.full(max_run_length + 1, PRIOR_KAPPA)
    alpha = np.full(max_run_length + 1, PRIOR_ALPHA)
    beta = np.full(max_run_length + 1, PRIOR_BETA)

    changepoint_probabilities = np.zeros(n)
    most_probable_run_length = np.zeros(n, dtype=int)
    expected_run_length = np.zeros(n)

    max_run_probabilities = np.zeros(n)

    for t in range(n):

        x = float(observations[t])

        # ----------------------------------------------------
        # Predictive probability for each current run length.
        # ----------------------------------------------------

        predictive = np.zeros(max_run_length + 1)

        active = np.where(run_probability > 0)[0]

        for r in active:

            mean, scale, df = predictive_parameters(
                mu[r],
                kappa[r],
                alpha[r],
                beta[r],
            )

            predictive[r] = student_t_pdf(
                x,
                mean,
                scale,
                df,
            )

        predictive = np.maximum(
            predictive,
            1e-300,
        )

        # ----------------------------------------------------
        # Growth probabilities:
        #
        # Existing regime continues.
        # ----------------------------------------------------

        growth = (
            run_probability
            * predictive
            * (1.0 - hazard_rate)
        )

        # ----------------------------------------------------
        # Changepoint probability:
        #
        # Current observation starts a new run.
        # ----------------------------------------------------

        cp_probability = np.sum(
            run_probability
            * predictive
            * hazard_rate
        )

        new_run_probability = np.zeros(
            max_run_length + 1
        )

        new_run_probability[0] = cp_probability

        # Shift growth probabilities.
        new_run_probability[1:] = growth[:-1]

        # Normalize.
        normalization = np.sum(
            new_run_probability
        )

        if (
            not np.isfinite(normalization)
            or normalization <= 0
        ):
            new_run_probability[:] = 0.0
            new_run_probability[0] = 1.0
        else:
            new_run_probability /= normalization

        # ----------------------------------------------------
        # Update sufficient statistics.
        # ----------------------------------------------------

        new_mu = np.full(
            max_run_length + 1,
            PRIOR_MEAN,
        )

        new_kappa = np.full(
            max_run_length + 1,
            PRIOR_KAPPA,
        )

        new_alpha = np.full(
            max_run_length + 1,
            PRIOR_ALPHA,
        )

        new_beta = np.full(
            max_run_length + 1,
            PRIOR_BETA,
        )

        # Run length 0 starts with the prior updated by x.
        (
            new_mu[0],
            new_kappa[0],
            new_alpha[0],
            new_beta[0],
        ) = update_normal_inverse_gamma(
            PRIOR_MEAN,
            PRIOR_KAPPA,
            PRIOR_ALPHA,
            PRIOR_BETA,
            x,
        )

        # Existing runs grow by one observation.
        active_previous = np.where(
            run_probability > 0
        )[0]

        for r in active_previous:

            new_r = r + 1

            if new_r > max_run_length:
                continue

            (
                updated_mu,
                updated_kappa,
                updated_alpha,
                updated_beta,
            ) = update_normal_inverse_gamma(
                mu[r],
                kappa[r],
                alpha[r],
                beta[r],
                x,
            )

            new_mu[new_r] = updated_mu
            new_kappa[new_r] = updated_kappa
            new_alpha[new_r] = updated_alpha
            new_beta[new_r] = updated_beta

        run_probability = new_run_probability
        mu = new_mu
        kappa = new_kappa
        alpha = new_alpha
        beta = new_beta

        # ----------------------------------------------------
        # Save posterior diagnostics.
        # ----------------------------------------------------

        cp_prob = run_probability[0]

        changepoint_probabilities[t] = cp_prob

        most_probable_run_length[t] = int(
            np.argmax(run_probability)
        )

        run_lengths = np.arange(
            max_run_length + 1
        )

        expected_run_length[t] = np.sum(
            run_lengths * run_probability
        )

        max_run_probabilities[t] = np.max(
            run_probability
        )

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "daily_return": observations,
            "changepoint_probability": changepoint_probabilities,
            "most_probable_run_length": most_probable_run_length,
            "expected_run_length": expected_run_length,
            "max_run_length_probability": max_run_probabilities,
        }
    )

    result["changepoint_alert"] = (
        result["changepoint_probability"]
        >= CHANGEPOINT_ALERT_THRESHOLD
    )

    result["strong_changepoint"] = (
        result["changepoint_probability"]
        >= STRONG_CHANGEPOINT_THRESHOLD
    )

    result["online_state"] = np.where(
        result["strong_changepoint"],
        "STRONG_CHANGEPOINT",
        np.where(
            result["changepoint_alert"],
            "CHANGEPOINT_ALERT",
            "STABLE_RUN",
        ),
    )

    return result


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    print("=" * 70)
    print("LOADING NIFTY 50 FEATURE DATA")
    print("=" * 70)

    validate_input_file(
        FEATURE_FILE,
        "NIFTY 50 feature file",
    )

    df = pd.read_csv(FEATURE_FILE)

    date_col = find_column(
        df,
        ["date"],
    )

    return_col = find_column(
        df,
        ["daily_return", "log_return"],
    )

    target_col = find_column(
        df,
        ["target_next_day_direction"],
        required=False,
    )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df[return_col] = pd.to_numeric(
        df[return_col],
        errors="coerce",
    )

    df = df[
        [date_col, return_col]
    ].copy()

    df = df.dropna()

    df = df.sort_values(
        date_col
    ).reset_index(drop=True)

    if df[date_col].duplicated().any():
        raise ValueError(
            "Duplicate dates detected."
        )

    print(
        f"Input file      : {FEATURE_FILE}"
    )
    print(
        f"Rows            : {len(df)}"
    )
    print(
        f"Date column     : {date_col}"
    )
    print(
        f"Return column   : {return_col}"
    )

    if target_col:
        print(
            f"Target column   : {target_col}"
        )

    print(
        f"Date range      : "
        f"{df[date_col].min().date()} -> "
        f"{df[date_col].max().date()}"
    )

    return (
        df,
        date_col,
        return_col,
    )


# ============================================================
# RUN BOCPD
# ============================================================

def create_bocpd_output():

    df, date_col, return_col = load_market_data()

    observations = (
        df[return_col]
        .astype(float)
        .to_numpy()
    )

    dates = (
        df[date_col]
        .to_numpy()
    )

    print()
    print("=" * 70)
    print("RUNNING BAYESIAN ONLINE CHANGEPOINT DETECTION")
    print("=" * 70)

    print(
        f"Observations    : {len(observations)}"
    )
    print(
        f"Hazard rate     : {HAZARD_RATE:.6f}"
    )
    print(
        f"Expected run    : "
        f"{1.0 / HAZARD_RATE:.1f} observations"
    )
    print(
        f"Max run length  : {MAX_RUN_LENGTH}"
    )

    result = run_bocpd(
        observations=observations,
        dates=dates,
        hazard_rate=HAZARD_RATE,
        max_run_length=MAX_RUN_LENGTH,
    )

    result.to_csv(
        BOCPD_OUTPUT,
        index=False,
    )

    print()
    print(
        f"BOCPD output saved:"
    )
    print(
        BOCPD_OUTPUT
    )

    return result


# ============================================================
# LOAD CAUSAL HMM
# ============================================================

def load_causal_hmm():

    print()
    print("=" * 70)
    print("LOADING CAUSAL HMM REGIME DATA")
    print("=" * 70)

    validate_input_file(
        CAUSAL_HMM_FILE,
        "Causal HMM regime file",
    )

    df = pd.read_csv(
        CAUSAL_HMM_FILE
    )

    date_col = find_column(
        df,
        ["date"],
    )

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df = df.dropna(
        subset=[date_col]
    )

    df = df.sort_values(
        date_col
    ).reset_index(drop=True)

    print(
        f"Input file      : {CAUSAL_HMM_FILE}"
    )
    print(
        f"Rows            : {len(df)}"
    )
    print(
        f"Columns         : {list(df.columns)}"
    )

    return df, date_col


# ============================================================
# IDENTIFY HMM REGIME COLUMN
# ============================================================

def identify_regime_columns(df):

    label_candidates = [
        "regime_label",
        "regime",
        "causal_regime",
        "predicted_regime",
        "hmm_regime",
        "state",
        "regime_name",
    ]

    label_col = find_column(
        df,
        label_candidates,
        required=False,
    )

    probability_cols = []

    for col in df.columns:

        col_lower = str(col).lower()

        if (
            "prob" in col_lower
            and "regime" in col_lower
        ):
            probability_cols.append(col)

    if not probability_cols:

        for col in df.columns:

            col_lower = str(col).lower()

            if (
                col_lower.startswith("probability_")
                or col_lower.startswith("state_probability")
                or col_lower.startswith("regime_probability")
            ):
                probability_cols.append(col)

    return (
        label_col,
        probability_cols,
    )


# ============================================================
# ONLINE / BATCH RECONCILIATION
# ============================================================

def reconcile_regimes(
    bocpd_df,
    hmm_df,
    hmm_date_col,
):

    print()
    print("=" * 70)
    print("RECONCILING ONLINE BOCPD WITH BATCH CAUSAL HMM")
    print("=" * 70)

    left = bocpd_df.copy()

    right = hmm_df.copy()

    left["date"] = pd.to_datetime(
        left["date"]
    )

    right["date"] = pd.to_datetime(
        right[hmm_date_col]
    )

    label_col, probability_cols = identify_regime_columns(
        right
    )

    print(
        f"HMM regime label column: "
        f"{label_col}"
    )

    print(
        f"HMM probability columns : "
        f"{probability_cols}"
    )

    # Keep only useful HMM columns.
    selected_hmm_cols = [
        "date"
    ]

    if label_col:
        selected_hmm_cols.append(
            label_col
        )

    for col in probability_cols:
        if col not in selected_hmm_cols:
            selected_hmm_cols.append(col)

    right_selected = right[
        selected_hmm_cols
    ].copy()

    merged = pd.merge(
        left,
        right_selected,
        on="date",
        how="inner",
    )

    if len(merged) == 0:
        raise ValueError(
            "No common dates found between BOCPD "
            "and causal HMM outputs."
        )

    # --------------------------------------------------------
    # Determine regime-change indicator from HMM labels.
    # --------------------------------------------------------

    if label_col:

        merged["batch_regime"] = (
            merged[label_col]
            .astype(str)
        )

        merged["batch_regime_change"] = (
            merged["batch_regime"]
            != merged["batch_regime"].shift(1)
        )

        # First observation is not a genuine change.
        merged.loc[
            merged.index[0],
            "batch_regime_change"
        ] = False

    else:

        merged["batch_regime"] = (
            "UNKNOWN"
        )

        merged["batch_regime_change"] = False

    # --------------------------------------------------------
    # Online event categories.
    # --------------------------------------------------------

    merged["online_change_event"] = (
        merged["changepoint_probability"]
        >= CHANGEPOINT_ALERT_THRESHOLD
    )

    merged["online_strong_change_event"] = (
        merged["changepoint_probability"]
        >= STRONG_CHANGEPOINT_THRESHOLD
    )

    # --------------------------------------------------------
    # Reconciliation logic.
    # --------------------------------------------------------

    merged["reconciliation_status"] = "NO_CHANGE"

    merged.loc[
        (
            merged["online_change_event"]
            & merged["batch_regime_change"]
        ),
        "reconciliation_status"
    ] = "CONFIRMED_REGIME_CHANGE"

    merged.loc[
        (
            merged["online_change_event"]
            & ~merged["batch_regime_change"]
        ),
        "reconciliation_status"
    ] = "ONLINE_CHANGE_ONLY"

    merged.loc[
        (
            ~merged["online_change_event"]
            & merged["batch_regime_change"]
        ),
        "reconciliation_status"
    ] = "BATCH_CHANGE_ONLY"

    # --------------------------------------------------------
    # Combined regime interpretation.
    # --------------------------------------------------------

    def combined_signal(row):

        cp = row[
            "changepoint_probability"
        ]

        batch_change = bool(
            row["batch_regime_change"]
        )

        if (
            cp >= STRONG_CHANGEPOINT_THRESHOLD
            and batch_change
        ):
            return "HIGH_CONFIDENCE_TRANSITION"

        if cp >= STRONG_CHANGEPOINT_THRESHOLD:
            return "ONLINE_STRONG_WARNING"

        if (
            cp >= CHANGEPOINT_ALERT_THRESHOLD
            and batch_change
        ):
            return "LIKELY_REGIME_TRANSITION"

        if cp >= CHANGEPOINT_ALERT_THRESHOLD:
            return "ONLINE_WARNING"

        if batch_change:
            return "BATCH_REGIME_CHANGE"

        return "STABLE"

    merged["combined_regime_signal"] = (
        merged.apply(
            combined_signal,
            axis=1,
        )
    )

    return merged


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    bocpd_df,
    reconciliation_df,
):

    print()
    print("=" * 70)
    print("CALCULATING BOCPD / RECONCILIATION METRICS")
    print("=" * 70)

    n = len(bocpd_df)

    alert_count = int(
        bocpd_df[
            "changepoint_alert"
        ].sum()
    )

    strong_count = int(
        bocpd_df[
            "strong_changepoint"
        ].sum()
    )

    max_cp = float(
        bocpd_df[
            "changepoint_probability"
        ].max()
    )

    mean_cp = float(
        bocpd_df[
            "changepoint_probability"
        ].mean()
    )

    median_run = float(
        bocpd_df[
            "expected_run_length"
        ].median()
    )

    # Reconciliation metrics.
    confirmed = int(
        (
            reconciliation_df[
                "reconciliation_status"
            ]
            == "CONFIRMED_REGIME_CHANGE"
        ).sum()
    )

    online_only = int(
        (
            reconciliation_df[
                "reconciliation_status"
            ]
            == "ONLINE_CHANGE_ONLY"
        ).sum()
    )

    batch_only = int(
        (
            reconciliation_df[
                "reconciliation_status"
            ]
            == "BATCH_CHANGE_ONLY"
        ).sum()
    )

    no_change = int(
        (
            reconciliation_df[
                "reconciliation_status"
            ]
            == "NO_CHANGE"
        ).sum()
    )

    batch_changes = int(
        reconciliation_df[
            "batch_regime_change"
        ].sum()
    )

    online_events = int(
        reconciliation_df[
            "online_change_event"
        ].sum()
    )

    if batch_changes > 0:
        online_recall = (
            confirmed / batch_changes
        )
    else:
        online_recall = np.nan

    if online_events > 0:
        confirmation_rate = (
            confirmed / online_events
        )
    else:
        confirmation_rate = np.nan

    metrics = pd.DataFrame(
        [
            {
                "metric": "observations",
                "value": n,
            },
            {
                "metric": "alert_threshold",
                "value": CHANGEPOINT_ALERT_THRESHOLD,
            },
            {
                "metric": "strong_threshold",
                "value": STRONG_CHANGEPOINT_THRESHOLD,
            },
            {
                "metric": "hazard_rate",
                "value": HAZARD_RATE,
            },
            {
                "metric": "expected_prior_run_length",
                "value": 1.0 / HAZARD_RATE,
            },
            {
                "metric": "changepoint_alert_count",
                "value": alert_count,
            },
            {
                "metric": "strong_changepoint_count",
                "value": strong_count,
            },
            {
                "metric": "maximum_changepoint_probability",
                "value": max_cp,
            },
            {
                "metric": "mean_changepoint_probability",
                "value": mean_cp,
            },
            {
                "metric": "median_expected_run_length",
                "value": median_run,
            },
            {
                "metric": "batch_regime_change_count",
                "value": batch_changes,
            },
            {
                "metric": "online_change_event_count",
                "value": online_events,
            },
            {
                "metric": "confirmed_regime_changes",
                "value": confirmed,
            },
            {
                "metric": "online_change_only",
                "value": online_only,
            },
            {
                "metric": "batch_change_only",
                "value": batch_only,
            },
            {
                "metric": "no_change",
                "value": no_change,
            },
            {
                "metric": "online_confirmation_rate",
                "value": confirmation_rate,
            },
            {
                "metric": "online_recall_of_batch_changes",
                "value": online_recall,
            },
        ]
    )

    metrics.to_csv(
        METRICS_FILE,
        index=False,
    )

    return metrics


# ============================================================
# TOP CHANGEPOINT EVENTS
# ============================================================

def print_top_changepoints(
    bocpd_df,
):

    print()
    print("=" * 70)
    print("TOP BOCPD CHANGEPOINT EVENTS")
    print("=" * 70)

    top = (
        bocpd_df
        .sort_values(
            "changepoint_probability",
            ascending=False,
        )
        .head(TOP_CHANGEPOINTS)
        .copy()
    )

    display_cols = [
        "date",
        "daily_return",
        "changepoint_probability",
        "expected_run_length",
        "online_state",
    ]

    print(
        top[display_cols]
        .to_string(
            index=False
        )
    )

    return top


# ============================================================
# PLOTS
# ============================================================

def create_changepoint_plot(
    bocpd_df,
):

    fig, ax1 = plt.subplots(
        figsize=(14, 6)
    )

    ax1.plot(
        bocpd_df["date"],
        bocpd_df["daily_return"],
        linewidth=0.8,
        label="Daily Return",
    )

    alerts = bocpd_df[
        bocpd_df[
            "changepoint_alert"
        ]
    ]

    strong = bocpd_df[
        bocpd_df[
            "strong_changepoint"
        ]
    ]

    if len(alerts) > 0:

        ax1.scatter(
            alerts["date"],
            alerts["daily_return"],
            marker="o",
            s=18,
            label="CP Alert",
        )

    if len(strong) > 0:

        ax1.scatter(
            strong["date"],
            strong["daily_return"],
            marker="x",
            s=35,
            label="Strong CP",
        )

    ax1.set_title(
        "NIFTY 50 Daily Returns with BOCPD Changepoint Alerts"
    )

    ax1.set_xlabel(
        "Date"
    )

    ax1.set_ylabel(
        "Daily Return"
    )

    ax1.legend()

    ax1.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        CHANGEPOINT_PLOT,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


def create_run_length_plot(
    bocpd_df,
):

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        bocpd_df["date"],
        bocpd_df["expected_run_length"],
        linewidth=1.0,
        label="Expected Run Length",
    )

    ax.plot(
        bocpd_df["date"],
        bocpd_df["most_probable_run_length"],
        linewidth=0.8,
        label="Most Probable Run Length",
    )

    ax.set_title(
        "BOCPD Posterior Run-Length Diagnostics"
    )

    ax.set_xlabel(
        "Date"
    )

    ax.set_ylabel(
        "Run Length"
    )

    ax.legend()

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    fig.savefig(
        RUN_LENGTH_PLOT,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# VALIDATION
# ============================================================

def validate_outputs(
    bocpd_df,
    reconciliation_df,
):

    print()
    print("=" * 70)
    print("FINAL STEP 24 VALIDATION")
    print("=" * 70)

    checks = {}

    # Required columns.
    required_bocpd = [
        "date",
        "daily_return",
        "changepoint_probability",
        "most_probable_run_length",
        "expected_run_length",
        "max_run_length_probability",
        "changepoint_alert",
        "strong_changepoint",
        "online_state",
    ]

    required_reconciliation = [
        "date",
        "changepoint_probability",
        "batch_regime",
        "batch_regime_change",
        "online_change_event",
        "reconciliation_status",
        "combined_regime_signal",
    ]

    checks[
        "Required BOCPD columns"
    ] = all(
        c in bocpd_df.columns
        for c in required_bocpd
    )

    checks[
        "Required reconciliation columns"
    ] = all(
        c in reconciliation_df.columns
        for c in required_reconciliation
    )

    # Probability validity.
    cp = bocpd_df[
        "changepoint_probability"
    ].to_numpy()

    checks[
        "Probability range valid"
    ] = bool(
        np.all(
            np.isfinite(cp)
        )
        and np.all(
            (cp >= 0)
            & (cp <= 1)
        )
    )

    # Run length validity.
    run_length = bocpd_df[
        "expected_run_length"
    ].to_numpy()

    checks[
        "Run length valid"
    ] = bool(
        np.all(
            np.isfinite(run_length)
        )
        and np.all(
            run_length >= 0
        )
    )

    # Dates.
    checks[
        "Dates sorted"
    ] = bool(
        bocpd_df[
            "date"
        ].is_monotonic_increasing
    )

    checks[
        "No duplicate dates"
    ] = bool(
        ~bocpd_df[
            "date"
        ].duplicated().any()
    )

    # Reconciliation date alignment.
    checks[
        "Reconciliation dates sorted"
    ] = bool(
        reconciliation_df[
            "date"
        ].is_monotonic_increasing
    )

    checks[
        "No reconciliation duplicates"
    ] = bool(
        ~reconciliation_df[
            "date"
        ].duplicated().any()
    )

    # Status validity.
    allowed_statuses = {
        "NO_CHANGE",
        "CONFIRMED_REGIME_CHANGE",
        "ONLINE_CHANGE_ONLY",
        "BATCH_CHANGE_ONLY",
    }

    checks[
        "Reconciliation statuses valid"
    ] = set(
        reconciliation_df[
            "reconciliation_status"
        ].dropna().unique()
    ).issubset(
        allowed_statuses
    )

    # No future information.
    #
    # BOCPD at date t is generated by processing
    # observations only through t.
    #
    # We verify chronological uniqueness and
    # one-to-one output structure.
    checks[
        "Online chronological structure valid"
    ] = bool(
        len(bocpd_df)
        == len(
            bocpd_df[
                "date"
            ].unique()
        )
    )

    # Files.
    checks[
        "BOCPD output exists"
    ] = BOCPD_OUTPUT.exists()

    checks[
        "Reconciliation output exists"
    ] = RECONCILIATION_OUTPUT.exists()

    checks[
        "Metrics file exists"
    ] = METRICS_FILE.exists()

    checks[
        "Summary file exists"
    ] = SUMMARY_FILE.exists()

    checks[
        "Changepoint plot exists"
    ] = CHANGEPOINT_PLOT.exists()

    checks[
        "Run length plot exists"
    ] = RUN_LENGTH_PLOT.exists()

    checks[
        "Configuration exists"
    ] = CONFIG_FILE.exists()

    for name, passed in checks.items():

        print(
            f"{name:<42}: "
            f"{bool(passed)}"
        )

    overall = all(
        checks.values()
    )

    print()

    if overall:
        print(
            "STEP 24 VALIDATION: PASSED"
        )
    else:
        print(
            "STEP 24 VALIDATION: FAILED"
        )

    return overall, checks


# ============================================================
# SUMMARY
# ============================================================

def write_summary(
    bocpd_df,
    reconciliation_df,
    metrics,
    validation_passed,
):

    top = (
        bocpd_df
        .sort_values(
            "changepoint_probability",
            ascending=False,
        )
        .head(TOP_CHANGEPOINTS)
    )

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "PROJECT 1A - STEP 24\n"
        )

        f.write(
            "Bayesian Online Changepoint Detection "
            "+ Online/Batch Reconciliation\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            "PURPOSE\n"
        )

        f.write(
            "Detect structural changes in NIFTY 50 daily "
            "returns using Bayesian Online Changepoint "
            "Detection and reconcile online change signals "
            "with causal batch HMM regimes.\n\n"
        )

        f.write(
            "CONFIGURATION\n"
        )

        f.write(
            f"Hazard rate: {HAZARD_RATE}\n"
        )

        f.write(
            f"Expected prior run length: "
            f"{1.0 / HAZARD_RATE:.2f}\n"
        )

        f.write(
            f"Maximum run length: "
            f"{MAX_RUN_LENGTH}\n"
        )

        f.write(
            f"Change point alert threshold: "
            f"{CHANGEPOINT_ALERT_THRESHOLD}\n"
        )

        f.write(
            f"Strong change point threshold: "
            f"{STRONG_CHANGEPOINT_THRESHOLD}\n\n"
        )

        f.write(
            "BOCPD RESULTS\n"
        )

        for _, row in metrics.iterrows():

            f.write(
                f"{row['metric']}: "
                f"{row['value']}\n"
            )

        f.write(
            "\nTOP CHANGEPOINT EVENTS\n"
        )

        f.write(
            top[
                [
                    "date",
                    "daily_return",
                    "changepoint_probability",
                    "expected_run_length",
                ]
            ].to_string(
                index=False
            )
        )

        f.write(
            "\n\nRECONCILIATION COUNTS\n"
        )

        counts = (
            reconciliation_df[
                "reconciliation_status"
            ]
            .value_counts()
        )

        for status, count in counts.items():

            f.write(
                f"{status}: {count}\n"
            )

        f.write(
            "\nVALIDATION\n"
        )

        f.write(
            f"STEP 24 validation passed: "
            f"{validation_passed}\n"
        )

        f.write(
            "\nINTERPRETATION\n"
        )

        f.write(
            "BOCPD provides an online probability that a "
            "new statistical run/regime begins at each "
            "observation. The causal HMM provides a batch "
            "regime classification. Events where both "
            "systems indicate a transition are labelled "
            "CONFIRMED_REGIME_CHANGE. Online-only and "
            "batch-only events are retained separately "
            "for auditability rather than being silently "
            "forced into agreement.\n"
        )


# ============================================================
# CONFIG FILE
# ============================================================

def write_config():

    config = {
        "project": "Bayesian Regime Detection Engine",
        "step": 24,
        "method": "Bayesian Online Changepoint Detection",
        "observation": "NIFTY 50 daily return",
        "hazard_rate": HAZARD_RATE,
        "expected_prior_run_length": 1.0 / HAZARD_RATE,
        "max_run_length": MAX_RUN_LENGTH,
        "prior": {
            "distribution": "Normal-Inverse-Gamma",
            "mean": PRIOR_MEAN,
            "kappa": PRIOR_KAPPA,
            "alpha": PRIOR_ALPHA,
            "beta": PRIOR_BETA,
        },
        "thresholds": {
            "change_point_alert": CHANGEPOINT_ALERT_THRESHOLD,
            "strong_change_point": STRONG_CHANGEPOINT_THRESHOLD,
        },
        "inputs": {
            "features": str(FEATURE_FILE),
            "causal_hmm": str(CAUSAL_HMM_FILE),
        },
        "outputs": {
            "bocpd": str(BOCPD_OUTPUT),
            "reconciliation": str(RECONCILIATION_OUTPUT),
            "metrics": str(METRICS_FILE),
            "summary": str(SUMMARY_FILE),
            "changepoint_plot": str(CHANGEPOINT_PLOT),
            "run_length_plot": str(RUN_LENGTH_PLOT),
        },
        "online_leakage_policy": (
            "BOCPD processes observations sequentially and "
            "does not use future observations for the "
            "changepoint probability at time t."
        ),
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("PROJECT 1A - BAYESIAN REGIME DETECTION ENGINE")
    print("STEP 24 - BOCPD + ONLINE/BATCH RECONCILIATION")
    print("=" * 70)
    print()

    ensure_directories()

    # --------------------------------------------------------
    # 1. BOCPD
    # --------------------------------------------------------

    bocpd_df = create_bocpd_output()

    # --------------------------------------------------------
    # 2. Causal HMM
    # --------------------------------------------------------

    hmm_df, hmm_date_col = load_causal_hmm()

    # --------------------------------------------------------
    # 3. Reconciliation
    # --------------------------------------------------------

    reconciliation_df = reconcile_regimes(
        bocpd_df,
        hmm_df,
        hmm_date_col,
    )

    reconciliation_df.to_csv(
        RECONCILIATION_OUTPUT,
        index=False,
    )

    print()
    print(
        "Reconciliation output saved:"
    )
    print(
        RECONCILIATION_OUTPUT
    )

    # --------------------------------------------------------
    # 4. Metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        bocpd_df,
        reconciliation_df,
    )

    # --------------------------------------------------------
    # 5. Top events
    # --------------------------------------------------------

    top = print_top_changepoints(
        bocpd_df
    )

    # --------------------------------------------------------
    # 6. Plots
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING BOCPD VISUALIZATIONS")
    print("=" * 70)

    create_changepoint_plot(
        bocpd_df
    )

    create_run_length_plot(
        bocpd_df
    )

    print(
        f"Saved: {CHANGEPOINT_PLOT}"
    )

    print(
        f"Saved: {RUN_LENGTH_PLOT}"
    )

    # --------------------------------------------------------
    # 7. Configuration
    # --------------------------------------------------------

    write_config()

    # --------------------------------------------------------
    # 8. Validation
    # --------------------------------------------------------

    validation_passed, checks = validate_outputs(
        bocpd_df,
        reconciliation_df,
    )

    # --------------------------------------------------------
    # 9. Summary
    # --------------------------------------------------------

    write_summary(
        bocpd_df,
        reconciliation_df,
        metrics,
        validation_passed,
    )

    print()
    print(
        "=" * 70
    )

    if validation_passed:

        print(
            "STEP 24 COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "STEP 24 COMPLETED WITH VALIDATION ERRORS"
        )

    print(
        "=" * 70
    )

    print()
    print(
        "Generated files:"
    )

    print(
        f"1. {BOCPD_OUTPUT}"
    )

    print(
        f"2. {RECONCILIATION_OUTPUT}"
    )

    print(
        f"3. {METRICS_FILE}"
    )

    print(
        f"4. {SUMMARY_FILE}"
    )

    print(
        f"5. {CHANGEPOINT_PLOT}"
    )

    print(
        f"6. {RUN_LENGTH_PLOT}"
    )

    print(
        f"7. {CONFIG_FILE}"
    )

    print()

    print(
        "Step 24 is ready for Step 25:"
    )

    print(
        "Bayesian Model Averaging + "
        "Constrained Stacking + WAIC/LOO"
    )


if __name__ == "__main__":
    main()