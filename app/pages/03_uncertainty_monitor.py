# ============================================================
# 03_uncERTAINTY_MONITOR.py
# BAYESIAN REGIME DETECTION ENGINE
# Advanced Quantitative Uncertainty Terminal
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Uncertainty Monitor | Bayesian Regime Engine",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

PROCESSED = ROOT / "data" / "processed"
EVALUATION = ROOT / "reports" / "evaluation"


# ============================================================
# ADVANCED CSS
# ============================================================

st.markdown(
    """
    <style>

    .metric-card {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 18px;
        background: rgba(128,128,128,0.05);
        min-height: 125px;
    }

    .metric-title {
        font-size: 13px;
        opacity: 0.70;
        margin-bottom: 8px;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .metric-sub {
        font-size: 12px;
        opacity: 0.65;
    }

    .status-box {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 14px;
        padding: 20px;
        background: rgba(128,128,128,0.04);
    }

    .section-note {
        font-size: 13px;
        opacity: 0.68;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_csv(path):

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)

    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_all_data():

    return {

        "bayesian": load_csv(
            PROCESSED / "bayesian_predictions.csv"
        ),

        "smc": load_csv(
            PROCESSED / "smc_predictions.csv"
        ),

        "chronos": load_csv(
            PROCESSED / "chronos_predictions.csv"
        ),

        "ensemble": load_csv(
            PROCESSED / "ensemble_predictions.csv"
        ),

        "integrated": load_csv(
            PROCESSED / "causal_regime_integrated_predictions.csv"
        ),

        "walk_forward": load_csv(
            PROCESSED / "walk_forward_backtest.csv"
        ),

        "regimes": load_csv(
            PROCESSED / "nifty50_regimes_causal.csv"
        ),

        "basic_performance": load_csv(
            EVALUATION / "statistical_basic_performance.csv"
        ),

        "bootstrap": load_csv(
            EVALUATION / "bootstrap_confidence_intervals.csv"
        ),

        "significance": load_csv(
            EVALUATION / "model_significance_tests.csv"
        ),

    }


DATA = load_all_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    if df.empty:
        return None

    mapping = {
        str(c).lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in mapping:
            return mapping[candidate.lower()]

    return None


def to_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def probability_uncertainty(prob):

    """
    Probability-based uncertainty.

    p = 0.50 -> uncertainty = 1
    p = 0 or 1 -> uncertainty = 0
    """

    return (
        1
        -
        2 * (prob - 0.5).abs()
    ).clip(0, 1)


def entropy_binary(prob):

    """
    Normalized binary entropy.

    0 -> maximum confidence
    1 -> maximum uncertainty
    """

    p = np.clip(
        prob,
        1e-9,
        1 - 1e-9
    )

    entropy = -(
        p * np.log2(p)
        +
        (1 - p)
        * np.log2(1 - p)
    )

    return entropy


def safe_pct(value):

    if pd.isna(value):
        return "N/A"

    return f"{value * 100:.1f}%"


def safe_float(value, decimals=3):

    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}"


# ============================================================
# HEADER
# ============================================================

st.title(
    "Uncertainty Monitor"
)

st.caption(
    "Advanced probabilistic confidence, predictive dispersion, "
    "entropy and regime-aware uncertainty analytics."
)


# ============================================================
# SELECT MODEL
# ============================================================

available_models = []

for key, label in [
    ("bayesian", "Bayesian Ridge"),
    ("smc", "Sequential Monte Carlo"),
    ("chronos", "Chronos"),
    ("ensemble", "Ensemble"),
]:

    if not DATA[key].empty:
        available_models.append(
            (key, label)
        )


if not available_models:

    st.error(
        "No prediction datasets were found."
    )

    st.stop()


with st.sidebar:

    st.subheader(
        "Uncertainty Controls"
    )

    model_labels = [
        label
        for _, label in available_models
    ]

    selected_label = st.selectbox(
        "Prediction Model",
        model_labels
    )

    selected_key = dict(
        (label, key)
        for key, label in available_models
    )[selected_label]

    st.divider()

    rolling_window = st.slider(
        "Rolling Window",
        min_value=10,
        max_value=120,
        value=30,
        step=5
    )

    high_threshold = st.slider(
        "High Uncertainty Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.75,
        step=0.05
    )


df = DATA[selected_key].copy()


# ============================================================
# DATE DETECTION
# ============================================================

date_col = find_column(
    df,
    [
        "forecast_for_date",
        "prediction_for_date",
        "date",
        "origin_date",
        "feature_date",
    ]
)


if date_col:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df = (
        df
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .reset_index(drop=True)
    )


# ============================================================
# PROBABILITY DETECTION
# ============================================================

prob_col = find_column(
    df,
    [
        "probability_positive",
        "probability_up",
        "prob_up",
        "probability",
    ]
)


if prob_col:

    df["probability"] = (
        to_numeric(
            df[prob_col]
        )
        .clip(0, 1)
    )

else:

    df["probability"] = np.nan


# ============================================================
# ACTUAL / PREDICTED DIRECTION
# ============================================================

pred_col = find_column(
    df,
    [
        "predicted_direction",
        "predicted_direction_label",
    ]
)

actual_col = find_column(
    df,
    [
        "actual_direction",
        "actual_next_day_direction",
    ]
)


if pred_col:

    df["predicted_direction_clean"] = (
        df[pred_col]
        .astype(str)
        .str.lower()
        .str.strip()
    )

else:

    df["predicted_direction_clean"] = ""


if actual_col:

    df["actual_direction_clean"] = (
        df[actual_col]
        .astype(str)
        .str.lower()
        .str.strip()
    )

else:

    df["actual_direction_clean"] = ""


# ============================================================
# PREDICTIVE STD
# ============================================================

std_col = find_column(
    df,
    [
        "predictive_std",
        "latent_state_std",
        "prediction_std",
        "std",
    ]
)


if std_col:

    df["predictive_std"] = to_numeric(
        df[std_col]
    )

else:

    df["predictive_std"] = np.nan


# ============================================================
# UNCERTAINTY ENGINE
# ============================================================

df["probability_uncertainty"] = (
    probability_uncertainty(
        df["probability"]
    )
)


df["entropy_uncertainty"] = (
    entropy_binary(
        df["probability"]
    )
)


# If actual model uncertainty exists,
# preserve it. Otherwise probability uncertainty.

df["uncertainty_score"] = np.where(
    df["predictive_std"].notna(),
    df["probability_uncertainty"],
    df["entropy_uncertainty"]
)


# ============================================================
# CONFIDENCE
# ============================================================

df["confidence_score"] = (
    1
    - df["uncertainty_score"]
).clip(0, 1)


# ============================================================
# HIGH UNCERTAINTY FLAG
# ============================================================

df["high_uncertainty"] = (
    df["uncertainty_score"]
    >= high_threshold
)


# ============================================================
# CORRECTNESS
# ============================================================

if (
    pred_col
    and actual_col
):

    df["correct"] = (
        df["predicted_direction_clean"]
        ==
        df["actual_direction_clean"]
    )

else:

    df["correct"] = np.nan


# ============================================================
# ROLLING ANALYTICS
# ============================================================

df["rolling_uncertainty"] = (
    df["uncertainty_score"]
    .rolling(
        rolling_window,
        min_periods=5
    )
    .mean()
)


df["rolling_confidence"] = (
    df["confidence_score"]
    .rolling(
        rolling_window,
        min_periods=5
    )
    .mean()
)


# ============================================================
# UNCERTAINTY STATE
# ============================================================

def classify_uncertainty(value):

    if pd.isna(value):
        return "Unknown"

    if value >= 0.80:
        return "Extreme"

    if value >= 0.65:
        return "High"

    if value >= 0.40:
        return "Moderate"

    return "Low"


df["uncertainty_state"] = (
    df["uncertainty_score"]
    .apply(classify_uncertainty)
)


# ============================================================
# CURRENT STATE
# ============================================================

latest = df.iloc[-1]

latest_probability = latest["probability"]
latest_uncertainty = latest["uncertainty_score"]
latest_confidence = latest["confidence_score"]
latest_state = latest["uncertainty_state"]


# ============================================================
# EXECUTIVE STATE
# ============================================================

st.subheader(
    "Current Uncertainty State"
)


state_col1, state_col2, state_col3, state_col4 = st.columns(4)


with state_col1:

    st.metric(
        "Model",
        selected_label
    )


with state_col2:

    if pd.notna(latest_probability):

        st.metric(
            "Latest Probability",
            f"{latest_probability * 100:.1f}%"
        )

    else:

        st.metric(
            "Latest Probability",
            "N/A"
        )


with state_col3:

    st.metric(
        "Uncertainty",
        f"{latest_uncertainty:.3f}"
        if pd.notna(latest_uncertainty)
        else "N/A"
    )


with state_col4:

    st.metric(
        "Confidence",
        f"{latest_confidence * 100:.1f}%"
        if pd.notna(latest_confidence)
        else "N/A"
    )


# ============================================================
# STATE INTERPRETATION
# ============================================================

if latest_state == "Extreme":

    st.error(
        "EXTREME UNCERTAINTY — directional conviction is weak."
    )

elif latest_state == "High":

    st.warning(
        "HIGH UNCERTAINTY — forecast should be interpreted cautiously."
    )

elif latest_state == "Moderate":

    st.info(
        "MODERATE UNCERTAINTY — signal has limited but usable conviction."
    )

else:

    st.success(
        "LOW UNCERTAINTY — model currently shows stronger directional conviction."
    )


st.divider()


# ============================================================
# KPI ENGINE
# ============================================================

valid_unc = df[
    "uncertainty_score"
].dropna()

valid_conf = df[
    "confidence_score"
].dropna()


avg_uncertainty = (
    valid_unc.mean()
    if not valid_unc.empty
    else np.nan
)


median_uncertainty = (
    valid_unc.median()
    if not valid_unc.empty
    else np.nan
)


avg_confidence = (
    valid_conf.mean()
    if not valid_conf.empty
    else np.nan
)


high_uncertainty_count = int(
    df["high_uncertainty"]
    .sum()
)


high_uncertainty_pct = (
    high_uncertainty_count
    / len(df)
    if len(df) > 0
    else np.nan
)


if df["correct"].notna().any():

    accuracy_high_unc = (
        df.loc[
            df["high_uncertainty"],
            "correct"
        ].mean()
    )

    accuracy_low_unc = (
        df.loc[
            ~df["high_uncertainty"],
            "correct"
        ].mean()
    )

else:

    accuracy_high_unc = np.nan
    accuracy_low_unc = np.nan


st.subheader(
    "Uncertainty Intelligence"
)


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.metric(
        "Mean Uncertainty",
        safe_float(avg_uncertainty)
    )


with k2:

    st.metric(
        "Median Uncertainty",
        safe_float(median_uncertainty)
    )


with k3:

    st.metric(
        "Mean Confidence",
        safe_pct(avg_confidence)
    )


with k4:

    st.metric(
        "High-Uncertainty Share",
        safe_pct(high_uncertainty_pct)
    )


with k5:

    if pd.notna(accuracy_high_unc):

        st.metric(
            "Accuracy in High-Unc.",
            safe_pct(accuracy_high_unc)
        )

    else:

        st.metric(
            "Accuracy in High-Unc.",
            "N/A"
        )


# ============================================================
# PROBABILITY CONVICTION
# ============================================================

st.divider()

st.subheader(
    "Directional Conviction"
)

conviction_df = df[
    ["probability"]
].dropna().copy()


if not conviction_df.empty:

    conviction_df["confidence"] = (
        (conviction_df["probability"] - 0.5)
        .abs()
        * 2
    )

    st.bar_chart(
        conviction_df["confidence"],
        height=300
    )

    st.caption(
        "0 = no directional conviction | 1 = maximum directional conviction"
    )

else:

    st.info(
        "Probability data is unavailable for this model."
    )


# ============================================================
# UNCERTAINTY TIME SERIES
# ============================================================

st.subheader(
    "Uncertainty Regime Over Time"
)


if date_col:

    ts = df[
        [
            date_col,
            "uncertainty_score",
            "rolling_uncertainty",
        ]
    ].dropna()

    if not ts.empty:

        ts = ts.set_index(
            date_col
        )

        st.line_chart(
            ts,
            height=380
        )


# ============================================================
# CONFIDENCE TIME SERIES
# ============================================================

st.subheader(
    "Confidence Dynamics"
)


if date_col:

    confidence_ts = df[
        [
            date_col,
            "confidence_score",
            "rolling_confidence",
        ]
    ].dropna()

    if not confidence_ts.empty:

        confidence_ts = confidence_ts.set_index(
            date_col
        )

        st.line_chart(
            confidence_ts,
            height=350
        )


# ============================================================
# UNCERTAINTY DISTRIBUTION
# ============================================================

st.divider()

st.subheader(
    "Uncertainty Distribution"
)


distribution = df[
    "uncertainty_score"
].dropna()


if not distribution.empty:

    hist, bins = np.histogram(
        distribution,
        bins=20,
        range=(0, 1)
    )

    histogram_df = pd.DataFrame(
        {
            "Uncertainty Range": [
                f"{bins[i]:.2f}-{bins[i+1]:.2f}"
                for i in range(len(hist))
            ],
            "Observations": hist
        }
    )

    st.bar_chart(
        histogram_df.set_index(
            "Uncertainty Range"
        ),
        height=350
    )


# ============================================================
# UNCERTAINTY VS ACCURACY
# ============================================================

st.divider()

st.subheader(
    "Uncertainty vs Prediction Accuracy"
)


if df["correct"].notna().any():

    buckets = pd.cut(
        df["uncertainty_score"],
        bins=[
            -0.001,
            0.25,
            0.50,
            0.75,
            1.001
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Extreme"
        ]
    )

    accuracy_analysis = (
        df.assign(
            uncertainty_bucket=buckets
        )
        .groupby(
            "uncertainty_bucket",
            observed=False
        )
        .agg(
            Observations=("correct", "count"),
            Accuracy=("correct", "mean")
        )
        .reset_index()
    )

    accuracy_analysis[
        "Accuracy"
    ] *= 100

    st.dataframe(
        accuracy_analysis,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "Actual direction labels are not available for this prediction file."
    )


# ============================================================
# MODEL UNCERTAINTY COMPARISON
# ============================================================

st.divider()

st.subheader(
    "Cross-Model Uncertainty Comparison"
)


comparison_rows = []


for key, label in [
    ("bayesian", "Bayesian Ridge"),
    ("smc", "Sequential Monte Carlo"),
    ("chronos", "Chronos"),
    ("ensemble", "Ensemble"),
]:

    model_df = DATA[key]

    if model_df.empty:
        continue

    p_col = find_column(
        model_df,
        [
            "probability_positive",
            "probability_up",
            "prob_up",
            "probability"
        ]
    )

    if p_col is None:
        continue

    probs = (
        to_numeric(
            model_df[p_col]
        )
        .clip(0, 1)
        .dropna()
    )

    if probs.empty:
        continue

    unc = probability_uncertainty(
        probs
    )

    comparison_rows.append(
        {
            "Model": label,
            "Observations": len(probs),
            "Mean Probability": probs.mean(),
            "Mean Uncertainty": unc.mean(),
            "Mean Confidence": (
                1 - unc.mean()
            ),
            "Median Uncertainty": unc.median(),
        }
    )


if comparison_rows:

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    display_df = comparison_df.copy()

    display_df[
        "Mean Probability"
    ] *= 100

    display_df[
        "Mean Confidence"
    ] *= 100

    st.dataframe(
        display_df.style.format(
            {
                "Mean Probability": "{:.2f}%",
                "Mean Uncertainty": "{:.4f}",
                "Mean Confidence": "{:.2f}%",
                "Median Uncertainty": "{:.4f}",
            }
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# REGIME-AWARE UNCERTAINTY
# ============================================================

st.divider()

st.subheader(
    "Regime-Aware Uncertainty"
)


integrated = DATA["integrated"].copy()


if not integrated.empty:

    regime_col = find_column(
        integrated,
        [
            "regime",
            "regime_label",
            "regime_name",
            "causal_regime"
        ]
    )

    prob_integrated = find_column(
        integrated,
        [
            "probability_positive",
            "probability_up",
            "ensemble_probability",
            "ensemble_probability_up"
        ]
    )

    if (
        regime_col
        and prob_integrated
    ):

        regime_probs = (
            to_numeric(
                integrated[prob_integrated]
            )
            .clip(0, 1)
        )

        regime_table = pd.DataFrame(
            {
                "Regime": integrated[regime_col],
                "Probability": regime_probs
            }
        )

        regime_table[
            "Uncertainty"
        ] = probability_uncertainty(
            regime_table["Probability"]
        )

        regime_summary = (
            regime_table
            .groupby("Regime")
            .agg(
                Observations=("Uncertainty", "count"),
                Mean_Uncertainty=("Uncertainty", "mean"),
                Mean_Probability=("Probability", "mean"),
            )
            .reset_index()
        )

        regime_summary[
            "Mean_Confidence"
        ] = (
            1
            - regime_summary[
                "Mean_Uncertainty"
            ]
        )

        regime_summary[
            "Mean_Probability"
        ] *= 100

        regime_summary[
            "Mean_Confidence"
        ] *= 100

        st.dataframe(
            regime_summary.style.format(
                {
                    "Mean_Uncertainty": "{:.4f}",
                    "Mean_Probability": "{:.2f}%",
                    "Mean_Confidence": "{:.2f}%",
                }
            ),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Regime-integrated probability columns were not found."
        )

else:

    st.info(
        "Causal regime-integrated predictions are not available."
    )


# ============================================================
# HIGH UNCERTAINTY EVENTS
# ============================================================

st.divider()

st.subheader(
    "High-Uncertainty Event Monitor"
)


events = df[
    df["high_uncertainty"]
].copy()


if not events.empty:

    event_cols = []

    if date_col:
        event_cols.append(
            date_col
        )

    if pred_col:
        event_cols.append(
            pred_col
        )

    if prob_col:
        event_cols.append(
            prob_col
        )

    event_cols += [
        "uncertainty_score",
        "confidence_score",
        "uncertainty_state",
    ]

    if actual_col:
        event_cols.append(
            actual_col
        )

    event_cols = [
        c
        for c in event_cols
        if c in events.columns
    ]

    st.dataframe(
        events[
            event_cols
        ].tail(30),
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No high-uncertainty events detected under the current threshold."
    )


# ============================================================
# RECENT SIGNAL
# ============================================================

st.divider()

st.subheader(
    "Latest Prediction Diagnostics"
)


recent_cols = []

if date_col:
    recent_cols.append(
        date_col
    )

if pred_col:
    recent_cols.append(
        pred_col
    )

if prob_col:
    recent_cols.append(
        prob_col
    )

recent_cols += [
    "uncertainty_score",
    "confidence_score",
    "uncertainty_state",
]

if actual_col:
    recent_cols.append(
        actual_col
    )

recent_cols = [
    c
    for c in recent_cols
    if c in df.columns
]


st.dataframe(
    df[
        recent_cols
    ].tail(15),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# QUANTITATIVE INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "Quantitative Interpretation"
)


if latest_state == "Extreme":

    interpretation = """
    **Current state: Extreme uncertainty.**

    The directional probability is close to the neutral boundary or the
    predictive dispersion is elevated. This indicates weak model conviction.
    The forecast should not be interpreted as a strong directional signal.
    """

elif latest_state == "High":

    interpretation = """
    **Current state: High uncertainty.**

    The model has limited directional conviction. This is an important
    monitoring state because prediction errors can become more frequent when
    probability mass concentrates around the neutral decision boundary.
    """

elif latest_state == "Moderate":

    interpretation = """
    **Current state: Moderate uncertainty.**

    The model contains a usable directional signal, but the forecast does not
    exhibit strong conviction. Regime context and calibration should therefore
    be considered before interpreting the prediction.
    """

else:

    interpretation = """
    **Current state: Low uncertainty.**

    The model currently exhibits stronger directional conviction. This does
    not guarantee correctness; uncertainty is only one component of forecast
    quality and should be evaluated together with calibration and historical
    performance.
    """


st.info(
    interpretation
)


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander(
    "Methodology — How Uncertainty Is Calculated"
):

    st.markdown(
        """
        ### Probability-based uncertainty

        For directional probability `p`:

        `Uncertainty = 1 - 2 × |p - 0.5|`

        Therefore:

        - `p = 0.50` → maximum uncertainty
        - `p = 0.75` → moderate uncertainty
        - `p = 1.00` → maximum directional conviction
        - `p = 0.00` → maximum directional conviction

        ### Entropy

        Binary Shannon entropy is also calculated:

        `H(p) = -p log2(p) - (1-p) log2(1-p)`

        Entropy reaches its maximum at `p = 0.50`.

        ### Important research caveat

        Uncertainty is **not equivalent to financial risk**.

        A confident model prediction can still be wrong. Therefore this
        dashboard should be interpreted together with:

        - Brier score
        - calibration
        - walk-forward performance
        - regime-specific accuracy
        - statistical significance
        """
    )


# ============================================================
# DATA QUALITY
# ============================================================

st.divider()

st.subheader(
    "Data Quality & Diagnostics"
)


q1, q2, q3, q4 = st.columns(4)


with q1:

    st.metric(
        "Rows",
        f"{len(df):,}"
    )


with q2:

    st.metric(
        "Missing Probability",
        f"{df['probability'].isna().sum():,}"
    )


with q3:

    st.metric(
        "Missing Uncertainty",
        f"{df['uncertainty_score'].isna().sum():,}"
    )


with q4:

    if date_col:

        duplicate_dates = int(
            df[date_col]
            .duplicated()
            .sum()
        )

    else:

        duplicate_dates = 0

    st.metric(
        "Duplicate Dates",
        f"{duplicate_dates:,}"
    )


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "Raw Prediction Dataset"
):

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD
# ============================================================

download_df = df.copy()

csv_bytes = download_df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "Download Advanced Uncertainty Dataset",
    data=csv_bytes,
    file_name="advanced_uncertainty_monitor.csv",
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Bayesian Regime Detection Engine • Advanced Uncertainty Analytics • Step 19"
)