# ============================================================
# 04_MODEL_PERFORMANCE.PY
# BAYESIAN REGIME DETECTION ENGINE
# RESEARCH-GRADE MODEL EVALUATION TERMINAL
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Model Performance | Bayesian Regime Engine",
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

    .hero {
        padding: 12px 0 20px 0;
    }

    .hero-kicker {
        font-size: 11px;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        opacity: 0.55;
        font-weight: 700;
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 6px;
    }

    .hero-subtitle {
        font-size: 14px;
        opacity: 0.62;
        margin-top: 8px;
        max-width: 950px;
        line-height: 1.55;
    }

    .research-box {
        border: 1px solid rgba(128,128,128,0.22);
        border-radius: 14px;
        padding: 18px;
        background: rgba(128,128,128,0.035);
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

        # Evaluation
        "basic":
            load_csv(
                EVALUATION /
                "statistical_basic_performance.csv"
            ),

        "significance":
            load_csv(
                EVALUATION /
                "model_significance_tests.csv"
            ),

        "bootstrap":
            load_csv(
                EVALUATION /
                "bootstrap_confidence_intervals.csv"
            ),

        "regime":
            load_csv(
                EVALUATION /
                "statistical_regime_robustness.csv"
            ),

        "walk_summary":
            load_csv(
                EVALUATION /
                "walk_forward_summary.csv"
            ),

        "walk_yearly":
            load_csv(
                EVALUATION /
                "walk_forward_yearly.csv"
            ),

        # Prediction / backtest
        "walk_backtest":
            load_csv(
                PROCESSED /
                "walk_forward_backtest.csv"
            ),

        "integrated":
            load_csv(
                PROCESSED /
                "causal_regime_integrated_predictions.csv"
            ),

        "bayesian":
            load_csv(
                PROCESSED /
                "bayesian_predictions.csv"
            ),

        "smc":
            load_csv(
                PROCESSED /
                "smc_predictions.csv"
            ),

        "chronos":
            load_csv(
                PROCESSED /
                "chronos_predictions.csv"
            ),

        "ensemble":
            load_csv(
                PROCESSED /
                "ensemble_predictions.csv"
            ),

        "causal_regime":
            load_csv(
                PROCESSED /
                "nifty50_regimes_causal.csv"
            ),
    }


DATA = load_all_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):

    if df.empty:
        return None

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def format_pct(value):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value) * 100:.2f}%"


def format_num(value, decimals=4):

    if value is None or pd.isna(value):
        return "N/A"

    return f"{float(value):.{decimals}f}"


def best_model(
    df,
    metric,
    higher_is_better=True
):
    """
    Safely return the best model.

    IMPORTANT:
    If the complete metric is NaN,
    return None instead of crashing.
    """

    if df.empty or metric not in df.columns:
        return None

    temp = df[
        ["Model", metric]
    ].copy()

    temp[metric] = numeric(
        temp[metric]
    )

    temp = temp.dropna(
        subset=[metric]
    )

    if temp.empty:
        return None

    if higher_is_better:

        idx = temp[metric].idxmax()

    else:

        idx = temp[metric].idxmin()

    return temp.loc[idx]


def rank_higher(series):

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    valid = series.notna()

    if valid.sum() == 0:
        return result

    if valid.sum() == 1:

        result.loc[valid] = 1.0

        return result

    result.loc[valid] = (
        series[valid]
        .rank(
            method="average",
            pct=True
        )
    )

    return result


def rank_lower(series):

    result = pd.Series(
        np.nan,
        index=series.index,
        dtype=float
    )

    valid = series.notna()

    if valid.sum() == 0:
        return result

    if valid.sum() == 1:

        result.loc[valid] = 1.0

        return result

    ranks = (
        series[valid]
        .rank(
            method="average",
            pct=True
        )
    )

    result.loc[valid] = (
        1.0
        - ranks
        + (1.0 / valid.sum())
    )

    return result


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-kicker">
            QUANTITATIVE MODEL EVALUATION • CALIBRATION • ROBUSTNESS
        </div>

        <div class="hero-title">
            Model Performance Terminal
        </div>

        <div class="hero-subtitle">
            Research-grade evaluation of Bayesian, Sequential Monte Carlo,
            Chronos and Ensemble forecasting models using directional accuracy,
            probabilistic scoring, discrimination, statistical significance,
            bootstrap uncertainty, walk-forward stability and regime attribution.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Evaluation Controls"
    )

    page = st.radio(
        "Analysis Module",
        [
            "Executive Summary",
            "Model Leaderboard",
            "Metric Diagnostics",
            "Calibration Intelligence",
            "Statistical Significance",
            "Bootstrap Robustness",
            "Walk-Forward Stability",
            "Temporal Stability",
            "Regime Attribution",
            "Error Intelligence",
            "Prediction Diagnostics",
            "Research Conclusion",
        ]
    )

    st.divider()

    rolling_window = st.slider(
        "Rolling Window",
        min_value=10,
        max_value=120,
        value=30,
        step=5
    )

    confidence_level = st.selectbox(
        "Confidence Level",
        [
            "90%",
            "95%",
            "99%",
        ],
        index=1
    )


# ============================================================
# BASIC PERFORMANCE FILE
# ============================================================

basic = DATA["basic"].copy()


if basic.empty:

    st.error(
        "Performance evaluation file not found:"
    )

    st.code(
        str(
            EVALUATION /
            "statistical_basic_performance.csv"
        )
    )

    st.stop()


# ============================================================
# DETECT MODEL COLUMN
# ============================================================

model_col = find_column(
    basic,
    [
        "model",
        "model_name",
        "name"
    ]
)


if model_col is None:

    st.error(
        "Could not detect the Model column "
        "inside statistical_basic_performance.csv."
    )

    st.write(
        "Available columns:"
    )

    st.code(
        list(
            basic.columns
        )
    )

    st.stop()


# ============================================================
# DETECT METRICS
# ============================================================

accuracy_col = find_column(
    basic,
    [
        "accuracy",
        "acc"
    ]
)

brier_col = find_column(
    basic,
    [
        "brier_score",
        "brier",
        "brier score"
    ]
)

auc_col = find_column(
    basic,
    [
        "roc_auc",
        "roc-auc",
        "auc",
        "roc auc"
    ]
)

logloss_col = find_column(
    basic,
    [
        "log_loss",
        "logloss",
        "log loss"
    ]
)

mae_col = find_column(
    basic,
    [
        "mae",
        "mean_absolute_error"
    ]
)


# ============================================================
# NORMALIZED PERFORMANCE TABLE
# ============================================================

performance = pd.DataFrame()

performance["Model"] = (
    basic[model_col]
    .astype(str)
)


if accuracy_col:

    performance["Accuracy"] = numeric(
        basic[accuracy_col]
    )

else:

    performance["Accuracy"] = np.nan


if brier_col:

    performance["Brier Score"] = numeric(
        basic[brier_col]
    )

else:

    performance["Brier Score"] = np.nan


if auc_col:

    performance["ROC-AUC"] = numeric(
        basic[auc_col]
    )

else:

    performance["ROC-AUC"] = np.nan


if logloss_col:

    performance["Log Loss"] = numeric(
        basic[logloss_col]
    )

else:

    performance["Log Loss"] = np.nan


if mae_col:

    performance["MAE"] = numeric(
        basic[mae_col]
    )

else:

    performance["MAE"] = np.nan


# ============================================================
# MODEL RANKING ENGINE
# ============================================================

performance["Accuracy Rank"] = rank_higher(
    performance["Accuracy"]
)

performance["AUC Rank"] = rank_higher(
    performance["ROC-AUC"]
)

performance["Brier Rank"] = rank_lower(
    performance["Brier Score"]
)

performance["Log Loss Rank"] = rank_lower(
    performance["Log Loss"]
)


# MAE is optional and should not affect the
# core ranking because some models may not
# have continuous-return MAE.

performance["MAE Rank"] = rank_lower(
    performance["MAE"]
)


# ------------------------------------------------------------
# CORE COMPOSITE SCORE
# ------------------------------------------------------------

core_rank_columns = [
    "Accuracy Rank",
    "AUC Rank",
    "Brier Rank",
    "Log Loss Rank",
]


performance["Composite Score"] = (
    performance[
        core_rank_columns
    ]
    .mean(
        axis=1,
        skipna=True
    )
)


# ------------------------------------------------------------
# RANK
# ------------------------------------------------------------

performance["Rank"] = (
    performance[
        "Composite Score"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)


performance["Rank"] = (
    performance["Rank"]
    .astype("Int64")
)


leaderboard = (
    performance
    .sort_values(
        [
            "Composite Score",
            "Accuracy"
        ],
        ascending=[
            False,
            False
        ],
        na_position="last"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# PAGE 1
# EXECUTIVE SUMMARY
# ============================================================

if page == "Executive Summary":

    st.subheader(
        "Executive Model Intelligence"
    )

    # --------------------------------------------------------
    # SAFE WINNERS
    # --------------------------------------------------------

    accuracy_winner = best_model(
        performance,
        "Accuracy",
        True
    )

    brier_winner = best_model(
        performance,
        "Brier Score",
        False
    )

    auc_winner = best_model(
        performance,
        "ROC-AUC",
        True
    )

    logloss_winner = best_model(
        performance,
        "Log Loss",
        False
    )


    # --------------------------------------------------------
    # TOP COMPOSITE MODEL
    # --------------------------------------------------------

    valid_composite = leaderboard[
        leaderboard[
            "Composite Score"
        ].notna()
    ]

    if not valid_composite.empty:

        top_model = valid_composite.iloc[0]

        top_model_name = top_model[
            "Model"
        ]

        top_score = top_model[
            "Composite Score"
        ]

    else:

        top_model_name = "N/A"
        top_score = np.nan


    # --------------------------------------------------------
    # METRIC CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)


    with c1:

        st.metric(
            "Research Rank #1",
            top_model_name
        )

        if pd.notna(top_score):

            st.caption(
                f"Composite: {top_score:.4f}"
            )


    with c2:

        if accuracy_winner is not None:

            st.metric(
                "Best Accuracy",
                accuracy_winner["Model"]
            )

            st.caption(
                format_pct(
                    accuracy_winner[
                        "Accuracy"
                    ]
                )
            )

        else:

            st.metric(
                "Best Accuracy",
                "N/A"
            )


    with c3:

        if brier_winner is not None:

            st.metric(
                "Best Brier",
                brier_winner["Model"]
            )

            st.caption(
                format_num(
                    brier_winner[
                        "Brier Score"
                    ]
                )
            )

        else:

            st.metric(
                "Best Brier",
                "N/A"
            )


    with c4:

        if auc_winner is not None:

            st.metric(
                "Best ROC-AUC",
                auc_winner["Model"]
            )

            st.caption(
                format_num(
                    auc_winner[
                        "ROC-AUC"
                    ]
                )
            )

        else:

            st.metric(
                "Best ROC-AUC",
                "N/A"
            )


    with c5:

        st.metric(
            "Models Evaluated",
            len(performance)
        )


    # --------------------------------------------------------
    # SCORECARD
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Research Scorecard"
    )

    scorecard = leaderboard[
        [
            "Rank",
            "Model",
            "Accuracy",
            "Brier Score",
            "ROC-AUC",
            "Log Loss",
            "MAE",
            "Composite Score"
        ]
    ].copy()


    st.dataframe(
        scorecard.style.format(
            {
                "Accuracy": "{:.4f}",
                "Brier Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "Log Loss": "{:.4f}",
                "MAE": "{:.6f}",
                "Composite Score": "{:.4f}",
            },
            na_rep="N/A"
        ),
        use_container_width=True,
        hide_index=True
    )


    # --------------------------------------------------------
    # RANKING CHART
    # --------------------------------------------------------

    st.subheader(
        "Composite Research Ranking"
    )

    chart_df = leaderboard[
        [
            "Model",
            "Composite Score"
        ]
    ].dropna()


    if not chart_df.empty:

        chart_df = chart_df.set_index(
            "Model"
        )

        st.bar_chart(
            chart_df,
            height=380
        )

    else:

        st.info(
            "Composite ranking cannot be displayed because "
            "no valid evaluation metrics are available."
        )


# ============================================================
# PAGE 2
# MODEL LEADERBOARD
# ============================================================

elif page == "Model Leaderboard":

    st.subheader(
        "Quantitative Model Leaderboard"
    )

    display = leaderboard[
        [
            "Rank",
            "Model",
            "Accuracy",
            "Brier Score",
            "ROC-AUC",
            "Log Loss",
            "MAE",
            "Composite Score"
        ]
    ].copy()


    st.dataframe(
        display.style.format(
            {
                "Accuracy": "{:.4f}",
                "Brier Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "Log Loss": "{:.4f}",
                "MAE": "{:.6f}",
                "Composite Score": "{:.4f}",
            },
            na_rep="N/A"
        ),
        use_container_width=True,
        hide_index=True
    )


    st.divider()

    st.subheader(
        "Metric Winners"
    )


    a, b, c, d = st.columns(4)


    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    with a:

        winner = best_model(
            performance,
            "Accuracy",
            True
        )

        if winner is not None:

            st.metric(
                "Accuracy Winner",
                winner["Model"],
                format_pct(
                    winner["Accuracy"]
                )
            )

        else:

            st.metric(
                "Accuracy Winner",
                "N/A"
            )

            st.caption(
                "No valid Accuracy values"
            )


    # --------------------------------------------------------
    # BRIER
    # --------------------------------------------------------

    with b:

        winner = best_model(
            performance,
            "Brier Score",
            False
        )

        if winner is not None:

            st.metric(
                "Brier Winner",
                winner["Model"],
                format_num(
                    winner["Brier Score"]
                )
            )

        else:

            st.metric(
                "Brier Winner",
                "N/A"
            )

            st.caption(
                "No valid Brier values"
            )


    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    with c:

        winner = best_model(
            performance,
            "ROC-AUC",
            True
        )

        if winner is not None:

            st.metric(
                "AUC Winner",
                winner["Model"],
                format_num(
                    winner["ROC-AUC"]
                )
            )

        else:

            st.metric(
                "AUC Winner",
                "N/A"
            )

            st.caption(
                "ROC-AUC unavailable"
            )


    # --------------------------------------------------------
    # LOG LOSS
    # --------------------------------------------------------

    with d:

        winner = best_model(
            performance,
            "Log Loss",
            False
        )

        if winner is not None:

            st.metric(
                "Log Loss Winner",
                winner["Model"],
                format_num(
                    winner["Log Loss"]
                )
            )

        else:

            st.metric(
                "Log Loss Winner",
                "N/A"
            )

            st.caption(
                "No valid Log Loss values"
            )


# ============================================================
# PAGE 3
# METRIC DIAGNOSTICS
# ============================================================

elif page == "Metric Diagnostics":

    st.subheader(
        "Metric-Level Diagnostics"
    )


    metric = st.selectbox(
        "Select Evaluation Metric",
        [
            "Accuracy",
            "Brier Score",
            "ROC-AUC",
            "Log Loss",
            "MAE",
            "Composite Score"
        ]
    )


    metric_df = leaderboard[
        [
            "Model",
            metric
        ]
    ].copy()


    metric_df[metric] = numeric(
        metric_df[metric]
    )


    metric_df = metric_df.dropna(
        subset=[metric]
    )


    if metric_df.empty:

        st.warning(
            f"No valid values are available for {metric}."
        )

    else:

        st.bar_chart(
            metric_df.set_index(
                "Model"
            ),
            height=420
        )


        st.dataframe(
            metric_df.style.format(
                {
                    metric: "{:.6f}"
                }
            ),
            use_container_width=True,
            hide_index=True
        )


    st.info(
        """
        Accuracy and ROC-AUC are maximized.

        Brier Score and Log Loss are minimized.

        MAE is minimized when continuous-return prediction data is available.

        No single metric should be treated as the complete definition of
        model quality.
        """
    )


# ============================================================
# PAGE 4
# CALIBRATION
# ============================================================

elif page == "Calibration Intelligence":

    st.subheader(
        "Probabilistic Calibration Intelligence"
    )


    calibration_path = (
        EVALUATION /
        "calibration_curve.png"
    )


    brier_path = (
        EVALUATION /
        "brier_score_comparison.png"
    )


    if calibration_path.exists():

        st.image(
            str(calibration_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Calibration curve image not found."
        )


    if brier_path.exists():

        st.image(
            str(brier_path),
            use_container_width=True
        )

    else:

        st.warning(
            "Brier score comparison image not found."
        )


    st.divider()


    calibration_table = leaderboard[
        [
            "Model",
            "Brier Score",
            "Log Loss"
        ]
    ].copy()


    st.dataframe(
        calibration_table.style.format(
            {
                "Brier Score": "{:.4f}",
                "Log Loss": "{:.4f}"
            },
            na_rep="N/A"
        ),
        use_container_width=True,
        hide_index=True
    )


    st.info(
        """
        Calibration measures whether predicted probabilities correspond
        reasonably to observed frequencies.

        Brier Score and Log Loss are therefore essential for this project
        because the system is explicitly uncertainty-aware.
        """
    )


# ============================================================
# PAGE 5
# STATISTICAL SIGNIFICANCE
# ============================================================

elif page == "Statistical Significance":

    st.subheader(
        "Pairwise Statistical Significance"
    )


    significance = DATA[
        "significance"
    ].copy()


    if significance.empty:

        st.warning(
            "model_significance_tests.csv was not found."
        )

    else:

        st.dataframe(
            significance,
            use_container_width=True,
            hide_index=True
        )


        p_col = find_column(
            significance,
            [
                "p_value",
                "p-value",
                "pvalue"
            ]
        )


        if p_col:

            p_values = numeric(
                significance[p_col]
            ).dropna()


            significant = (
                p_values < 0.05
            )


            c1, c2, c3 = st.columns(3)


            with c1:

                st.metric(
                    "Pairwise Tests",
                    len(p_values)
                )


            with c2:

                st.metric(
                    "p < 0.05",
                    int(
                        significant.sum()
                    )
                )


            with c3:

                st.metric(
                    "p ≥ 0.05",
                    int(
                        (~significant).sum()
                    )
                )


            if (
                len(p_values) > 0
                and
                significant.sum() == 0
            ):

                st.success(
                    "No pairwise comparison establishes statistically "
                    "significant superiority at the 5% level."
                )

            elif significant.sum() > 0:

                st.warning(
                    "At least one comparison crosses the 5% significance "
                    "threshold. Inspect effect direction and methodology."
                )

            else:

                st.info(
                    "No valid p-values were available."
                )


# ============================================================
# PAGE 6
# BOOTSTRAP ROBUSTNESS
# ============================================================

elif page == "Bootstrap Robustness":

    st.subheader(
        "Bootstrap Confidence Intervals"
    )


    bootstrap = DATA[
        "bootstrap"
    ].copy()


    if bootstrap.empty:

        st.warning(
            "bootstrap_confidence_intervals.csv was not found."
        )

    else:

        st.dataframe(
            bootstrap,
            use_container_width=True,
            hide_index=True
        )


        st.info(
            f"""
            Selected confidence level: **{confidence_level}**

            Bootstrap intervals describe sampling uncertainty around observed
            performance. They should be interpreted together with paired
            statistical tests and walk-forward evaluation.
            """
        )


# ============================================================
# PAGE 7
# WALK-FORWARD STABILITY
# ============================================================

elif page == "Walk-Forward Stability":

    st.subheader(
        "Chronological Walk-Forward Stability"
    )


    wf = DATA[
        "walk_backtest"
    ].copy()


    if wf.empty:

        st.warning(
            "walk_forward_backtest.csv was not found."
        )

    else:

        date_col = find_column(
            wf,
            [
                "forecast_for_date",
                "prediction_for_date",
                "date"
            ]
        )


        if date_col:

            wf[date_col] = pd.to_datetime(
                wf[date_col],
                errors="coerce"
            )

            wf = wf.sort_values(
                date_col
            )


        correct_columns = [
            c
            for c in wf.columns
            if "correct" in str(c).lower()
        ]


        if (
            date_col
            and correct_columns
        ):

            rolling = pd.DataFrame(
                index=wf[date_col]
            )


            for col in correct_columns:

                model_name = (
                    str(col)
                    .replace(
                        "_correct",
                        ""
                    )
                    .replace(
                        "_is_correct",
                        ""
                    )
                    .replace(
                        "correct_",
                        ""
                    )
                    .title()
                )


                values = numeric(
                    wf[col]
                )


                rolling[
                    model_name
                ] = (
                    values
                    .rolling(
                        rolling_window,
                        min_periods=5
                    )
                    .mean()
                )


            st.line_chart(
                rolling,
                height=420
            )


            st.caption(
                f"{rolling_window}-observation rolling accuracy."
            )


        else:

            st.info(
                "Correctness/date columns were not available "
                "for rolling analysis."
            )


        st.divider()


        st.subheader(
            "Recent Walk-Forward Observations"
        )


        st.dataframe(
            wf.tail(30),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 8
# TEMPORAL STABILITY
# ============================================================

elif page == "Temporal Stability":

    st.subheader(
        "Year-by-Year Performance Stability"
    )


    yearly = DATA[
        "walk_yearly"
    ].copy()


    if yearly.empty:

        st.warning(
            "walk_forward_yearly.csv was not found."
        )

    else:

        st.dataframe(
            yearly,
            use_container_width=True,
            hide_index=True
        )


        year_col = find_column(
            yearly,
            [
                "year",
                "Year"
            ]
        )


        accuracy_cols = [
            c
            for c in yearly.columns
            if "accuracy"
            in str(c).lower()
        ]


        if (
            year_col
            and accuracy_cols
        ):

            temporal = yearly[
                [
                    year_col
                ] + accuracy_cols
            ].copy()


            temporal = temporal.set_index(
                year_col
            )


            st.line_chart(
                temporal,
                height=420
            )


        else:

            st.info(
                "Year or accuracy columns were not detected."
            )


# ============================================================
# PAGE 9
# REGIME ATTRIBUTION
# ============================================================

elif page == "Regime Attribution":

    st.subheader(
        "Regime-Conditional Model Performance"
    )


    regime = DATA[
        "regime"
    ].copy()


    if regime.empty:

        st.warning(
            "statistical_regime_robustness.csv was not found."
        )

    else:

        st.dataframe(
            regime,
            use_container_width=True,
            hide_index=True
        )


        regime_col = find_column(
            regime,
            [
                "regime",
                "regime_label",
                "state"
            ]
        )


        numeric_cols = (
            regime
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )


        if (
            regime_col
            and numeric_cols
        ):

            selected_metric = st.selectbox(
                "Select Regime Metric",
                numeric_cols
            )


            plot_df = regime[
                [
                    regime_col,
                    selected_metric
                ]
            ].dropna()


            if not plot_df.empty:

                st.bar_chart(
                    plot_df.set_index(
                        regime_col
                    ),
                    height=400
                )

            else:

                st.info(
                    "No valid values for selected metric."
                )


        st.info(
            """
            Regime attribution answers:

            **When does each model work best?**

            Aggregate accuracy can hide significant differences between
            high-volatility, medium-volatility and low-volatility states.
            """
        )


# ============================================================
# PAGE 10
# ERROR INTELLIGENCE
# ============================================================

elif page == "Error Intelligence":

    st.subheader(
        "Prediction Error Intelligence"
    )


    backtest = DATA[
        "walk_backtest"
    ].copy()


    if backtest.empty:

        st.warning(
            "walk_forward_backtest.csv was not found."
        )

    else:

        correct_columns = [
            c
            for c in backtest.columns
            if "correct" in str(c).lower()
        ]


        error_rows = []


        for col in correct_columns:

            values = numeric(
                backtest[col]
            ).dropna()


            if values.empty:
                continue


            accuracy = values.mean()


            error_rows.append(
                {
                    "Model":
                        str(col)
                        .replace(
                            "_correct",
                            ""
                        )
                        .replace(
                            "_is_correct",
                            ""
                        )
                        .title(),

                    "Observations":
                        len(values),

                    "Correct":
                        int(
                            values.sum()
                        ),

                    "Errors":
                        int(
                            len(values)
                            -
                            values.sum()
                        ),

                    "Accuracy":
                        accuracy,

                    "Error Rate":
                        1 - accuracy,
                }
            )


        if error_rows:

            error_df = pd.DataFrame(
                error_rows
            )


            st.dataframe(
                error_df.style.format(
                    {
                        "Accuracy": "{:.2%}",
                        "Error Rate": "{:.2%}",
                    }
                ),
                use_container_width=True,
                hide_index=True
            )


            st.subheader(
                "Error Rate Comparison"
            )


            st.bar_chart(
                error_df.set_index(
                    "Model"
                )[
                    "Error Rate"
                ],
                height=350
            )


        else:

            st.info(
                "No correctness columns were detected."
            )


# ============================================================
# PAGE 11
# PREDICTION DIAGNOSTICS
# ============================================================

elif page == "Prediction Diagnostics":

    st.subheader(
        "Latest Prediction Diagnostics"
    )


    model_options = {
        "Bayesian Ridge":
            "bayesian",

        "Sequential Monte Carlo":
            "smc",

        "Chronos":
            "chronos",

        "Ensemble":
            "ensemble",
    }


    available_models = [
        name
        for name, key in model_options.items()
        if not DATA[key].empty
    ]


    if not available_models:

        st.warning(
            "No model prediction files were found."
        )

    else:

        selected_model = st.selectbox(
            "Prediction Model",
            available_models
        )


        prediction_df = DATA[
            model_options[
                selected_model
            ]
        ].copy()


        date_col = find_column(
            prediction_df,
            [
                "forecast_for_date",
                "prediction_for_date",
                "date",
                "origin_date"
            ]
        )


        if date_col:

            prediction_df[date_col] = pd.to_datetime(
                prediction_df[date_col],
                errors="coerce"
            )

            prediction_df = (
                prediction_df
                .sort_values(
                    date_col
                )
            )


        probability_col = find_column(
            prediction_df,
            [
                "probability_positive",
                "probability_up",
                "prob_up"
            ]
        )


        prediction_col = find_column(
            prediction_df,
            [
                "predicted_direction",
                "predicted_direction_label"
            ]
        )


        actual_col = find_column(
            prediction_df,
            [
                "actual_direction",
                "actual_next_day_direction"
            ]
        )


        latest = prediction_df.tail(
            1
        )


        if not latest.empty:

            c1, c2, c3 = st.columns(3)


            with c1:

                if prediction_col:

                    st.metric(
                        "Latest Direction",
                        str(
                            latest.iloc[0][
                                prediction_col
                            ]
                        )
                    )

                else:

                    st.metric(
                        "Latest Direction",
                        "N/A"
                    )


            with c2:

                if probability_col:

                    probability = numeric(
                        latest.iloc[0][
                            probability_col
                        ]
                    )

                    if pd.notna(
                        probability
                    ):

                        st.metric(
                            "Latest Probability",
                            f"{probability:.2%}"
                        )

                    else:

                        st.metric(
                            "Latest Probability",
                            "N/A"
                        )

                else:

                    st.metric(
                        "Latest Probability",
                        "N/A"
                    )


            with c3:

                if actual_col:

                    st.metric(
                        "Actual Direction",
                        str(
                            latest.iloc[0][
                                actual_col
                            ]
                        )
                    )

                else:

                    st.metric(
                        "Actual Direction",
                        "N/A"
                    )


        st.divider()


        st.dataframe(
            prediction_df.tail(25),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PAGE 12
# RESEARCH CONCLUSION
# ============================================================

elif page == "Research Conclusion":

    st.subheader(
        "Research-Grade Model Assessment"
    )


    accuracy_winner = best_model(
        performance,
        "Accuracy",
        True
    )


    brier_winner = best_model(
        performance,
        "Brier Score",
        False
    )


    auc_winner = best_model(
        performance,
        "ROC-AUC",
        True
    )


    logloss_winner = best_model(
        performance,
        "Log Loss",
        False
    )


    accuracy_name = (
        accuracy_winner["Model"]
        if accuracy_winner is not None
        else "Not available"
    )


    brier_name = (
        brier_winner["Model"]
        if brier_winner is not None
        else "Not available"
    )


    auc_name = (
        auc_winner["Model"]
        if auc_winner is not None
        else "Not available"
    )


    logloss_name = (
        logloss_winner["Model"]
        if logloss_winner is not None
        else "Not available"
    )


    st.markdown(
        f"""
        ### Evidence-Based Assessment

        **Directional accuracy leader:**  
        `{accuracy_name}`

        **Probabilistic scoring leader:**  
        `{brier_name}`

        **Discrimination leader:**  
        `{auc_name}`

        **Log-loss leader:**  
        `{logloss_name}`

        These metrics represent different dimensions of forecast quality.
        Therefore, the model with the highest directional accuracy should
        not automatically be declared the best probabilistic model.

        The final research assessment should combine:

        - Directional accuracy
        - Brier Score
        - ROC-AUC
        - Log Loss
        - Calibration
        - Bootstrap confidence intervals
        - Pairwise statistical tests
        - Walk-forward stability
        - Temporal stability
        - Regime-specific performance
        """
    )


    st.divider()


    # --------------------------------------------------------
    # STATISTICAL CONCLUSION
    # --------------------------------------------------------

    significance = DATA[
        "significance"
    ].copy()


    if not significance.empty:

        p_col = find_column(
            significance,
            [
                "p_value",
                "p-value",
                "pvalue"
            ]
        )


        if p_col:

            p_values = numeric(
                significance[p_col]
            ).dropna()


            if not p_values.empty:

                significant_count = int(
                    (
                        p_values < 0.05
                    ).sum()
                )


                if significant_count == 0:

                    st.warning(
                        """
                        ### Statistical Evidence

                        The available pairwise significance tests do not
                        establish statistically significant superiority at
                        the 5% level.

                        Therefore, the defensible research conclusion is:

                        **Observed performance differences exist, but the
                        current statistical evidence is insufficient to claim
                        that one model is significantly superior to another.**
                        """
                    )

                else:

                    st.info(
                        f"""
                        {significant_count} pairwise comparison(s) cross the
                        5% significance threshold.

                        These results should be interpreted together with
                        effect direction, bootstrap intervals and
                        walk-forward stability.
                        """
                    )


    # --------------------------------------------------------
    # PROJECT-SPECIFIC INTERPRETATION
    # --------------------------------------------------------

    st.divider()


    st.subheader(
        "Project Interpretation"
    )


    st.info(
        """
        The purpose of this project is not simply to maximize directional
        accuracy.

        The system is designed to combine:

        **Bayesian forecasting + Sequential Monte Carlo inference +
        causal regime detection + Chronos benchmarking + Ensemble
        forecasting + uncertainty quantification.**

        Therefore the strongest model should be evaluated across multiple
        dimensions rather than using a single accuracy number.
        """
    )


# ============================================================
# DATA QUALITY
# ============================================================

st.divider()


with st.expander(
    "Evaluation Data Quality"
):

    q1, q2, q3, q4 = st.columns(4)


    with q1:

        st.metric(
            "Models",
            len(performance)
        )


    with q2:

        st.metric(
            "Evaluation Rows",
            len(basic)
        )


    with q3:

        st.metric(
            "Missing Accuracy",
            int(
                performance[
                    "Accuracy"
                ].isna().sum()
            )
        )


    with q4:

        st.metric(
            "Missing ROC-AUC",
            int(
                performance[
                    "ROC-AUC"
                ].isna().sum()
            )
        )


# ============================================================
# RAW DATA
# ============================================================

with st.expander(
    "Raw Performance Evaluation Data"
):

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DOWNLOAD
# ============================================================

download_df = leaderboard.copy()


csv_data = download_df.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="Download Advanced Performance Table",
    data=csv_data,
    file_name="research_model_performance.csv",
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Bayesian Regime Detection Engine • Research-Grade Model Evaluation • Step 19"
)