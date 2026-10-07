# ============================================================
# STEP 19 — RESEARCH-GRADE BACKTESTING DASHBOARD
# Bayesian Regime Detection Engine
# ============================================================

import os
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Backtesting | Bayesian Regime Engine",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

DATA = os.path.join(ROOT, "data", "processed")
REPORTS = os.path.join(ROOT, "reports", "evaluation")


# ============================================================
# FILE PATHS
# ============================================================

FILES = {
    "walk_forward": os.path.join(
        DATA, "walk_forward_backtest.csv"
    ),
    "integrated": os.path.join(
        DATA, "causal_regime_integrated_predictions.csv"
    ),
    "regimes": os.path.join(
        DATA, "nifty50_regimes_causal.csv"
    ),
    "basic": os.path.join(
        REPORTS, "statistical_basic_performance.csv"
    ),
    "yearly": os.path.join(
        REPORTS, "walk_forward_yearly.csv"
    ),
    "summary": os.path.join(
        REPORTS, "walk_forward_summary.csv"
    ),
    "significance": os.path.join(
        REPORTS, "model_significance_tests.csv"
    ),
    "bootstrap": os.path.join(
        REPORTS, "bootstrap_confidence_intervals.csv"
    ),
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .hero {
        padding: 28px;
        border-radius: 18px;
        margin-bottom: 25px;
        background:
        linear-gradient(
            135deg,
            rgba(30,41,59,0.98),
            rgba(15,23,42,0.98)
        );
        border: 1px solid rgba(148,163,184,0.20);
    }

    .hero-title {
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }

    .metric-card {
        padding: 18px;
        border-radius: 14px;
        background: rgba(30,41,59,0.55);
        border: 1px solid rgba(148,163,184,0.15);
        min-height: 110px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
    }

    .metric-value {
        font-size: 25px;
        font-weight: 800;
        margin-top: 7px;
    }

    .section {
        margin-top: 28px;
        margin-bottom: 12px;
    }

    .info-box {
        padding: 16px;
        border-radius: 12px;
        background: rgba(30,41,59,0.45);
        border: 1px solid rgba(148,163,184,0.15);
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .success-box {
        padding: 16px;
        border-radius: 12px;
        background: rgba(22,101,52,0.18);
        border: 1px solid rgba(74,222,128,0.25);
    }

    .warning-box {
        padding: 16px;
        border-radius: 12px;
        background: rgba(146,64,14,0.18);
        border: 1px solid rgba(251,191,36,0.25);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

@st.cache_data
def load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


def find_column(df, keywords):
    if df.empty:
        return None

    for col in df.columns:
        low = str(col).lower()

        for keyword in keywords:
            if keyword in low:
                return col

    return None


def format_pct(value, decimals=2):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value * 100:.{decimals}f}%"


def format_num(value, decimals=4):
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}"


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def detect_date_column(df):
    candidates = [
        "forecast_for_date",
        "prediction_for_date",
        "date",
        "origin_date",
        "feature_date",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def detect_actual_direction(df):
    candidates = [
        "actual_next_day_direction",
        "actual_direction",
        "actual_direction_label",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def detect_actual_return(df):
    candidates = [
        "actual_next_day_return",
        "daily_return",
        "actual_return",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    return None


def detect_prediction_column(df, model):
    model = model.lower()

    candidates = []

    if model == "bayesian":
        candidates = [
            "bayesian_predicted_direction",
            "bayesian_prediction",
            "bayesian_direction",
            "predicted_direction_bayesian",
        ]

    elif model == "smc":
        candidates = [
            "smc_predicted_direction",
            "smc_prediction",
            "smc_direction",
            "predicted_direction_smc",
        ]

    elif model == "ensemble":
        candidates = [
            "ensemble_predicted_direction",
            "ensemble_prediction",
            "ensemble_direction",
            "predicted_direction_ensemble",
        ]

    elif model == "chronos":
        candidates = [
            "chronos_predicted_direction",
            "chronos_prediction",
            "chronos_direction",
            "predicted_direction_chronos",
        ]

    for c in candidates:
        if c in df.columns:
            return c

    for c in df.columns:
        low = c.lower()

        if model in low and "pred" in low and "direction" in low:
            return c

    return None


def detect_correct_column(df, model):
    model = model.lower()

    for c in df.columns:
        low = c.lower()

        if model in low and "correct" in low:
            return c

    return None


def get_models(df):
    models = []

    for model in ["Bayesian", "SMC", "Chronos", "Ensemble"]:
        correct_col = detect_correct_column(df, model)

        if correct_col:
            models.append(model)

    return models


def calculate_brier(actual, probability):
    actual = numeric(actual)
    probability = numeric(probability)

    mask = actual.notna() & probability.notna()

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        (probability[mask] - actual[mask]) ** 2
    )


def annualized_sharpe(returns):
    returns = numeric(returns).dropna()

    if len(returns) < 2:
        return np.nan

    std = returns.std()

    if std == 0 or pd.isna(std):
        return np.nan

    return (
        returns.mean() / std
    ) * np.sqrt(252)


def max_drawdown(returns):
    returns = numeric(returns).dropna()

    if returns.empty:
        return np.nan

    equity = (1 + returns).cumprod()
    peak = equity.cummax()

    drawdown = equity / peak - 1

    return drawdown.min()


# ============================================================
# LOAD DATA
# ============================================================

wf = load_csv(FILES["walk_forward"])
integrated = load_csv(FILES["integrated"])
regimes = load_csv(FILES["regimes"])

basic = load_csv(FILES["basic"])
yearly = load_csv(FILES["yearly"])
summary = load_csv(FILES["summary"])
significance = load_csv(FILES["significance"])
bootstrap = load_csv(FILES["bootstrap"])


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            📈 Research-Grade Walk-Forward Backtesting
        </div>
        <div class="hero-subtitle">
            Chronological forecast validation, model stability,
            regime attribution, statistical robustness and error intelligence
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA CHECK
# ============================================================

if wf.empty:
    st.error(
        "walk_forward_backtest.csv not found or contains no data."
    )

    st.stop()


date_col = detect_date_column(wf)
actual_direction_col = detect_actual_direction(wf)
actual_return_col = detect_actual_return(wf)

if date_col:
    wf[date_col] = pd.to_datetime(
        wf[date_col],
        errors="coerce"
    )

    wf = wf.sort_values(date_col).reset_index(drop=True)


models = get_models(wf)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Backtesting")

section = st.sidebar.radio(
    "Research Module",
    [
        "Executive Backtest",
        "Walk-Forward Validation",
        "Model Performance",
        "Rolling Skill",
        "Cumulative Forecast Skill",
        "Yearly Stability",
        "Regime Attribution",
        "Confusion Matrix",
        "Prediction Streaks",
        "Model Agreement",
        "Error Intelligence",
        "Statistical Significance",
        "Bootstrap Robustness",
        "Trading Simulation",
        "Validation Audit",
        "Raw Data Explorer",
    ],
)


# ============================================================
# EXECUTIVE BACKTEST
# ============================================================

if section == "Executive Backtest":

    st.subheader("Executive Backtest")

    total_obs = len(wf)

    start_date = (
        wf[date_col].min()
        if date_col
        else None
    )

    end_date = (
        wf[date_col].max()
        if date_col
        else None
    )

    duplicates = (
        wf[date_col].duplicated().sum()
        if date_col
        else 0
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        metric_card(
            "Forecast Observations",
            f"{total_obs:,}",
        )

    with c2:
        metric_card(
            "Models",
            len(models),
        )

    with c3:
        metric_card(
            "Start",
            str(start_date.date())
            if start_date is not None
            else "N/A",
        )

    with c4:
        metric_card(
            "End",
            str(end_date.date())
            if end_date is not None
            else "N/A",
        )

    with c5:
        metric_card(
            "Duplicate Dates",
            int(duplicates),
        )

    st.markdown("### Model Snapshot")

    rows = []

    for model in models:

        correct_col = detect_correct_column(
            wf,
            model
        )

        if correct_col:

            correct = numeric(
                wf[correct_col]
            )

            accuracy = correct.mean()

            observations = correct.notna().sum()

            rows.append(
                {
                    "Model": model,
                    "Observations": observations,
                    "Accuracy": accuracy,
                    "Error Rate": 1 - accuracy,
                }
            )

    performance = pd.DataFrame(rows)

    if not performance.empty:

        performance["Accuracy"] = (
            performance["Accuracy"] * 100
        ).round(2)

        performance["Error Rate"] = (
            performance["Error Rate"] * 100
        ).round(2)

        st.dataframe(
            performance,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Research Interpretation")

    st.info(
        """
        This dashboard evaluates the saved one-step-ahead predictions
        in chronological order.

        It is a walk-forward evaluation of stored predictions.
        It should NOT be interpreted as a complete rolling model-refit
        backtest unless the underlying models were actually retrained
        at every historical forecast origin.
        """
    )


# ============================================================
# WALK-FORWARD VALIDATION
# ============================================================

elif section == "Walk-Forward Validation":

    st.subheader("🔄 Walk-Forward Validation")

    if date_col:

        dates = wf[date_col].dropna()

        increasing = dates.is_monotonic_increasing
        duplicates = dates.duplicated().sum()

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Chronological Order",
                "PASS" if increasing else "FAIL",
            )

        with c2:
            metric_card(
                "Duplicate Dates",
                int(duplicates),
            )

        with c3:
            metric_card(
                "Observations",
                f"{len(wf):,}",
            )

        if increasing and duplicates == 0:
            st.success(
                "Chronological sequence validation passed."
            )
        else:
            st.warning(
                "Chronological sequence requires review."
            )

    st.markdown("### Forecast Sequence")

    if date_col:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=wf[date_col],
                y=np.arange(len(wf)),
                mode="lines",
                name="Forecast Sequence",
            )
        )

        fig.update_layout(
            title="Chronological Forecast Sequence",
            xaxis_title="Forecast Date",
            yaxis_title="Observation Index",
            height=450,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif section == "Model Performance":

    st.subheader("🏆 Model Performance")

    rows = []

    for model in models:

        col = detect_correct_column(
            wf,
            model
        )

        if not col:
            continue

        values = numeric(wf[col]).dropna()

        if values.empty:
            continue

        rows.append(
            {
                "Model": model,
                "Accuracy": values.mean(),
                "Error Rate": 1 - values.mean(),
                "Correct": int(values.sum()),
                "Incorrect": int(len(values) - values.sum()),
                "Observations": len(values),
            }
        )

    perf = pd.DataFrame(rows)

    if perf.empty:

        st.warning(
            "No model correctness columns were detected."
        )

    else:

        display = perf.copy()

        display["Accuracy"] = (
            display["Accuracy"] * 100
        ).round(2)

        display["Error Rate"] = (
            display["Error Rate"] * 100
        ).round(2)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            perf,
            x="Model",
            y="Accuracy",
            title="Model Accuracy",
            text=(
                perf["Accuracy"] * 100
            ).round(2),
        )

        fig.update_yaxes(
            tickformat=".0%",
            range=[
                max(0, perf["Accuracy"].min() - 0.1),
                min(1, perf["Accuracy"].max() + 0.1),
            ],
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# ROLLING SKILL
# ============================================================

elif section == "Rolling Skill":

    st.subheader("📉 Rolling Forecast Skill")

    if not models:

        st.warning("No model correctness columns available.")

    else:

        window = st.slider(
            "Rolling Window",
            min_value=20,
            max_value=250,
            value=60,
            step=10,
        )

        fig = go.Figure()

        for model in models:

            col = detect_correct_column(
                wf,
                model
            )

            if not col:
                continue

            series = numeric(
                wf[col]
            )

            rolling = (
                series
                .rolling(window)
                .mean()
            )

            fig.add_trace(
                go.Scatter(
                    x=wf[date_col],
                    y=rolling,
                    mode="lines",
                    name=model,
                )
            )

        fig.add_hline(
            y=0.5,
            line_dash="dash",
            annotation_text="50% Baseline",
        )

        fig.update_layout(
            title=f"{window}-Observation Rolling Accuracy",
            yaxis_title="Accuracy",
            xaxis_title="Date",
            height=520,
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.info(
            """
            Rolling accuracy is useful for detecting temporal instability.
            A model that performs well overall but repeatedly falls below
            the 50% directional baseline may have unstable forecasting skill.
            """
        )

    st.markdown("### Rolling Brier Score")

    probability_columns = [
        c for c in wf.columns
        if "probability" in c.lower()
        or "probability_up" in c.lower()
    ]

    if probability_columns and actual_direction_col:

        actual = numeric(
            wf[actual_direction_col]
        )

        fig = go.Figure()

        for col in probability_columns:

            probability = numeric(
                wf[col]
            )

            rolling_brier = (
                (probability - actual) ** 2
            ).rolling(
                60
            ).mean()

            fig.add_trace(
                go.Scatter(
                    x=wf[date_col],
                    y=rolling_brier,
                    mode="lines",
                    name=col,
                )
            )

        fig.update_layout(
            title="Rolling 60-Observation Brier Score",
            xaxis_title="Date",
            yaxis_title="Brier Score",
            height=500,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info(
            "Probability and actual-direction columns are not available in the walk-forward file."
        )


# ============================================================
# CUMULATIVE FORECAST SKILL
# ============================================================

elif section == "Cumulative Forecast Skill":

    st.subheader("📈 Cumulative Forecast Skill")

    st.caption(
        "This is a cumulative correctness-based research diagnostic, "
        "not a financial equity curve."
    )

    fig = go.Figure()

    for model in models:

        col = detect_correct_column(
            wf,
            model
        )

        if not col:
            continue

        correct = numeric(
            wf[col]
        ).fillna(0)

        cumulative_accuracy = (
            correct.cumsum()
            /
            np.arange(1, len(correct) + 1)
        )

        fig.add_trace(
            go.Scatter(
                x=wf[date_col],
                y=cumulative_accuracy,
                mode="lines",
                name=model,
            )
        )

    fig.add_hline(
        y=0.5,
        line_dash="dash",
        annotation_text="Directional Baseline",
    )

    fig.update_layout(
        title="Cumulative Directional Accuracy",
        xaxis_title="Date",
        yaxis_title="Cumulative Accuracy",
        height=520,
    )

    fig.update_yaxes(
        tickformat=".0%"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# YEARLY STABILITY
# ============================================================

elif section == "Yearly Stability":

    st.subheader("📅 Yearly Forecast Stability")

    if yearly.empty:

        st.warning(
            "walk_forward_yearly.csv not found."
        )

    else:

        st.dataframe(
            yearly,
            use_container_width=True,
            hide_index=True,
        )

        numeric_cols = yearly.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_cols:

            selected_metric = st.selectbox(
                "Metric",
                numeric_cols,
            )

            year_col = None

            for c in yearly.columns:

                if "year" in c.lower():

                    year_col = c
                    break

            if year_col:

                fig = px.line(
                    yearly,
                    x=year_col,
                    y=selected_metric,
                    markers=True,
                    title=f"Yearly {selected_metric}",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


# ============================================================
# REGIME ATTRIBUTION
# ============================================================

elif section == "Regime Attribution":

    st.subheader("🌡️ Regime-Wise Forecast Performance")

    if integrated.empty:

        st.warning(
            "Causal regime integrated prediction file not found."
        )

    else:

        regime_col = find_column(
            integrated,
            ["regime"]
        )

        if not regime_col:

            st.warning(
                "No regime column detected."
            )

        else:

            st.write(
                f"Detected regime column: `{regime_col}`"
            )

            regime_values = (
                integrated[regime_col]
                .dropna()
                .unique()
            )

            selected_regime = st.selectbox(
                "Select Regime",
                list(regime_values),
            )

            subset = integrated[
                integrated[regime_col]
                == selected_regime
            ].copy()

            st.metric(
                "Observations",
                len(subset),
            )

            correctness_cols = [
                c for c in subset.columns
                if "correct" in c.lower()
            ]

            if correctness_cols:

                rows = []

                for col in correctness_cols:

                    values = numeric(
                        subset[col]
                    ).dropna()

                    if not values.empty:

                        rows.append(
                            {
                                "Metric": col,
                                "Accuracy": values.mean(),
                                "Observations": len(values),
                            }
                        )

                regime_perf = pd.DataFrame(rows)

                if not regime_perf.empty:

                    regime_perf["Accuracy"] = (
                        regime_perf["Accuracy"] * 100
                    ).round(2)

                    st.dataframe(
                        regime_perf,
                        use_container_width=True,
                        hide_index=True,
                    )

            st.markdown("### Regime Distribution")

            distribution = (
                integrated[regime_col]
                .value_counts()
                .reset_index()
            )

            distribution.columns = [
                "Regime",
                "Observations",
            ]

            fig = px.bar(
                distribution,
                x="Regime",
                y="Observations",
                title="Forecast Observations by Regime",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# CONFUSION MATRIX
# ============================================================

elif section == "Confusion Matrix":

    st.subheader("🎯 Directional Confusion Matrix")

    prediction_columns = []

    for model in models:

        col = detect_prediction_column(
            wf,
            model
        )

        if col:
            prediction_columns.append(
                (model, col)
            )

    if not prediction_columns or not actual_direction_col:

        st.warning(
            "Explicit predicted-direction and actual-direction columns "
            "are required for confusion matrix analysis."
        )

    else:

        model_name, pred_col = st.selectbox(
            "Model",
            prediction_columns,
            format_func=lambda x: x[0],
        )

        temp = wf[
            [
                actual_direction_col,
                pred_col,
            ]
        ].copy()

        temp.columns = [
            "Actual",
            "Predicted",
        ]

        temp = temp.dropna()

        if not temp.empty:

            actual_values = temp["Actual"]
            pred_values = temp["Predicted"]

            labels = sorted(
                set(actual_values.unique())
                |
                set(pred_values.unique())
            )

            matrix = pd.crosstab(
                actual_values,
                pred_values,
                rownames=["Actual"],
                colnames=["Predicted"],
            )

            st.dataframe(
                matrix,
                use_container_width=True,
            )

            fig = px.imshow(
                matrix,
                text_auto=True,
                aspect="auto",
                title=f"{model_name} Confusion Matrix",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# PREDICTION STREAKS
# ============================================================

elif section == "Prediction Streaks":

    st.subheader("🔥 Prediction Streak Analysis")

    model = st.selectbox(
        "Model",
        models,
    )

    correct_col = detect_correct_column(
        wf,
        model
    )

    if not correct_col:

        st.warning(
            "Correctness column unavailable."
        )

    else:

        values = (
            numeric(wf[correct_col])
            .dropna()
            .astype(int)
            .tolist()
        )

        streaks = []

        current = None
        length = 0

        for value in values:

            if value == current:
                length += 1
            else:

                if current is not None:

                    streaks.append(
                        {
                            "Result": current,
                            "Length": length,
                        }
                    )

                current = value
                length = 1

        if current is not None:

            streaks.append(
                {
                    "Result": current,
                    "Length": length,
                }
            )

        streak_df = pd.DataFrame(streaks)

        correct_streaks = streak_df[
            streak_df["Result"] == 1
        ]

        wrong_streaks = streak_df[
            streak_df["Result"] == 0
        ]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            metric_card(
                "Longest Correct",
                int(
                    correct_streaks["Length"].max()
                )
                if not correct_streaks.empty
                else 0,
            )

        with c2:
            metric_card(
                "Longest Incorrect",
                int(
                    wrong_streaks["Length"].max()
                )
                if not wrong_streaks.empty
                else 0,
            )

        with c3:
            metric_card(
                "Avg Correct Streak",
                format_num(
                    correct_streaks["Length"].mean(),
                    2,
                ),
            )

        with c4:
            metric_card(
                "Avg Incorrect Streak",
                format_num(
                    wrong_streaks["Length"].mean(),
                    2,
                ),
            )

        fig = px.histogram(
            streak_df,
            x="Length",
            color="Result",
            nbins=20,
            title="Prediction Streak Distribution",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# MODEL AGREEMENT
# ============================================================

elif section == "Model Agreement":

    st.subheader("🤝 Model Agreement & Disagreement")

    prediction_data = {}

    for model in models:

        col = detect_prediction_column(
            wf,
            model
        )

        if col:

            prediction_data[model] = numeric(
                wf[col]
            )

    if len(prediction_data) < 2:

        st.warning(
            "At least two explicit model prediction columns are required."
        )

    else:

        pred_df = pd.DataFrame(
            prediction_data
        )

        agreement_count = (
            pred_df.nunique(axis=1) == 1
        )

        agreement_rate = agreement_count.mean()

        disagreement_rate = 1 - agreement_rate

        c1, c2 = st.columns(2)

        with c1:
            metric_card(
                "Full Model Agreement",
                format_pct(agreement_rate),
            )

        with c2:
            metric_card(
                "Model Disagreement",
                format_pct(disagreement_rate),
            )

        fig = go.Figure()

        rolling_agreement = (
            agreement_count
            .rolling(60)
            .mean()
        )

        fig.add_trace(
            go.Scatter(
                x=wf[date_col],
                y=rolling_agreement,
                mode="lines",
                name="Agreement",
            )
        )

        fig.update_layout(
            title="Rolling Model Agreement",
            yaxis_title="Agreement Rate",
            xaxis_title="Date",
            height=500,
        )

        fig.update_yaxes(
            tickformat=".0%"
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

        st.dataframe(
            pred_df.tail(30),
            use_container_width=True,
        )


# ============================================================
# ERROR INTELLIGENCE
# ============================================================

elif section == "Error Intelligence":

    st.subheader("🧠 Forecast Error Intelligence")

    rows = []

    for model in models:

        col = detect_correct_column(
            wf,
            model
        )

        if not col:
            continue

        correct = numeric(
            wf[col]
        )

        error = 1 - correct

        rows.append(
            {
                "Model": model,
                "Total Errors": int(
                    error.sum()
                ),
                "Error Rate": error.mean(),
                "Worst Rolling Accuracy": (
                    correct
                    .rolling(60)
                    .mean()
                    .min()
                ),
                "Best Rolling Accuracy": (
                    correct
                    .rolling(60)
                    .mean()
                    .max()
                ),
            }
        )

    error_df = pd.DataFrame(rows)

    if not error_df.empty:

        display = error_df.copy()

        display["Error Rate"] = (
            display["Error Rate"] * 100
        ).round(2)

        display["Worst Rolling Accuracy"] = (
            display["Worst Rolling Accuracy"] * 100
        ).round(2)

        display["Best Rolling Accuracy"] = (
            display["Best Rolling Accuracy"] * 100
        ).round(2)

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("### Hit / Miss Timeline")

    model = st.selectbox(
        "Select model",
        models,
        key="error_model",
    )

    correct_col = detect_correct_column(
        wf,
        model
    )

    if correct_col:

        temp = wf[
            [
                date_col,
                correct_col,
            ]
        ].copy()

        temp["Status"] = np.where(
            temp[correct_col] == 1,
            "Correct",
            "Incorrect",
        )

        fig = px.scatter(
            temp,
            x=date_col,
            y="Status",
            color="Status",
            title=f"{model} Hit / Miss Timeline",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# STATISTICAL SIGNIFICANCE
# ============================================================

elif section == "Statistical Significance":

    st.subheader("📐 Statistical Significance")

    if significance.empty:

        st.warning(
            "model_significance_tests.csv not found."
        )

    else:

        st.dataframe(
            significance,
            use_container_width=True,
            hide_index=True,
        )

        p_columns = [
            c for c in significance.columns
            if "p" in c.lower()
        ]

        if p_columns:

            p_col = p_columns[0]

            p_values = numeric(
                significance[p_col]
            )

            significant = (
                p_values < 0.05
            )

            st.metric(
                "Significant Tests at α = 0.05",
                int(significant.sum()),
            )

            st.info(
                """
                A p-value below 0.05 indicates evidence against
                the null hypothesis for that specific comparison.

                Statistical significance should not be confused
                with economic usefulness or practical significance.
                """
            )


# ============================================================
# BOOTSTRAP ROBUSTNESS
# ============================================================

elif section == "Bootstrap Robustness":

    st.subheader("🎲 Bootstrap Confidence Intervals")

    if bootstrap.empty:

        st.warning(
            "bootstrap_confidence_intervals.csv not found."
        )

    else:

        st.dataframe(
            bootstrap,
            use_container_width=True,
            hide_index=True,
        )

        model_col = find_column(
            bootstrap,
            ["model"]
        )

        lower_col = find_column(
            bootstrap,
            ["lower"]
        )

        upper_col = find_column(
            bootstrap,
            ["upper"]
        )

        score_col = find_column(
            bootstrap,
            ["accuracy", "score", "mean"]
        )

        if model_col and lower_col and upper_col:

            chart_df = bootstrap.copy()

            chart_df["Lower"] = numeric(
                chart_df[lower_col]
            )

            chart_df["Upper"] = numeric(
                chart_df[upper_col]
            )

            if score_col:

                chart_df["Score"] = numeric(
                    chart_df[score_col]
                )

            else:

                chart_df["Score"] = (
                    chart_df["Lower"]
                    +
                    chart_df["Upper"]
                ) / 2

            fig = go.Figure()

            for _, row in chart_df.iterrows():

                fig.add_trace(
                    go.Scatter(
                        x=[
                            row["Lower"],
                            row["Upper"],
                        ],
                        y=[
                            row[model_col],
                            row[model_col],
                        ],
                        mode="lines+markers",
                        name=str(
                            row[model_col]
                        ),
                        showlegend=False,
                    )
                )

            fig.update_layout(
                title="Bootstrap Confidence Intervals",
                xaxis_title="Accuracy",
                yaxis_title="Model",
                height=450,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )


# ============================================================
# TRADING SIMULATION
# ============================================================

elif section == "Trading Simulation":

    st.subheader("💹 Trading Simulation")

    st.warning(
        """
        Financial backtesting is only enabled when the dataset contains
        explicit predicted trading directions/signals AND actual returns.

        The dashboard will NOT convert correctness labels into returns,
        because doing so would create a scientifically invalid equity curve.
        """
    )

    if not actual_return_col:

        st.error(
            "No explicit actual-return column was detected."
        )

        st.info(
            "Available columns:"
        )

        st.write(
            list(wf.columns)
        )

    else:

        available_predictions = []

        for model in models:

            col = detect_prediction_column(
                wf,
                model
            )

            if col:
                available_predictions.append(
                    (model, col)
                )

        if not available_predictions:

            st.error(
                "No explicit model trading-direction columns were detected."
            )

        else:

            model_name, prediction_col = st.selectbox(
                "Model",
                available_predictions,
                format_func=lambda x: x[0],
                key="trading_model",
            )

            temp = wf[
                [
                    date_col,
                    prediction_col,
                    actual_return_col,
                ]
            ].copy()

            temp["Signal"] = numeric(
                temp[prediction_col]
            )

            temp["Actual Return"] = numeric(
                temp[actual_return_col]
            )

            temp = temp.dropna()

            unique_signals = set(
                temp["Signal"].unique()
            )

            if not unique_signals.issubset(
                {-1, 0, 1}
            ):

                st.error(
                    "Trading signal column does not use the expected -1 / 0 / +1 convention."
                )

            else:

                temp["Strategy Return"] = (
                    temp["Signal"]
                    *
                    temp["Actual Return"]
                )

                temp["Equity"] = (
                    1
                    +
                    temp["Strategy Return"]
                ).cumprod()

                total_return = (
                    temp["Equity"].iloc[-1] - 1
                )

                sharpe = annualized_sharpe(
                    temp["Strategy Return"]
                )

                drawdown = max_drawdown(
                    temp["Strategy Return"]
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_card(
                        "Total Return",
                        format_pct(total_return),
                    )

                with c2:
                    metric_card(
                        "Sharpe",
                        format_num(sharpe, 3),
                    )

                with c3:
                    metric_card(
                        "Max Drawdown",
                        format_pct(drawdown),
                    )

                fig = px.line(
                    temp,
                    x=date_col,
                    y="Equity",
                    title=f"{model_name} Simulated Equity Curve",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )


# ============================================================
# VALIDATION AUDIT
# ============================================================

elif section == "Validation Audit":

    st.subheader("🔐 Backtest Validation Audit")

    checks = []

    # Date check
    if date_col:

        dates = wf[date_col]

        checks.append(
            {
                "Validation": "Dates available",
                "Result": "PASS"
                if dates.notna().all()
                else "FAIL",
            }
        )

        checks.append(
            {
                "Validation": "Chronological ordering",
                "Result": "PASS"
                if dates.is_monotonic_increasing
                else "FAIL",
            }
        )

        checks.append(
            {
                "Validation": "Duplicate dates",
                "Result": "PASS"
                if dates.duplicated().sum() == 0
                else "FAIL",
            }
        )

    # Actual direction
    if actual_direction_col:

        checks.append(
            {
                "Validation": "Actual direction available",
                "Result": "PASS",
            }
        )

    else:

        checks.append(
            {
                "Validation": "Actual direction available",
                "Result": "WARNING",
            }
        )

    # Actual return
    if actual_return_col:

        checks.append(
            {
                "Validation": "Actual return available",
                "Result": "PASS",
            }
        )

    else:

        checks.append(
            {
                "Validation": "Actual return available",
                "Result": "WARNING",
            }
        )

    # Model correctness
    checks.append(
        {
            "Validation": "Model correctness fields",
            "Result": "PASS"
            if len(models) > 0
            else "FAIL",
        }
    )

    audit_df = pd.DataFrame(checks)

    st.dataframe(
        audit_df,
        use_container_width=True,
        hide_index=True,
    )

    if (
        (audit_df["Result"] == "FAIL")
        .any()
    ):

        st.error(
            "One or more validation checks failed."
        )

    else:

        st.success(
            "Core backtesting integrity checks passed."
        )

    st.markdown("### Scientific Limitations")

    st.info(
        """
        1. The dashboard evaluates saved chronological predictions.

        2. It does not automatically prove that every model was
           retrained at every forecast origin.

        3. Therefore this should be described as a chronological
           walk-forward evaluation of saved one-step predictions.

        4. A financial equity curve is only calculated when explicit
           trading signals and realized returns are available.

        5. Statistical significance does not imply economic significance.

        6. Historical performance does not guarantee future performance.
        """
    )


# ============================================================
# RAW DATA EXPLORER
# ============================================================

elif section == "Raw Data Explorer":

    st.subheader("🗂️ Backtest Data Explorer")

    st.write(
        f"Dataset shape: **{wf.shape[0]:,} rows × {wf.shape[1]} columns**"
    )

    selected_columns = st.multiselect(
        "Select columns",
        wf.columns.tolist(),
        default=wf.columns.tolist()[: min(8, len(wf.columns))],
    )

    if selected_columns:

        st.dataframe(
            wf[selected_columns].tail(100),
            use_container_width=True,
        )

    st.download_button(
        label="⬇️ Download Walk-Forward Backtest",
        data=wf.to_csv(index=False).encode(
            "utf-8"
        ),
        file_name="walk_forward_backtest_dashboard.csv",
        mime="text/csv",
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Bayesian Regime Detection Engine • Step 19 • "
    "Research-grade backtesting interface • "
    "Chronological evaluation • No fabricated financial metrics"
)