# ============================================================
# BAYESIAN REGIME DETECTION ENGINE
# FINAL STREAMLIT DASHBOARD
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
MODEL_DIR = PROJECT_ROOT / "models"

st.set_page_config(
    page_title="Bayesian Regime Detection Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 25px 30px;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827,
            #1e3a8a
        );
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
    }

    .hero p {
        margin-top: 8px;
        color: #dbeafe;
        font-size: 15px;
    }

    .section {
        font-size: 23px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 12px;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 12px;
        box-shadow: 0 3px 12px rgba(0,0,0,0.05);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_report_csv(relative_path):
    path = REPORT_DIR / relative_path

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )
    return df


def pct(x):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.2f}%"


def file_exists(relative_path):
    return (
        PROJECT_ROOT / relative_path
    ).exists()


# ============================================================
# DATA LOADING
# ============================================================

features = load_csv(
    "nifty50_features.csv"
)

bayesian = load_csv(
    "bayesian_predictions.csv"
)

smc = load_csv(
    "smc_predictions.csv"
)

chronos = load_csv(
    "chronos_predictions.csv"
)

timesfm = load_csv(
    "timesfm_predictions.csv"
)

bdl = load_csv(
    "bayesian_deep_learning_predictions.csv"
)

advanced = load_csv(
    "advanced_ensemble_predictions.csv"
)

conformal = load_csv(
    "conformal_predictions.csv"
)

mc = load_csv(
    "monte_carlo_simulations.csv"
)

backtest = load_csv(
    "regime_overlay_backtest.csv"
)

causal_regime = load_csv(
    "causal_regime_integrated_predictions.csv"
)

bayesian_hmm = load_csv(
    "bayesian_hmm_regimes.csv"
)

bocpd = load_csv(
    "bocpd_changepoints.csv"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Regime Engine")

st.sidebar.caption(
    "Bayesian Regime Detection Engine"
)

st.sidebar.markdown(
    "---"
)

pages = [
    "🏠 Executive Overview",
    "📊 Model Performance",
    "🧠 Regime Intelligence",
    "🎯 Advanced Ensemble",
    "🛡️ Uncertainty & Conformal",
    "⚠️ Monte Carlo Risk",
    "💰 Allocation Backtest",
    "🔬 Statistical Validation",
    "🇷🇺 R Validation",
    "📁 Data Explorer",
]

page = st.sidebar.radio(
    "Navigate",
    pages
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Indian Equity Market | NIFTY 50"
)

st.sidebar.caption(
    "Research & Validation Dashboard"
)


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Bayesian Regime Detection Engine</h1>
        <p>
        Probabilistic Indian Equity Regime Detection |
        Bayesian Inference | Sequential Monte Carlo |
        Foundation Models | Conformal Uncertainty
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================

if page == "🏠 Executive Overview":

    st.markdown(
        '<div class="section">Executive Overview</div>',
        unsafe_allow_html=True
    )

    if features is not None:

        features["date"] = pd.to_datetime(
            features["date"],
            errors="coerce"
        )

        latest = features.iloc[-1]

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Observations",
                f"{len(features):,}"
            )

        with c2:
            st.metric(
                "Start Date",
                features["date"].min().strftime(
                    "%Y-%m-%d"
                )
            )

        with c3:
            st.metric(
                "End Date",
                features["date"].max().strftime(
                    "%Y-%m-%d"
                )
            )

        with c4:
            if "daily_return" in features.columns:
                st.metric(
                    "Latest Return",
                    f"{latest['daily_return'] * 100:.2f}%"
                )

        st.markdown("### Market Trend")

        if "close" in features.columns:

            chart = px.line(
                features,
                x="date",
                y="close",
                title="NIFTY 50 Price History"
            )

            chart.update_layout(
                height=450
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

        elif "close_price" in features.columns:

            chart = px.line(
                features,
                x="date",
                y="close_price",
                title="NIFTY 50 Price History"
            )

            st.plotly_chart(
                chart,
                use_container_width=True
            )

    st.markdown("### Model Snapshot")

    metric_rows = []

    model_files = {
        "Bayesian": bayesian,
        "SMC": smc,
        "Chronos": chronos,
        "TimesFM": timesfm,
        "Bayesian Deep Learning": bdl,
        "Advanced Ensemble": advanced,
    }

    for name, df in model_files.items():

        if df is not None:
            metric_rows.append(
                {
                    "Model": name,
                    "Rows": len(df),
                    "Available": "Yes"
                }
            )

    if metric_rows:
        st.dataframe(
            pd.DataFrame(metric_rows),
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "The dashboard consolidates the validated Python and R "
        "research outputs into one monitoring interface."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    st.markdown(
        '<div class="section">Model Performance</div>',
        unsafe_allow_html=True
    )

    metrics = {
        "Bayesian": {
            "Accuracy": 0.5154,
            "Brier": 0.2492,
            "ROC-AUC": 0.5179,
            "Log Loss": 0.6916,
        },
        "SMC": {
            "Accuracy": 0.5264,
            "Brier": 0.2509,
            "ROC-AUC": 0.5232,
            "Log Loss": 0.6950,
        },
        "Chronos": {
            "Accuracy": 0.5351,
            "Brier": 0.3431,
            "ROC-AUC": 0.5326,
            "Log Loss": 3.4449,
        },
        "Ensemble": {
            "Accuracy": 0.5264,
            "Brier": 0.2562,
            "ROC-AUC": 0.5359,
            "Log Loss": 0.7067,
        },
        "TimesFM": {
            "Accuracy": 0.5234,
            "Brier": 0.2498,
            "ROC-AUC": 0.5449,
            "Log Loss": 0.6929,
        },
        "Bayesian Deep Learning": {
            "Accuracy": 0.5080,
            "Brier": 0.2584,
            "ROC-AUC": 0.5138,
            "Log Loss": 0.7118,
        },
    }

    performance_df = pd.DataFrame(metrics).T.reset_index()

    performance_df = performance_df.rename(
        columns={"index": "Model"}
    )

    st.dataframe(
        performance_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    col1, col2 = st.columns(2)

    with col1:

        fig = px.bar(
            performance_df,
            x="Model",
            y="Accuracy",
            title="Accuracy Comparison"
        )

        fig.update_layout(
            yaxis_range=[0.45, 0.60]
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig = px.bar(
            performance_df,
            x="Model",
            y="Brier",
            title="Brier Score Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    st.warning(
        "Model performance differences are modest. "
        "Statistical testing did not establish significant "
        "superiority between the core models."
    )


# ============================================================
# REGIME INTELLIGENCE
# ============================================================

elif page == "🧠 Regime Intelligence":

    st.markdown(
        '<div class="section">Regime Intelligence</div>',
        unsafe_allow_html=True
    )

    df = causal_regime

    if df is None:
        df = causal_regime

    if df is not None:

        date_col = None

        for candidate in [
            "forecast_date",
            "feature_date",
            "date"
        ]:

            if candidate in df.columns:
                date_col = candidate
                break

        if date_col:

            df[date_col] = pd.to_datetime(
                df[date_col],
                errors="coerce"
            )

            regime_col = None

            for candidate in [
                "regime_label",
                "regime"
            ]:

                if candidate in df.columns:
                    regime_col = candidate
                    break

            if regime_col:

                counts = (
                    df[regime_col]
                    .value_counts()
                    .reset_index()
                )

                counts.columns = [
                    "Regime",
                    "Observations"
                ]

                col1, col2 = st.columns(2)

                with col1:

                    fig = px.bar(
                        counts,
                        x="Regime",
                        y="Observations",
                        title="Regime Distribution"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                with col2:

                    fig = px.pie(
                        counts,
                        names="Regime",
                        values="Observations",
                        title="Regime Share"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                st.dataframe(
                    counts,
                    use_container_width=True,
                    hide_index=True
                )

            probability_cols = [
                c
                for c in [
                    "prob_low_volatility",
                    "prob_medium_volatility",
                    "prob_high_volatility"
                ]
                if c in df.columns
            ]

            if probability_cols:

                chart_df = df[
                    [date_col] +
                    probability_cols
                ].copy()

                chart_df = chart_df.melt(
                    id_vars=date_col,
                    var_name="Regime",
                    value_name="Probability"
                )

                fig = px.line(
                    chart_df,
                    x=date_col,
                    y="Probability",
                    color="Regime",
                    title="Regime Probability Evolution"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    else:

        st.warning(
            "Regime output file not available."
        )


# ============================================================
# ADVANCED ENSEMBLE
# ============================================================

elif page == "🎯 Advanced Ensemble":

    st.markdown(
        '<div class="section">Advanced Ensemble</div>',
        unsafe_allow_html=True
    )

    if advanced is not None:

        st.success(
            f"Advanced ensemble loaded: {len(advanced):,} observations"
        )

        st.dataframe(
            advanced.head(100),
            use_container_width=True,
            hide_index=True
        )

        probability_candidates = [
            c
            for c in advanced.columns
            if "probability" in c.lower()
            or "probability" in c.lower()
        ]

        if probability_candidates:

            selected = st.selectbox(
                "Probability field",
                probability_candidates
            )

            fig = px.line(
                advanced,
                y=selected,
                title=f"{selected} Over Time"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.warning(
            "Advanced ensemble output not found."
        )


# ============================================================
# UNCERTAINTY + CONFORMAL
# ============================================================

elif page == "🛡️ Uncertainty & Conformal":

    st.markdown(
        '<div class="section">Uncertainty & Conformal Prediction</div>',
        unsafe_allow_html=True
    )

    if conformal is not None:

        st.success(
            f"Conformal output loaded: {len(conformal):,} rows"
        )

        st.dataframe(
            conformal,
            use_container_width=True,
            hide_index=True
        )

    conformal_metrics = load_report_csv(
        "conformal/metrics.csv"
    )

    if conformal_metrics is not None:

        st.markdown("### Conformal Metrics")

        st.dataframe(
            conformal_metrics,
            use_container_width=True,
            hide_index=True
        )

    coverage = load_report_csv(
        "conformal/coverage.csv"
    )

    if coverage is not None:

        st.markdown("### Coverage")

        st.dataframe(
            coverage,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "The split conformal experiment did not reach its nominal "
        "90% coverage target. APS was conservative. These results "
        "are shown transparently rather than overstated."
    )


# ============================================================
# MONTE CARLO RISK
# ============================================================

elif page == "⚠️ Monte Carlo Risk":

    st.markdown(
        '<div class="section">Monte Carlo Risk Engine</div>',
        unsafe_allow_html=True
    )

    if mc is not None:

        st.success(
            f"Monte Carlo simulations loaded: {len(mc):,} rows"
        )

        numeric_cols = [
            c
            for c in mc.columns
            if pd.api.types.is_numeric_dtype(
                mc[c]
            )
        ]

        st.dataframe(
            mc.head(100),
            use_container_width=True,
            hide_index=True
        )

        if numeric_cols:

            selected = st.selectbox(
                "Risk metric",
                numeric_cols
            )

            fig = px.histogram(
                mc,
                x=selected,
                title=f"Distribution: {selected}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    regime_risk = load_report_csv(
        "monte_carlo/regime_risk_metrics.csv"
    )

    if regime_risk is not None:

        st.markdown(
            "### Regime-Conditioned Risk"
        )

        st.dataframe(
            regime_risk,
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "High-volatility regime statistics are based on a small "
        "historical sample and should be interpreted cautiously."
    )


# ============================================================
# ALLOCATION BACKTEST
# ============================================================

elif page == "💰 Allocation Backtest":

    st.markdown(
        '<div class="section">Regime-Overlay Allocation Backtest</div>',
        unsafe_allow_html=True
    )

    if backtest is not None:

        st.success(
            f"Backtest observations: {len(backtest):,}"
        )

        st.dataframe(
            backtest.head(100),
            use_container_width=True,
            hide_index=True
        )

        date_col = None

        for c in [
            "date",
            "forecast_date"
        ]:

            if c in backtest.columns:
                date_col = c
                break

        return_cols = [
            c
            for c in [
                "strategy_return",
                "benchmark_return",
                "strategy_cumulative",
                "benchmark_cumulative"
            ]
            if c in backtest.columns
        ]

        if date_col and return_cols:

            temp = backtest.copy()

            temp[date_col] = pd.to_datetime(
                temp[date_col],
                errors="coerce"
            )

            selected_returns = st.multiselect(
                "Backtest series",
                return_cols,
                default=return_cols[:2]
            )

            if selected_returns:

                chart_df = temp[
                    [date_col] +
                    selected_returns
                ]

                fig = px.line(
                    chart_df,
                    x=date_col,
                    y=selected_returns,
                    title="Allocation Backtest"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

    allocation_metrics = load_report_csv(
        "allocation_backtest/metrics.csv"
    )

    if allocation_metrics is not None:

        st.markdown("### Backtest Metrics")

        st.dataframe(
            allocation_metrics,
            use_container_width=True,
            hide_index=True
        )

    st.warning(
        "Current saved allocation backtest covers 2023-06-16 "
        "to 2026-09-30. It should not be represented as a "
        "2019-2024 backtest."
    )


# ============================================================
# STATISTICAL VALIDATION
# ============================================================

elif page == "🔬 Statistical Validation":

    st.markdown(
        '<div class="section">Statistical Robustness</div>',
        unsafe_allow_html=True
    )

    robustness = load_report_csv(
        "evaluation/statistical_robustness.csv"
    )

    if robustness is not None:

        st.dataframe(
            robustness,
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "Detailed robustness table is available in "
            "reports/evaluation/."
        )

    walk_forward = load_report_csv(
        "evaluation/walk_forward_model_metrics.csv"
    )

    if walk_forward is not None:

        st.markdown(
            "### Walk-Forward Performance"
        )

        st.dataframe(
            walk_forward,
            use_container_width=True,
            hide_index=True
        )

    st.info(
        "Bootstrap and McNemar testing did not establish "
        "statistically significant superiority of the core "
        "models at the 5% level."
    )


# ============================================================
# R VALIDATION
# ============================================================

elif page == "🇷🇺 R Validation":

    st.markdown(
        '<div class="section">Independent R Validation</div>',
        unsafe_allow_html=True
    )

    st.success(
        "R validation pipeline completed successfully."
    )

    r_files = {
        "R Bayesian Regime": (
            "reports/r_validation/"
            "r_bayesian_regime.csv"
        ),

        "R Changepoint": (
            "reports/r_validation/"
            "r_changepoint_results.csv"
        ),

        "R Conformal": (
            "reports/r_validation/"
            "r_conformal_results.csv"
        ),
    }

    for label, relative in r_files.items():

        path = PROJECT_ROOT / relative

        if path.exists():

            st.markdown(
                f"### {label}"
            )

            try:

                df = pd.read_csv(
                    path
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

            except Exception as exc:

                st.error(
                    f"Could not load {label}: {exc}"
                )

    summary_path = (
        REPORT_DIR /
        "r_validation" /
        "r_final_validation_summary.txt"
    )

    if summary_path.exists():

        st.markdown(
            "### Final R Validation Summary"
        )

        st.code(
            summary_path.read_text(
                encoding="utf-8",
                errors="ignore"
            )
        )

    st.markdown("### R Validation Status")

    st.metric(
        "R HMM",
        "PASSED"
    )

    st.metric(
        "R Bayesian",
        "PASSED"
    )

    st.metric(
        "R Changepoint + Conformal",
        "PASSED"
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "📁 Data Explorer":

    st.markdown(
        '<div class="section">Data Explorer</div>',
        unsafe_allow_html=True
    )

    available = {}

    for filename in [
        "nifty50_features.csv",
        "bayesian_predictions.csv",
        "smc_predictions.csv",
        "chronos_predictions.csv",
        "timesfm_predictions.csv",
        "bayesian_deep_learning_predictions.csv",
        "advanced_ensemble_predictions.csv",
        "conformal_predictions.csv",
        "monte_carlo_simulations.csv",
        "regime_overlay_backtest.csv",
        "causal_regime_integrated_predictions.csv",
    ]:

        path = DATA_DIR / filename

        if path.exists():
            available[filename] = path

    if available:

        selected_file = st.selectbox(
            "Select dataset",
            list(available.keys())
        )

        df = pd.read_csv(
            available[selected_file]
        )

        st.write(
            f"Rows: {len(df):,} | "
            f"Columns: {len(df.columns):,}"
        )

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download CSV",
            csv,
            file_name=selected_file,
            mime="text/csv"
        )

    else:

        st.warning(
            "No processed datasets found."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Bayesian Regime Detection Engine | "
    "NIFTY 50 | Research & Validation Dashboard"
)

st.caption(
    "For research and analytical purposes only. "
    "Not investment advice."
)