from pathlib import Path
import pandas as pd
import streamlit as st


# ============================================================
# BAYESIAN REGIME DETECTION ENGINE
# FINAL STREAMLIT DASHBOARD
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "evaluation"


# ------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Bayesian Regime Detection Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

@st.cache_data
def load_csv(path):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_col(df, candidates):
    """Return the first matching column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def fmt_pct(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.2%}"


def fmt_num(value):
    if pd.isna(value):
        return "N/A"
    return f"{value:.4f}"


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

walk_forward = load_csv(
    DATA / "walk_forward_backtest.csv"
)

performance = load_csv(
    REPORTS / "walk_forward_summary.csv"
)

yearly = load_csv(
    REPORTS / "walk_forward_yearly.csv"
)

significance = load_csv(
    REPORTS / "model_significance_tests.csv"
)

bootstrap = load_csv(
    REPORTS / "bootstrap_confidence_intervals.csv"
)

regime_robustness = load_csv(
    REPORTS / "statistical_regime_robustness.csv"
)

regime_integration = load_csv(
    REPORTS / "causal_regime_performance.csv"
)

causal_regimes = load_csv(
    DATA / "nifty50_regimes_causal.csv"
)


# ------------------------------------------------------------