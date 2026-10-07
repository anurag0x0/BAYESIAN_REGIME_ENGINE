from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# MARKET OVERVIEW
# BAYESIAN REGIME DETECTION ENGINE
# ============================================================

st.set_page_config(
    page_title="Market Overview | Bayesian Regime Engine",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_market_data():

    files = [
        DATA_DIR / "nifty50_features.csv",
        DATA_DIR / "nifty50_clean.csv",
        RAW_DIR / "nifty50.csv",
    ]

    for file in files:

        if file.exists():

            try:
                df = pd.read_csv(file)
                return df, file.name

            except Exception:
                pass

    return pd.DataFrame(), "Not Found"


df, source_file = load_market_data()


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================
# DATE
# ============================================================

date_col = find_column(
    [
        "Date",
        "date",
        "Datetime",
        "datetime",
    ]
)


if not df.empty and date_col:

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .reset_index(drop=True)
    )


# ============================================================
# MARKET COLUMNS
# ============================================================

close_col = find_column(
    [
        "Close",
        "close",
        "Adj Close",
        "adj_close",
    ]
)

open_col = find_column(
    [
        "Open",
        "open",
    ]
)

high_col = find_column(
    [
        "High",
        "high",
    ]
)

low_col = find_column(
    [
        "Low",
        "low",
    ]
)

volume_col = find_column(
    [
        "Volume",
        "volume",
    ]
)

return_col = find_column(
    [
        "daily_return",
        "Daily_Return",
        "return",
    ]
)


# ============================================================
# CALCULATED FEATURES
# ============================================================

if not df.empty and close_col:

    df["market_return"] = (
        df[close_col]
        .pct_change()
    )

    df["volatility_20d"] = (
        df["market_return"]
        .rolling(20)
        .std()
    )

    df["rolling_high_252"] = (
        df[close_col]
        .rolling(252)
        .max()
    )

    df["drawdown"] = (
        df[close_col]
        /
        df[close_col].cummax()
        - 1
    )

    if return_col is None:

        return_col = "market_return"


# ============================================================
# HEADER
# ============================================================

st.title("📊 Market Overview")

st.caption(
    "NIFTY 50 historical market structure, "
    "price behaviour, returns and volatility"
)


if df.empty:

    st.error(
        "Market dataset could not be loaded."
    )

    st.stop()


# ============================================================
# DATE RANGE
# ============================================================

start_date = (
    df[date_col].min()
    if date_col
    else None
)

end_date = (
    df[date_col].max()
    if date_col
    else None
)


# ============================================================
# LATEST MARKET SNAPSHOT
# ============================================================

st.subheader("Market Snapshot")


latest = df.iloc[-1]


latest_close = (
    float(latest[close_col])
    if close_col
    else np.nan
)


previous_close = (
    float(df[close_col].iloc[-2])
    if close_col and len(df) > 1
    else np.nan
)


daily_change = (
    latest_close / previous_close - 1
    if (
        not pd.isna(latest_close)
        and not pd.isna(previous_close)
        and previous_close != 0
    )
    else np.nan
)


latest_volatility = (
    float(latest["volatility_20d"])
    if "volatility_20d" in df.columns
    else np.nan
)


latest_drawdown = (
    float(latest["drawdown"])
    if "drawdown" in df.columns
    else np.nan
)


high_52w = (
    float(df[close_col].tail(252).max())
    if close_col
    else np.nan
)


# ============================================================
# KPI CARDS
# ============================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "NIFTY 50 Close",
        (
            f"{latest_close:,.2f}"
            if not pd.isna(latest_close)
            else "N/A"
        ),
        (
            f"{daily_change:+.2%}"
            if not pd.isna(daily_change)
            else None
        ),
    )


with c2:

    st.metric(
        "20D Volatility",
        (
            f"{latest_volatility:.2%}"
            if not pd.isna(latest_volatility)
            else "N/A"
        ),
    )


with c3:

    st.metric(
        "52-Week High",
        (
            f"{high_52w:,.2f}"
            if not pd.isna(high_52w)
            else "N/A"
        ),
    )


with c4:

    st.metric(
        "Current Drawdown",
        (
            f"{latest_drawdown:.2%}"
            if not pd.isna(latest_drawdown)
            else "N/A"
        ),
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

st.divider()

st.subheader("Dataset Summary")


d1, d2, d3, d4 = st.columns(4)


with d1:

    st.metric(
        "Observations",
        f"{len(df):,}",
    )


with d2:

    st.metric(
        "Start",
        (
            start_date.strftime("%d %b %Y")
            if start_date is not None
            else "N/A"
        ),
    )


with d3:

    st.metric(
        "End",
        (
            end_date.strftime("%d %b %Y")
            if end_date is not None
            else "N/A"
        ),
    )


with d4:

    st.metric(
        "Dataset",
        source_file,
    )


# ============================================================
# PRICE TREND
# ============================================================

st.divider()

st.subheader("NIFTY 50 Price Trend")


if close_col and date_col:

    price_chart = (
        df[
            [
                date_col,
                close_col,
            ]
        ]
        .set_index(date_col)
    )

    st.line_chart(
        price_chart,
        height=400,
    )


# ============================================================
# RETURNS
# ============================================================

st.subheader("Daily Return Behaviour")


if return_col and date_col:

    return_chart = (
        df[
            [
                date_col,
                return_col,
            ]
        ]
        .set_index(date_col)
    )

    st.line_chart(
        return_chart,
        height=300,
    )


# ============================================================
# VOLATILITY
# ============================================================

st.subheader("20-Day Rolling Volatility")


if "volatility_20d" in df.columns:

    volatility_chart = (
        df[
            [
                date_col,
                "volatility_20d",
            ]
        ]
        .set_index(date_col)
    )

    st.line_chart(
        volatility_chart,
        height=300,
    )


# ============================================================
# MARKET STRUCTURE
# ============================================================

st.divider()

st.subheader("Recent Market Structure")


if (
    open_col
    and high_col
    and low_col
    and close_col
):

    recent = df[
        [
            date_col,
            open_col,
            high_col,
            low_col,
            close_col,
        ]
    ].tail(30)

    st.dataframe(
        recent,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "OHLC columns are not available "
        "in the selected dataset."
    )


# ============================================================
# VOLUME
# ============================================================

if volume_col:

    st.subheader("Trading Volume")

    volume_chart = (
        df[
            [
                date_col,
                volume_col,
            ]
        ]
        .set_index(date_col)
        .tail(252)
    )

    st.bar_chart(
        volume_chart,
        height=300,
    )


# ============================================================
# RESEARCH CONTEXT
# ============================================================

st.divider()

st.subheader("Why Market Overview Matters")


st.info(
    """
    **Role in the Bayesian Regime Detection Engine**

    This page provides the historical market context used by
    the downstream regime-detection and forecasting pipeline.

    • Price trend → long-term market direction

    • Daily returns → short-term market movement

    • Rolling volatility → changing market conditions

    • Drawdown → distance from historical peak

    • OHLC structure → daily price behaviour

    These market characteristics feed into the project's
    regime detection and probabilistic forecasting workflow.

    **Note:** This dashboard is for historical quantitative
    research and model evaluation. It is not financial advice.
    """
)


# ============================================================
# DATA EXPLORER
# ============================================================

with st.expander("🔎 Explore Market Dataset"):

    st.write(
        f"Dataset: **{source_file}**"
    )

    st.write(
        f"Rows: **{len(df):,}** | "
        f"Columns: **{len(df.columns):,}**"
    )

    st.dataframe(
        df.tail(100),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DOWNLOAD
# ============================================================

csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    "⬇️ Download Market Dataset",
    csv_data,
    file_name="nifty50_market_data.csv",
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bayesian Regime Detection Engine • "
    "Market Overview • Historical analysis only"
)