
from pathlib import Path
import logging

import pandas as pd
import matplotlib.pyplot as plt


# ==========================================
# PROJECT PATHS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT_DIR / "data" / "processed" / "nifty50_features.csv"
)

OUTPUT_DIR = ROOT_DIR / "reports" / "eda"


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():
    """Load feature-engineered NIFTY 50 data."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}\n"
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(
        INPUT_FILE,
        parse_dates=["date"]
    )

    df = df.sort_values("date").reset_index(drop=True)

    logger.info("Dataset loaded: %s records", len(df))

    return df


# ==========================================
# SAVE CHART
# ==========================================

def save_plot(filename):
    """Save the current Matplotlib chart."""

    output_path = OUTPUT_DIR / filename

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )
    plt.close()

    logger.info("Chart saved: %s", output_path)


# ==========================================
# 1. PRICE TREND
# ==========================================

def plot_price_trend(df):
    """Plot historical NIFTY 50 closing prices."""

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["date"],
        df["close"],
        label="NIFTY 50 Close",
        linewidth=1
    )

    plt.title("NIFTY 50 Historical Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot("01_price_trend.png")


# ==========================================
# 2. DAILY RETURNS
# ==========================================

def plot_daily_returns(df):
    """Plot daily percentage returns."""

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["date"],
        df["daily_return"] * 100,
        label="Daily Return",
        linewidth=0.7
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=1
    )

    plt.title("NIFTY 50 Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Daily Return (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot("02_daily_returns.png")


# ==========================================
# 3. ROLLING VOLATILITY
# ==========================================

def plot_volatility(df):
    """Plot 20-day rolling volatility."""

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["date"],
        df["volatility_20d"] * 100,
        label="20-Day Volatility",
        linewidth=1
    )

    plt.title("NIFTY 50 Rolling Volatility")
    plt.xlabel("Date")
    plt.ylabel("Daily Volatility (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot("03_rolling_volatility.png")


# ==========================================
# 4. MOVING AVERAGES
# ==========================================

def plot_moving_averages(df):
    """Compare close price with moving averages."""

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["date"],
        df["close"],
        label="Closing Price",
        linewidth=1
    )

    for window in [20, 50]:
        column = f"sma_{window}"

        if column in df.columns:
            plt.plot(
                df["date"],
                df[column],
                label=f"{window}-Day SMA",
                linewidth=1
            )

    plt.title("NIFTY 50 Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot("04_moving_averages.png")


# ==========================================
# 5. RETURN DISTRIBUTION
# ==========================================

def plot_return_distribution(df):
    """Plot histogram of daily returns."""

    returns = (
        df["daily_return"].dropna() * 100
    )

    plt.figure(figsize=(12, 6))

    plt.hist(
        returns,
        bins=60,
        edgecolor="black"
    )

    plt.axvline(
        returns.mean(),
        linestyle="--",
        linewidth=1,
        label=f"Mean: {returns.mean():.4f}%"
    )

    plt.title("Distribution of NIFTY 50 Daily Returns")
    plt.xlabel("Daily Return (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_plot("05_return_distribution.png")


# ==========================================
# 6. FEATURE CORRELATION
# ==========================================

def plot_correlation(df):
    """Plot correlation heatmap of selected features."""

    candidate_columns = [
        "daily_return",
        "log_return",
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        "momentum_5d",
        "momentum_10d",
        "momentum_20d",
        "price_to_sma_20",
        "daily_range",
        "overnight_gap",
        "close_position"
    ]

    columns = [
        col for col in candidate_columns
        if col in df.columns
    ]

    if len(columns) < 2:
        logger.warning(
            "Not enough numeric features for correlation plot"
        )
        return

    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(12, 9))

    image = ax.imshow(
        corr,
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))

    ax.set_xticklabels(
        columns,
        rotation=60,
        ha="right"
    )

    ax.set_yticklabels(columns)

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

    ax.set_title("Market Feature Correlation Matrix")

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation"
    )

    save_plot("06_feature_correlation.png")


# ==========================================
# EDA SUMMARY REPORT
# ==========================================

def generate_report(df):
    """Generate a descriptive EDA text report."""

    returns = df["daily_return"].dropna()

    total_change = (
        df["close"].iloc[-1] / df["close"].iloc[0] - 1
    ) * 100

    lines = [
        "NIFTY 50 - EXPLORATORY DATA ANALYSIS",
        "=" * 45,
        "",
        "DATASET OVERVIEW",
        "-" * 45,
        f"Total Records: {len(df)}",
        f"Total Columns: {len(df.columns)}",
        f"Start Date: {df['date'].min().date()}",
        f"End Date: {df['date'].max().date()}",
        f"Missing Values: {df.isnull().sum().sum()}",
        f"Duplicate Dates: {df['date'].duplicated().sum()}",
        "",
        "PRICE SUMMARY",
        "-" * 45,
        f"Starting Close: {df['close'].iloc[0]:.2f}",
        f"Ending Close: {df['close'].iloc[-1]:.2f}",
        f"Total Price Change: {total_change:.2f}%",
        f"Minimum Close: {df['close'].min():.2f}",
        f"Maximum Close: {df['close'].max():.2f}",
        "",
        "DAILY RETURN SUMMARY",
        "-" * 45,
        f"Mean Return: {returns.mean() * 100:.5f}%",
        f"Median Return: {returns.median() * 100:.5f}%",
        f"Std Deviation: {returns.std() * 100:.5f}%",
        f"Minimum Return: {returns.min() * 100:.5f}%",
        f"Maximum Return: {returns.max() * 100:.5f}%",
        f"Positive Days: {(returns > 0).sum()}",
        f"Negative Days: {(returns < 0).sum()}",
        f"Zero Return Days: {(returns == 0).sum()}",
        "",
        "VOLATILITY SUMMARY",
        "-" * 45,
        (
            "Average 20-Day Volatility: "
            f"{df['volatility_20d'].mean() * 100:.5f}%"
        ),
        (
            "Maximum 20-Day Volatility: "
            f"{df['volatility_20d'].max() * 100:.5f}%"
        ),
        "",
        "Note: Historical statistics do not guarantee "
        "future market performance."
    ]

    report_path = OUTPUT_DIR / "eda_summary.txt"

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    logger.info("EDA report saved: %s", report_path)


# ==========================================
# MAIN
# ==========================================

def main():
    logger.info("Starting Step 4: EDA & Visualization")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_data()

    required_columns = [
        "date",
        "close",
        "daily_return",
        "volatility_20d",
        "sma_20",
        "sma_50"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. "
            "Run feature_engineering.py again."
        )

    plot_price_trend(df)
    plot_daily_returns(df)
    plot_volatility(df)
    plot_moving_averages(df)
    plot_return_distribution(df)
    plot_correlation(df)

    generate_report(df)

    logger.info("Step 4 completed successfully!")

    print("\nEDA completed successfully!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
