
from pathlib import Path
import logging

import numpy as np
import pandas as pd


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "nifty50_clean.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "nifty50_features.csv"


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def load_data():
    """Load cleaned NIFTY 50 data."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}\n"
            "Please complete Step 2 first."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    required_columns = [
        "date", "open", "high", "low", "close", "volume"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required_columns].copy()
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        raise ValueError("Duplicate dates found in cleaned data.")

    logger.info("Loaded %s records", len(df))

    return df


def perform_eda(df):
    """Print basic descriptive statistics."""

    logger.info("=" * 45)
    logger.info("BASIC DATA ANALYSIS")
    logger.info("=" * 45)

    logger.info("Dataset shape: %s", df.shape)
    logger.info(
        "Date range: %s to %s",
        df["date"].min().date(),
        df["date"].max().date()
    )

    logger.info(
        "Closing price summary:\n%s",
        df["close"].describe().round(2).to_string()
    )

    daily_returns = df["close"].pct_change().dropna()

    logger.info(
        "Daily return summary:\n%s",
        daily_returns.describe().round(5).to_string()
    )

    total_change = (
        df["close"].iloc[-1] / df["close"].iloc[0] - 1
    ) * 100

    logger.info("Total price change: %.2f%%", total_change)
    logger.info(
        "Positive return days: %s",
        (daily_returns > 0).sum()
    )
    logger.info(
        "Negative return days: %s",
        (daily_returns < 0).sum()
    )


def create_features(df):
    """Create historical market features."""

    df = df.copy()

    # 1. Daily percentage return
    df["daily_return"] = df["close"].pct_change()

    # 2. Logarithmic return
    df["log_return"] = np.log(
        df["close"] / df["close"].shift(1)
    )

    # 3. Rolling volatility
    for window in [5, 10, 20]:
        df[f"volatility_{window}d"] = (
            df["log_return"].rolling(window=window).std()
        )

    # Annualized 20-day volatility
    df["annualized_volatility_20d"] = (
        df["volatility_20d"] * np.sqrt(252)
    )

    # 4. Simple moving averages
    for window in [5, 10, 20, 50]:
        df[f"sma_{window}"] = (
            df["close"].rolling(window=window).mean()
        )

    # 5. Price distance from moving averages
    for window in [5, 20, 50]:
        df[f"price_to_sma_{window}"] = (
            df["close"] / df[f"sma_{window}"] - 1
        )

    # 6. Price momentum
    for window in [5, 10, 20]:
        df[f"momentum_{window}d"] = (
            df["close"].pct_change(periods=window)
        )

    # 7. Intraday high-low range relative to opening price
    df["daily_range"] = (
        (df["high"] - df["low"]) / df["open"]
    )

    # 8. Overnight opening gap from previous close
    previous_close = df["close"].shift(1)

    df["overnight_gap"] = (
        (df["open"] - previous_close) / previous_close
    )

    # 9. Closing price position within the day's range
    day_range = df["high"] - df["low"]

    df["close_position"] = np.where(
        day_range > 0,
        (df["close"] - df["low"]) / day_range,
        np.nan
    )

    # 10. Next trading day's direction (target label)
    next_close = df["close"].shift(-1)

    df["target_next_day_direction"] = np.where(
        next_close.isna(),
        np.nan,
        (next_close > df["close"]).astype(int)
    )

    # Remove rows with unavailable rolling features or target
    original_count = len(df)
    df = df.dropna().reset_index(drop=True)

    logger.info(
        "Rows removed due to unavailable feature/target values: %s",
        original_count - len(df)
    )

    if df.empty:
        raise ValueError("No records remain after feature creation.")

    return df


def save_data(df):
    """Save engineered features to processed folder."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    logger.info("Features saved to: %s", OUTPUT_FILE)


def main():
    logger.info("Starting Step 3: Feature Engineering")

    df = load_data()

    perform_eda(df)

    feature_df = create_features(df)

    save_data(feature_df)

    logger.info("=" * 45)
    logger.info("FEATURE ENGINEERING SUMMARY")
    logger.info("=" * 45)

    logger.info("Final records: %s", len(feature_df))
    logger.info("Total columns: %s", len(feature_df.columns))
    logger.info(
        "Missing values: %s",
        feature_df.isnull().sum().sum()
    )
    logger.info(
        "Duplicate dates: %s",
        feature_df["date"].duplicated().sum()
    )
    logger.info(
        "Target distribution:\n%s",
        feature_df["target_next_day_direction"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nFirst 5 engineered records:")
    print(feature_df.head().to_string(index=False))

    print("\nFeature columns:")
    print(feature_df.columns.tolist())

    print("\nStep 3 completed successfully!")


if __name__ == "__main__":
    main()
