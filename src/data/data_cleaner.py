
from pathlib import Path
import logging

import pandas as pd


# Project paths
ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_FILE = ROOT_DIR / "data" / "raw" / "nifty50.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "nifty50_clean.csv"


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = [
    "date", "close", "high", "low", "open", "volume"
]

PRICE_COLUMNS = ["open", "high", "low", "close"]


def load_data(file_path: Path) -> pd.DataFrame:
    """Load raw market data from CSV."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    logger.info("Raw dataset loaded: %s rows", len(df))

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate NIFTY 50 market data."""

    df = df.copy()

    # Normalize column names
    df.columns = [
        str(col).strip().lower()
        for col in df.columns
    ]

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep only required columns
    df = df[REQUIRED_COLUMNS]

    # Convert date column
    df["date"] = pd.to_datetime(
        df["date"], errors="coerce"
    )

    # Convert numeric columns
    numeric_columns = PRICE_COLUMNS + ["volume"]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    # Report missing values before cleaning
    logger.info(
        "Missing values before cleaning:\n%s",
        df.isnull().sum().to_string()
    )

    # Remove rows with invalid dates
    invalid_dates = df["date"].isna().sum()
    if invalid_dates:
        logger.warning(
            "Removing %s rows with invalid dates",
            invalid_dates
        )
        df = df.dropna(subset=["date"])

    # Remove duplicate dates, keeping the last record
    duplicate_count = df.duplicated(
        subset=["date"]
    ).sum()

    if duplicate_count:
        logger.warning(
            "Removing %s duplicate dates",
            duplicate_count
        )
        df = df.drop_duplicates(
            subset=["date"], keep="last"
        )

    # Remove rows with missing price or volume data
    before = len(df)

    df = df.dropna(
        subset=numeric_columns
    )

    removed = before - len(df)

    if removed:
        logger.warning(
            "Removed %s rows with missing numeric values",
            removed
        )

    # Check for non-positive prices
    valid_prices = (df[PRICE_COLUMNS] > 0).all(axis=1)

    invalid_price_count = (~valid_prices).sum()

    if invalid_price_count:
        logger.warning(
            "Removing %s rows with non-positive prices",
            invalid_price_count
        )
        df = df.loc[valid_prices].copy()

    # Check OHLC consistency
    valid_ohlc = (
        (df["high"] >= df[["open", "close", "low"]].max(axis=1))
        & (df["low"] <= df[["open", "close", "high"]].min(axis=1))
        & (df["volume"] >= 0)
    )

    invalid_ohlc_count = (~valid_ohlc).sum()

    if invalid_ohlc_count:
        logger.warning(
            "Removing %s rows with invalid OHLC/volume values",
            invalid_ohlc_count
        )
        df = df.loc[valid_ohlc].copy()

    # Sort chronologically
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure integer volume
    df["volume"] = df["volume"].astype("int64")

    # Final validation
    if df.empty:
        raise ValueError(
            "No valid records remain after cleaning."
        )

    if df["date"].duplicated().any():
        raise ValueError(
            "Duplicate dates remain after cleaning."
        )

    logger.info("Cleaned dataset: %s rows", len(df))

    return df


def save_data(df: pd.DataFrame, file_path: Path) -> None:
    """Save cleaned data to processed directory."""

    file_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    df.to_csv(file_path, index=False)

    logger.info("Cleaned data saved to: %s", file_path)


def main():
    logger.info("Starting data cleaning process")

    raw_df = load_data(RAW_FILE)

    clean_df = clean_data(raw_df)

    save_data(clean_df, OUTPUT_FILE)

    logger.info("CLEANING SUMMARY")
    logger.info("Records: %s", len(clean_df))
    logger.info(
        "Start date: %s",
        clean_df["date"].min().date()
    )
    logger.info(
        "End date: %s",
        clean_df["date"].max().date()
    )
    logger.info(
        "Missing values: %s",
        clean_df.isnull().sum().sum()
    )
    logger.info(
        "Duplicate dates: %s",
        clean_df["date"].duplicated().sum()
    )
    logger.info(
        "Columns: %s",
        clean_df.columns.tolist()
    )

    print("\nFirst 5 cleaned records:")
    print(clean_df.head().to_string(index=False))

    print("\nData cleaning completed successfully!")


if __name__ == "__main__":
    main()
