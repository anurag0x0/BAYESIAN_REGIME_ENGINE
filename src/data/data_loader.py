
from pathlib import Path
import logging

import pandas as pd
import yfinance as yf


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

NIFTY_TICKER = "^NSEI"
START_DATE = "2010-01-01"
END_DATE = None

OUTPUT_FILE = RAW_DATA_DIR / "nifty50.csv"


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# MARKET DATA LOADER
# ==========================================

class MarketDataLoader:

    def __init__(
        self,
        ticker=NIFTY_TICKER,
        start_date=START_DATE,
        end_date=END_DATE
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    # ======================================
    # DOWNLOAD DATA
    # ======================================

    def download_data(self):

        logger.info(
            "Downloading market data for %s",
            self.ticker
        )

        data = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=False
        )

        if data is None or data.empty:
            raise ValueError("Downloaded dataset is empty.")

        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # Preserve date from DatetimeIndex before resetting
        if isinstance(data.index, pd.DatetimeIndex):
            data.index.name = "date"

        # Convert index into a regular column
        data = data.reset_index()

        # Standardize column names
        data.columns = [
            str(col).strip().lower().replace(" ", "_")
            for col in data.columns
        ]

        # Identify date column
        possible_date_cols = [
            "date",
            "datetime",
            "index",
            "level_0"
        ]

        date_col = next(
            (
                col for col in possible_date_cols
                if col in data.columns
            ),
            None
        )

        # Fallback: detect a datetime-like column
        if date_col is None:
            for col in data.columns:
                converted = pd.to_datetime(
                    data[col],
                    errors="coerce"
                )

                if converted.notna().sum() > 0:
                    date_col = col
                    break

        if date_col is None:
            raise ValueError(
                f"Date column not found. Columns: {data.columns.tolist()}"
            )

        if date_col != "date":
            data = data.rename(
                columns={date_col: "date"}
            )

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )

        # Validate dates
        data = data.dropna(subset=["date"])

        # Remove timezone if present
        if data["date"].dt.tz is not None:
            data["date"] = data["date"].dt.tz_localize(None)

        # Remove duplicate dates
        data = data.drop_duplicates(
            subset=["date"]
        )

        # Sort by date
        data = data.sort_values("date")

        # Validate closing prices
        if "close" not in data.columns:
            raise ValueError(
                f"Close column missing. Columns: {data.columns.tolist()}"
            )

        data["close"] = pd.to_numeric(
            data["close"],
            errors="coerce"
        )

        data = data.dropna(subset=["close"])

        if (data["close"] <= 0).any():
            raise ValueError(
                "Non-positive closing prices found."
            )

        data = data.reset_index(drop=True)

        logger.info(
            "Successfully downloaded %d records",
            len(data)
        )

        logger.info(
            "Available columns: %s",
            data.columns.tolist()
        )

        return data

    # ======================================
    # SAVE DATA
    # ======================================

    def save_data(self, data):

        RAW_DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        data.to_csv(
            OUTPUT_FILE,
            index=False
        )

        logger.info(
            "Data saved to %s",
            OUTPUT_FILE
        )

    # ======================================
    # DATA SUMMARY
    # ======================================

    def get_summary(self, data):

        return {
            "ticker": self.ticker,
            "total_records": len(data),
            "start_date": str(
                data["date"].min().date()
            ),
            "end_date": str(
                data["date"].max().date()
            ),
            "missing_values": int(
                data.isnull().sum().sum()
            ),
            "duplicate_dates": int(
                data["date"].duplicated().sum()
            )
        }

    # ======================================
    # RUN PIPELINE
    # ======================================

    def run(self):

        data = self.download_data()

        self.save_data(data)

        summary = self.get_summary(data)

        logger.info("DATA SUMMARY")

        for key, value in summary.items():
            logger.info("%s: %s", key, value)

        return data


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    loader = MarketDataLoader()

    try:
        market_data = loader.run()

        print("\nFirst 5 records:")
        print(market_data.head().to_string(index=False))

        print("\nDataset information:")
        market_data.info()

        print("\nData collection completed!")

    except Exception as error:
        print(f"\nError: {error}")
