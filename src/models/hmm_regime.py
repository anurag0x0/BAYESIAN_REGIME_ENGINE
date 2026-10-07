
from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler


# ==========================================
# STEP 5: HMM MARKET REGIME DETECTION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "nifty50_features.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "nifty50_regimes.csv"

MODEL_DIR = ROOT_DIR / "models" / "hmm"
REPORT_DIR = ROOT_DIR / "reports" / "hmm"

MODEL_FILE = MODEL_DIR / "hmm_model.pkl"
SCALER_FILE = MODEL_DIR / "hmm_scaler.pkl"
REPORT_FILE = REPORT_DIR / "regime_summary.txt"

FEATURES = [
    "daily_return",
    "volatility_20d",
    "momentum_5d",
    "daily_range",
]

N_STATES = 3
TRAIN_RATIO = 0.80
RANDOM_STATE = 42


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main():
    setup_logging()
    logging.info("Starting Step 5: HMM Market Regime Detection")

    # 1. Load dataset
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}\n"
            "Please verify that Step 3 is complete."
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required_columns = ["date"] + FEATURES

    missing_columns = [
        col for col in required_columns if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    logging.info("Dataset loaded: %s records", len(df))

    # 2. Validate feature values
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    feature_array = df[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(feature_array).all():
        raise ValueError(
            "HMM features contain missing or infinite values. "
            "Please check the feature-engineering output."
        )

    if len(df) < 100:
        raise ValueError("Insufficient records for HMM training.")

    # 3. Chronological train split
    X = feature_array
    split_index = int(len(df) * TRAIN_RATIO)

    X_train = X[:split_index]

    logging.info(
        "Training records: %s | Total records: %s",
        len(X_train),
        len(X),
    )

    # 4. Scale features using training data only
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_scaled = scaler.transform(X)

    # 5. Train Gaussian Hidden Markov Model
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=200,
        tol=1e-3,
        min_covar=1e-4,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train_scaled)

    logging.info("HMM training completed.")
    logging.info("Model converged: %s", model.monitor_.converged)
    logging.info("Training iterations: %s", model.monitor_.iter)

    # 6. Infer latent states for retrospective analysis
    # Full-sequence decoding is NOT suitable as a live signal.
    hidden_states = model.predict(X_scaled)

    # Get training states to determine descriptive labels
    train_states = model.predict(X_train_scaled)

    # 7. Map states by training-period volatility
    train_volatility = df["volatility_20d"].iloc[:split_index].to_numpy()

    state_volatility = {}

    for state in range(N_STATES):
        state_values = train_volatility[train_states == state]

        if len(state_values) > 0:
            state_volatility[state] = float(np.mean(state_values))
        else:
            state_volatility[state] = float("inf")

    sorted_states = sorted(
        state_volatility,
        key=lambda state: state_volatility[state],
    )

    regime_names = [
        "Low Volatility",
        "Medium Volatility",
        "High Volatility",
    ]

    state_to_regime = {
        state: regime_names[index]
        for index, state in enumerate(sorted_states)
    }

    # 8. Create output dataset
    result = df.copy()

    result["hidden_state"] = hidden_states
    result["regime"] = [
        state_to_regime[int(state)]
        for state in hidden_states
    ]

    # 9. Create output folders
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # 10. Save model and scaler
    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    # 11. Save regime dataset
    result.to_csv(OUTPUT_FILE, index=False)

    # 12. Generate summary report
    report_lines = [
        "HMM MARKET REGIME DETECTION REPORT",
        "=" * 45,
        f"Total records: {len(result)}",
        f"Training records: {split_index}",
        f"Training ratio: {TRAIN_RATIO:.0%}",
        f"Number of hidden states: {N_STATES}",
        f"Model converged: {model.monitor_.converged}",
        f"Training iterations: {model.monitor_.iter}",
        "",
        f"Full data period: {df['date'].iloc[0].date()} "
        f"to {df['date'].iloc[-1].date()}",
        f"Training period: {df['date'].iloc[0].date()} "
        f"to {df['date'].iloc[split_index - 1].date()}",
        "",
        "REGIME SUMMARY",
        "-" * 45,
    ]

    for state in sorted_states:
        state_data = result[result["hidden_state"] == state]

        count = len(state_data)
        percentage = count / len(result) * 100

        mean_volatility = state_data["volatility_20d"].mean()
        mean_return = state_data["daily_return"].mean()

        report_lines.extend([
            f"Hidden state: {state}",
            f"Regime: {state_to_regime[state]}",
            f"Records: {count}",
            f"Percentage: {percentage:.2f}%",
            f"Mean 20-day volatility: {mean_volatility:.6f}",
            f"Mean daily return: {mean_return:.6f}",
            "",
        ])

    report_lines.extend([
        "IMPORTANT NOTES",
        "-" * 45,
        "Regime names are assigned using training-period volatility.",
        "HMM states are latent statistical patterns and do not",
        "guarantee specific market conditions or future direction.",
        "Full-sequence decoding is retrospective and can use",
        "future observations to infer earlier states.",
        "Use causal walk-forward inference for valid backtesting",
        "and live forecasting.",
    ])

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    # 13. Display results
    logging.info("Regime data saved: %s", OUTPUT_FILE)
    logging.info("HMM model saved: %s", MODEL_FILE)
    logging.info("Scaler saved: %s", SCALER_FILE)
    logging.info("Summary report saved: %s", REPORT_FILE)

    print("\nHMM REGIME DISTRIBUTION")
    print(result["regime"].value_counts().to_string())

    print("\nStep 5 completed successfully!")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Report file: {REPORT_FILE}")


if __name__ == "__main__":
    main()
