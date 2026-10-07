
from pathlib import Path
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
)
from sklearn.preprocessing import StandardScaler


# ==========================================
# STEP 6: BAYESIAN DIRECTION MODEL
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT_DIR / "data" / "processed" / "nifty50_features.csv"
OUTPUT_FILE = ROOT_DIR / "data" / "processed" / "bayesian_predictions.csv"

MODEL_DIR = ROOT_DIR / "models" / "bayesian"
REPORT_DIR = ROOT_DIR / "reports" / "bayesian"

MODEL_FILE = MODEL_DIR / "bayesian_model.pkl"
SCALER_FILE = MODEL_DIR / "bayesian_scaler.pkl"
REPORT_FILE = REPORT_DIR / "bayesian_summary.txt"

FEATURES = [
    "daily_return",
    "volatility_20d",
    "momentum_5d",
    "daily_range",
    "close_position",
]

TRAIN_RATIO = 0.80


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def main():
    setup_logging()
    logging.info("Starting Step 6: Bayesian Direction Model")

    # 1. Load dataset
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    required = ["date", "log_return"] + FEATURES
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logging.info("Dataset loaded: %s records", len(df))

    # 2. Create next-day return target.
    # Features on date t predict log_return on date t+1.
    df["target_next_day_return"] = df["log_return"].shift(-1)
    df["prediction_for_date"] = df["date"].shift(-1)

    for col in FEATURES + ["target_next_day_return"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_mask = np.isfinite(
        df[FEATURES + ["target_next_day_return"]].to_numpy(dtype=float)
    ).all(axis=1)

    data = df.loc[valid_mask].copy().reset_index(drop=True)

    if len(data) < 100:
        raise ValueError("Insufficient valid records for training.")

    X = data[FEATURES].to_numpy(dtype=float)
    y = data["target_next_day_return"].to_numpy(dtype=float)
    y_direction = (y > 0).astype(int)

    # 3. Chronological train-test split
    split_index = int(len(data) * TRAIN_RATIO)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]
    y_test_direction = y_direction[split_index:]

    logging.info(
        "Training records: %s | Testing records: %s",
        len(X_train),
        len(X_test),
    )

    # 4. Scale using training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Bayesian Ridge Regression
    model = BayesianRidge(
        max_iter=500,
        tol=1e-4,
        compute_score=True,
    )
    model.fit(X_train_scaled, y_train)

    logging.info("Bayesian Ridge training completed.")

    # 6. Predict return and predictive uncertainty
    predicted_return, predictive_std = model.predict(
        X_test_scaled,
        return_std=True,
    )

    predictive_std = np.maximum(predictive_std, 1e-12)

    # Estimated probability of a positive next-day return
    probability_positive = norm.cdf(
        predicted_return / predictive_std
    )

    predicted_direction = (
        probability_positive >= 0.5
    ).astype(int)

    # 7. Evaluate on chronological holdout test set
    accuracy = accuracy_score(
        y_test_direction,
        predicted_direction,
    )

    brier = brier_score_loss(
        y_test_direction,
        probability_positive,
    )

    mae = mean_absolute_error(
        y_test,
        predicted_return,
    )

    # 8. Create test prediction table
    test_data = data.iloc[split_index:].copy()

    predictions = pd.DataFrame({
        "feature_date": test_data["date"].to_numpy(),
        "prediction_for_date": (
            test_data["prediction_for_date"].to_numpy()
        ),
        "predicted_next_day_return": predicted_return,
        "predictive_std": predictive_std,
        "probability_positive": probability_positive,
        "predicted_direction": predicted_direction,
        "actual_next_day_return": y_test,
        "actual_direction": y_test_direction,
    })

    predictions["predicted_direction_label"] = (
        predictions["predicted_direction"]
        .map({1: "Positive", 0: "Non-positive"})
    )

    # 9. Save model, scaler and predictions
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    predictions.to_csv(OUTPUT_FILE, index=False)

    # 10. Generate report
    report_lines = [
        "BAYESIAN DIRECTION MODEL REPORT",
        "=" * 45,
        f"Total valid records: {len(data)}",
        f"Training records: {len(X_train)}",
        f"Testing records: {len(X_test)}",
        f"Training ratio: {TRAIN_RATIO:.0%}",
        f"Features: {', '.join(FEATURES)}",
        "",
        "TEST METRICS",
        "-" * 45,
        f"Direction accuracy: {accuracy:.4f}",
        f"Brier score: {brier:.4f}",
        f"Return MAE: {mae:.8f}",
        "",
        "INTERPRETATION",
        "-" * 45,
        "Probabilities are model-based estimates, not certainties.",
        "A 0.5 threshold is used for descriptive direction labels.",
        "Accuracy alone does not establish forecasting value.",
        "Brier score measures probabilistic prediction error;",
        "lower values are better, with 0 being perfect.",
        "The test set is chronological and held out from fitting.",
        "Walk-forward evaluation and calibration checks are still needed.",
    ]

    REPORT_FILE.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    logging.info("Model saved: %s", MODEL_FILE)
    logging.info("Scaler saved: %s", SCALER_FILE)
    logging.info("Predictions saved: %s", OUTPUT_FILE)
    logging.info("Report saved: %s", REPORT_FILE)

    # 11. Display results
    print("\nBAYESIAN MODEL TEST RESULTS")
    print("-" * 35)
    print(f"Direction accuracy: {accuracy:.4f}")
    print(f"Brier score: {brier:.4f}")
    print(f"Return MAE: {mae:.8f}")

    print("\nSample predictions:")
    print(
        predictions[
            [
                "prediction_for_date",
                "predicted_next_day_return",
                "probability_positive",
                "actual_direction",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nStep 6 completed successfully!")
    print(f"Predictions: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
