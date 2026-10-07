
from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from chronos import ChronosPipeline
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error

warnings.filterwarnings("ignore")

# ==========================================
# 1. PROJECT PATHS
# ==========================================
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "nifty50_features.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "chronos"
REPORT_DIR = ROOT / "reports" / "chronos"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "chronos_predictions.csv"
CONFIG_PATH = MODEL_DIR / "chronos_config.json"
REPORT_PATH = REPORT_DIR / "chronos_summary.txt"

# ==========================================
# 2. CONFIGURATION
# ==========================================
MODEL_NAME = "amazon/chronos-t5-small"
TRAIN_RATIO = 0.80
CONTEXT_LENGTH = 256
NUM_SAMPLES = 100
BATCH_SIZE = 16
RANDOM_SEED = 42


# ==========================================
# 3. LOAD DATA
# ==========================================
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    required = [
        "date",
        "daily_return",
        "target_next_day_direction",
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df[required].copy()

    for col in ["daily_return", "target_next_day_direction"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = (
        df.sort_values("date")
        .drop_duplicates("date")
        .dropna(subset=required)
        .reset_index(drop=True)
    )

    if not set(df["target_next_day_direction"].unique()).issubset({0, 1}):
        raise ValueError("Direction target must contain only 0 and 1.")

    if len(df) < CONTEXT_LENGTH + 20:
        raise ValueError("Not enough data for Chronos forecasting.")

    return df


# ==========================================
# 4. LOAD PRETRAINED CHRONOS
# ==========================================
def load_chronos():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading pretrained model: {MODEL_NAME}")
    print("First run may download model weights.")

    pipeline = ChronosPipeline.from_pretrained(
        MODEL_NAME,
        device_map=device,
        torch_dtype=torch.float32,
    )

    return pipeline, device


# ==========================================
# 5. WALK-FORWARD FORECASTING
# ==========================================
def forecast_test_period(df, pipeline, train_end, device):
    returns = df["daily_return"].to_numpy(dtype=np.float32)

    # Each origin uses information available up to that date.
    # The first forecast is made from the last training date.
    forecast_indices = list(range(train_end - 1, len(df) - 1))

    results = []

    for start in range(0, len(forecast_indices), BATCH_SIZE):
        batch_indices = forecast_indices[start:start + BATCH_SIZE]

        contexts = []

        for idx in batch_indices:
            context = returns[
                max(0, idx - CONTEXT_LENGTH + 1):idx + 1
            ]

            if len(context) < CONTEXT_LENGTH:
                context = np.pad(
                    context,
                    (CONTEXT_LENGTH - len(context), 0),
                    mode="constant",
                )

            contexts.append(context)

        context_tensor = torch.tensor(
            np.stack(contexts),
            dtype=torch.float32,
            device=device,
        )

        with torch.no_grad():
            forecasts = pipeline.predict(
                context_tensor,
                prediction_length=1,
                num_samples=NUM_SAMPLES,
            )

        samples = forecasts.detach().cpu().numpy()

        if samples.ndim != 3:
            raise ValueError(
                f"Unexpected Chronos output shape: {samples.shape}"
            )

        samples = samples[:, :, 0]

        for row, idx in enumerate(batch_indices):
            sample_returns = samples[row]

            predicted_return = float(np.median(sample_returns))
            probability_up = float(np.mean(sample_returns > 0))

            results.append({
                "origin_date": df.loc[idx, "date"],
                "forecast_for_date": df.loc[idx + 1, "date"],
                "predicted_return": predicted_return,
                "actual_next_day_return": float(returns[idx + 1]),
                "probability_up": probability_up,
                "predicted_direction": int(probability_up >= 0.5),
                "actual_next_day_direction": int(
                    df.loc[idx, "target_next_day_direction"]
                ),
                "context_length": min(idx + 1, CONTEXT_LENGTH),
            })

        completed = min(start + BATCH_SIZE, len(forecast_indices))
        print(
            f"Forecasted {completed}/{len(forecast_indices)} "
            "test observations"
        )

    return pd.DataFrame(results)


# ==========================================
# 6. EVALUATION
# ==========================================
def evaluate_predictions(predictions):
    accuracy = accuracy_score(
        predictions["actual_next_day_direction"],
        predictions["predicted_direction"],
    )

    brier = brier_score_loss(
        predictions["actual_next_day_direction"],
        predictions["probability_up"],
    )

    mae = mean_absolute_error(
        predictions["actual_next_day_return"],
        predictions["predicted_return"],
    )

    return {
        "direction_accuracy": float(accuracy),
        "brier_score": float(brier),
        "return_mae": float(mae),
    }


# ==========================================
# 7. MAIN
# ==========================================
def main():
    print("Loading NIFTY 50 feature dataset...")
    df = load_data()

    train_end = int(len(df) * TRAIN_RATIO)

    print(f"Total records: {len(df)}")
    print(f"Training history: {train_end}")
    print(f"Test forecasts: {len(df) - train_end}")
    print(f"Context length: {CONTEXT_LENGTH}")

    pipeline, device = load_chronos()

    print("\nRunning walk-forward forecasts...")

    predictions = forecast_test_period(
        df=df,
        pipeline=pipeline,
        train_end=train_end,
        device=device,
    )

    metrics = evaluate_predictions(predictions)

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    config = {
        "model": MODEL_NAME,
        "train_ratio": TRAIN_RATIO,
        "context_length": CONTEXT_LENGTH,
        "prediction_length": 1,
        "num_samples": NUM_SAMPLES,
        "batch_size": BATCH_SIZE,
        "device": device,
        "test_forecasts": len(predictions),
        "note": (
            "Zero-shot pretrained model with one-step-ahead "
            "walk-forward forecasting."
        ),
    }

    with open(CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write("CHRONOS TIME-SERIES FOUNDATION MODEL BENCHMARK\n")
        file.write("===============================================\n")
        file.write(f"Model: {MODEL_NAME}\n")
        file.write(f"Total source records: {len(df)}\n")
        file.write(f"Training history rows: {train_end}\n")
        file.write(f"Test forecasts: {len(predictions)}\n")
        file.write(f"Context length: {CONTEXT_LENGTH}\n")
        file.write(f"Prediction horizon: 1 trading day\n")
        file.write(f"Device: {device}\n\n")
        file.write(
            f"Direction accuracy: {metrics['direction_accuracy']:.4f}\n"
        )
        file.write(
            f"Brier score: {metrics['brier_score']:.4f}\n"
        )
        file.write(
            f"Return MAE: {metrics['return_mae']:.8f}\n"
        )
        file.write(
            "\nNote: Experimental zero-shot benchmark. "
            "Metrics do not establish profitability or "
            "future performance. Compare with simple baselines "
            "and perform further walk-forward validation.\n"
        )

    print("\nChronos benchmark completed successfully!")
    print(f"Direction accuracy: {metrics['direction_accuracy']:.4f}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Return MAE: {metrics['return_mae']:.8f}")
    print(f"Predictions saved: {PREDICTIONS_PATH}")
    print(f"Config saved: {CONFIG_PATH}")
    print(f"Summary saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
