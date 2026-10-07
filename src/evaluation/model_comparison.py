
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
REPORT = ROOT / "reports" / "evaluation"

def load_csv(filename):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)

def parse_date(df, column):
    df[column] = pd.to_datetime(df[column], errors="coerce").dt.normalize()
    if df[column].isna().any():
        raise ValueError(f"Invalid dates in {column}")
    return df

def prepare_data():
    bayes = parse_date(load_csv("bayesian_predictions.csv"), "prediction_for_date")
    smc = parse_date(load_csv("smc_predictions.csv"), "date")
    chronos = parse_date(load_csv("chronos_predictions.csv"), "forecast_for_date")
    ensemble = parse_date(load_csv("ensemble_predictions.csv"), "forecast_date")
    features = parse_date(load_csv("nifty50_features.csv"), "date")

    # Map SMC origin date to the next available trading date.
    dates = (
        features["date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    next_date = dict(zip(dates.iloc[:-1], dates.iloc[1:]))
    smc["forecast_date"] = smc["date"].map(next_date)
    smc = smc.dropna(subset=["forecast_date"])

    bayes = bayes[
        ["prediction_for_date", "probability_positive", "actual_direction"]
    ].rename(columns={
        "prediction_for_date": "forecast_date",
        "probability_positive": "bayes_prob",
        "actual_direction": "bayes_actual",
    })

    smc = smc[
        ["forecast_date", "probability_up", "actual_next_day_direction"]
    ].rename(columns={
        "probability_up": "smc_prob",
        "actual_next_day_direction": "smc_actual",
    })

    chronos = chronos[
        ["forecast_for_date", "probability_up", "actual_next_day_direction"]
    ].rename(columns={
        "forecast_for_date": "forecast_date",
        "probability_up": "chronos_prob",
        "actual_next_day_direction": "chronos_actual",
    })

    ensemble = ensemble[
        ["forecast_date", "ensemble_probability", "actual_direction"]
    ].rename(columns={
        "ensemble_probability": "ensemble_prob",
        "actual_direction": "ensemble_actual",
    })

    data = bayes.merge(smc, on="forecast_date", how="inner")
    data = data.merge(chronos, on="forecast_date", how="inner")
    data = data.merge(ensemble, on="forecast_date", how="inner")

    if data.empty:
        raise ValueError("No common forecast dates found.")

    actual_cols = [
        "bayes_actual", "smc_actual", "chronos_actual", "ensemble_actual"
    ]
    mismatch = data[actual_cols].nunique(axis=1) != 1

    if mismatch.any():
        print(data.loc[mismatch, ["forecast_date"] + actual_cols].head(10))
        raise ValueError("Actual labels mismatch across models.")

    data["actual"] = data["bayes_actual"].astype(int)

    prob_cols = ["bayes_prob", "smc_prob", "chronos_prob", "ensemble_prob"]
    for col in prob_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        if data[col].isna().any() or not data[col].between(0, 1).all():
            raise ValueError(f"Invalid probabilities in {col}")

    return data.sort_values("forecast_date").reset_index(drop=True)

def evaluate(name, actual, probability):
    predicted = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    auc = roc_auc_score(actual, probability) if len(np.unique(actual)) == 2 else np.nan
    safe_prob = np.clip(probability, 1e-15, 1 - 1e-15)

    metrics = {
        "model": name,
        "observations": len(actual),
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1_score": f1_score(actual, predicted, zero_division=0),
        "brier_score": brier_score_loss(actual, probability),
        "roc_auc": auc,
        "log_loss": log_loss(actual, safe_prob, labels=[0, 1]),
    }

    matrix = {
        "model": name,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }
    return metrics, matrix

def calibration_table(name, actual, probability):
    df = pd.DataFrame({"actual": actual, "probability": probability})
    df["bin"] = pd.cut(
        df["probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    result = df.groupby("bin", observed=True).agg(
        mean_predicted_probability=("probability", "mean"),
        observed_up_rate=("actual", "mean"),
        observations=("actual", "size"),
    ).reset_index()
    result["model"] = name
    result["bin"] = result["bin"].astype(str)
    return result[[
        "model", "bin", "mean_predicted_probability",
        "observed_up_rate", "observations"
    ]]

def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    print("Loading and aligning predictions...")

    data = prepare_data()
    actual = data["actual"].to_numpy()
    print(f"Common observations: {len(data)}")

    models = {
        "Bayesian": "bayes_prob",
        "SMC": "smc_prob",
        "Chronos": "chronos_prob",
        "Ensemble": "ensemble_prob",
    }

    metric_rows = []
    matrix_rows = []
    calibration_rows = []

    for name, column in models.items():
        probability = data[column].to_numpy()
        metrics, matrix = evaluate(name, actual, probability)
        metric_rows.append(metrics)
        matrix_rows.append(matrix)
        calibration_rows.append(
            calibration_table(name, actual, probability)
        )

    metrics_df = pd.DataFrame(metric_rows)
    matrices_df = pd.DataFrame(matrix_rows)
    calibration_df = pd.concat(calibration_rows, ignore_index=True)

    metrics_df.to_csv(REPORT / "model_comparison.csv", index=False)
    matrices_df.to_csv(REPORT / "confusion_matrices.csv", index=False)
    calibration_df.to_csv(REPORT / "calibration_data.csv", index=False)

    summary = [
        "MODEL EVALUATION & COMPARISON",
        "==============================",
        f"Common observations: {len(data)}",
        f"Forecast start: {data['forecast_date'].min().date()}",
        f"Forecast end: {data['forecast_date'].max().date()}",
        "",
        "METRICS",
        metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        "",
        "CONFUSION MATRICES",
        matrices_df.to_string(index=False),
        "",
        "Historical evaluation only; results do not guarantee future performance.",
    ]
    (REPORT / "model_comparison.txt").write_text(
        "\n".join(summary), encoding="utf-8"
    )

    print("\nMODEL COMPARISON")
    print(metrics_df.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    ))
    print("\nCONFUSION MATRICES")
    print(matrices_df.to_string(index=False))
    print("\nEvaluation completed.")
    print(f"Reports saved in: {REPORT}")

if __name__ == "__main__":
    main()
