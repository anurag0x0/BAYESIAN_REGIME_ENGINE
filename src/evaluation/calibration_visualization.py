
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "evaluation"

CALIBRATION_FILE = REPORT_DIR / "calibration_data.csv"
METRICS_FILE = REPORT_DIR / "model_comparison.csv"


def main():
    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {CALIBRATION_FILE}"
        )

    if not METRICS_FILE.exists():
        raise FileNotFoundError(
            f"File not found: {METRICS_FILE}"
        )

    calibration = pd.read_csv(CALIBRATION_FILE)
    metrics = pd.read_csv(METRICS_FILE)

    required_columns = {
        "model",
        "mean_predicted_probability",
        "observed_up_rate",
        "observations",
    }

    if not required_columns.issubset(calibration.columns):
        raise ValueError("Unexpected calibration data columns.")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Chart 1: Probability calibration curve
    fig, ax = plt.subplots(figsize=(9, 6))

    for model in calibration["model"].dropna().unique():
        subset = calibration[
            calibration["model"] == model
        ].dropna(
            subset=[
                "mean_predicted_probability",
                "observed_up_rate",
            ]
        ).sort_values("mean_predicted_probability")

        if not subset.empty:
            ax.plot(
                subset["mean_predicted_probability"],
                subset["observed_up_rate"],
                marker="o",
                label=model,
            )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    ax.set_title("Model Probability Calibration")
    ax.set_xlabel("Predicted Probability of Up")
    ax.set_ylabel("Observed Up Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    calibration_path = REPORT_DIR / "calibration_curve.png"
    fig.savefig(calibration_path, dpi=150)
    plt.close(fig)

    # Chart 2: Brier score comparison
    required_metrics = {"model", "brier_score"}
    if not required_metrics.issubset(metrics.columns):
        raise ValueError("Unexpected model metrics columns.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(metrics["model"], metrics["brier_score"])
    ax.set_title("Brier Score Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Brier Score (lower is better)")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    brier_path = REPORT_DIR / "brier_score_comparison.png"
    fig.savefig(brier_path, dpi=150)
    plt.close(fig)

    print("Step 12 completed successfully.")
    print(f"Calibration curve: {calibration_path}")
    print(f"Brier score chart: {brier_path}")


if __name__ == "__main__":
    main()
