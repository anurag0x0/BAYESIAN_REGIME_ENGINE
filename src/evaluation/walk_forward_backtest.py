from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "evaluation"

REPORTS.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 17: WALK-FORWARD BACKTEST")
print("=" * 60)

# Load Step 16 output
file_path = DATA / "causal_regime_integrated_predictions.csv"

df = pd.read_csv(file_path)

if df.empty:
    raise ValueError("Integrated prediction file is empty.")

print(f"Loaded observations: {len(df):,}")

# Dates
df["origin_date"] = pd.to_datetime(df["origin_date"])
df["forecast_date"] = pd.to_datetime(df["forecast_date"])

df = df.sort_values("forecast_date").reset_index(drop=True)

# Required columns
required = [
    "origin_date",
    "forecast_date",
    "bayesian_probability",
    "smc_probability",
    "ensemble_probability",
    "actual_direction",
]

missing = [c for c in required if c not in df.columns]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Chronological check
chronological = df["forecast_date"].is_monotonic_increasing

print(f"Chronological order: {chronological}")

if not chronological:
    raise ValueError("Forecast dates are not chronological.")

# Show date relationship
df["days_difference"] = (
    df["forecast_date"] - df["origin_date"]
).dt.days

print("\nDate difference:")
print(df["days_difference"].value_counts().sort_index())

print("\nFirst 5 date pairs:")
print(
    df[
        ["origin_date", "forecast_date", "days_difference"]
    ].head().to_string(index=False)
)

# ---------------------------------------------------------
# IMPORTANT:
# Step 16 contains the regime at the forecast origin.
# We validate the chronological forecast sequence itself.
# We do not call the model leakage-free merely because
# origin/target dates are present.
# ---------------------------------------------------------

valid_sequence = (
    df["forecast_date"].diff().dropna() > pd.Timedelta(0)
).all()

print(
    f"\nForecast sequence strictly increasing: "
    f"{valid_sequence}"
)

if not valid_sequence:
    raise ValueError(
        "Forecast sequence is not strictly increasing."
    )

# Probability checks
models = {
    "Bayesian": "bayesian_probability",
    "SMC": "smc_probability",
    "Ensemble": "ensemble_probability",
}

for name, column in models.items():

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    if values.isna().any():
        raise ValueError(
            f"{name}: missing probability values."
        )

    if not np.isfinite(values).all():
        raise ValueError(
            f"{name}: non-finite probability values."
        )

    if ((values < 0) | (values > 1)).any():
        raise ValueError(
            f"{name}: probability outside [0,1]."
        )

print("\nProbability validation: PASS")

# Actual labels
actual = pd.to_numeric(
    df["actual_direction"],
    errors="coerce"
)

if actual.isna().any():
    raise ValueError("Missing actual direction values.")

actual = actual.astype(int).to_numpy()

# Overall performance
results = []

for model_name, column in models.items():

    probability = df[column].astype(float).to_numpy()

    prediction = (
        probability >= 0.5
    ).astype(int)

    accuracy = np.mean(
        prediction == actual
    )

    brier = np.mean(
        (probability - actual) ** 2
    )

    results.append({
        "model": model_name,
        "observations": len(actual),
        "accuracy": accuracy,
        "brier_score": brier,
    })

results_df = pd.DataFrame(results)

# Year-wise performance
df["year"] = df["forecast_date"].dt.year

yearly_results = []

for year, group in df.groupby("year"):

    y = group["actual_direction"].astype(int).to_numpy()

    for model_name, column in models.items():

        p = group[column].astype(float).to_numpy()

        prediction = (
            p >= 0.5
        ).astype(int)

        yearly_results.append({
            "year": year,
            "model": model_name,
            "observations": len(group),
            "accuracy": np.mean(
                prediction == y
            ),
            "brier_score": np.mean(
                (p - y) ** 2
            ),
        })

yearly_df = pd.DataFrame(yearly_results)

# Save predictions
walk_forward_file = (
    DATA / "walk_forward_backtest.csv"
)

df[
    [
        "origin_date",
        "forecast_date",
        "regime_label",
        "bayesian_probability",
        "smc_probability",
        "ensemble_probability",
        "actual_direction",
    ]
].to_csv(
    walk_forward_file,
    index=False
)

# Save summary
summary_file = (
    REPORTS / "walk_forward_summary.csv"
)

results_df.to_csv(
    summary_file,
    index=False
)

yearly_file = (
    REPORTS / "walk_forward_yearly.csv"
)

yearly_df.to_csv(
    yearly_file,
    index=False
)

# Report
report_file = (
    REPORTS / "walk_forward_backtest.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("WALK-FORWARD BACKTEST REPORT\n")
    f.write("=" * 55 + "\n\n")

    f.write(
        f"Observations: {len(df)}\n"
    )

    f.write(
        f"Chronological order: {chronological}\n"
    )

    f.write(
        f"Forecast sequence valid: {valid_sequence}\n\n"
    )

    f.write("OVERALL PERFORMANCE\n")
    f.write("=" * 55 + "\n")

    f.write(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    f.write("\n\nYEAR-WISE PERFORMANCE\n")
    f.write("=" * 55 + "\n")

    f.write(
        yearly_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    f.write("\n\nLIMITATION\n")
    f.write(
        "This is a historical sequential evaluation. "
        "It does not establish future profitability. "
        "Transaction costs, slippage, market impact and "
        "position sizing are not modeled.\n"
    )

print("\n" + "=" * 60)
print("WALK-FORWARD RESULTS")
print("=" * 60)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nWalk-forward file:")
print(walk_forward_file)

print("\nSummary:")
print(summary_file)

print("\nYearly results:")
print(yearly_file)

print("\nReport:")
print(report_file)

print("\n" + "=" * 60)
print("STEP 17 COMPLETED")
print("=" * 60)