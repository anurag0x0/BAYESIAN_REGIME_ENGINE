from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "evaluation"

REPORTS.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("STEP 16: CAUSAL REGIME INTEGRATION")
print("=" * 60)

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

regimes = pd.read_csv(
    DATA / "nifty50_regimes_causal.csv"
)

bayesian = pd.read_csv(
    DATA / "bayesian_predictions.csv"
)

smc = pd.read_csv(
    DATA / "smc_predictions.csv"
)

ensemble = pd.read_csv(
    DATA / "ensemble_predictions.csv"
)

print(f"Causal regime rows: {len(regimes):,}")
print(f"Bayesian prediction rows: {len(bayesian):,}")
print(f"SMC prediction rows: {len(smc):,}")
print(f"Ensemble prediction rows: {len(ensemble):,}")

# ---------------------------------------------------------
# 2. DATE CONVERSION
# ---------------------------------------------------------

regimes["date"] = pd.to_datetime(regimes["date"])

bayesian["feature_date"] = pd.to_datetime(
    bayesian["feature_date"]
)

bayesian["prediction_for_date"] = pd.to_datetime(
    bayesian["prediction_for_date"]
)

smc["date"] = pd.to_datetime(smc["date"])

ensemble["forecast_date"] = pd.to_datetime(
    ensemble["forecast_date"]
)

# ---------------------------------------------------------
# 3. KEEP ONLY TEST PERIOD
# ---------------------------------------------------------

bayesian_test = bayesian[
    bayesian["prediction_for_date"]
    >= pd.Timestamp("2023-06-15")
].copy()

smc_test = smc[
    smc["split"].astype(str).str.lower() == "test"
].copy()

ensemble_test = ensemble.copy()

print("\nTest datasets:")
print(f"Bayesian: {len(bayesian_test):,}")
print(f"SMC: {len(smc_test):,}")
print(f"Ensemble: {len(ensemble_test):,}")

# ---------------------------------------------------------
# 4. CREATE ORIGIN -> FORECAST DATE MAPPING
# ---------------------------------------------------------

# SMC date represents forecast origin.
# The next trading date is the forecast target.

smc_test = smc_test.sort_values("date").copy()

smc_test["forecast_date"] = (
    smc_test["date"].shift(-1)
)

smc_test = smc_test.dropna(
    subset=["forecast_date"]
)

# ---------------------------------------------------------
# 5. PREPARE BAYESIAN
# ---------------------------------------------------------

bayesian_test = bayesian_test.rename(
    columns={
        "prediction_for_date": "forecast_date",
        "probability_positive":
            "bayesian_probability",
        "actual_direction":
            "bayesian_actual"
    }
)

bayesian_test = bayesian_test[
    [
        "feature_date",
        "forecast_date",
        "bayesian_probability",
        "bayesian_actual"
    ]
]

# ---------------------------------------------------------
# 6. PREPARE SMC
# ---------------------------------------------------------

smc_test = smc_test.rename(
    columns={
        "date": "origin_date",
        "probability_up":
            "smc_probability",
        "actual_next_day_direction":
            "smc_actual"
    }
)

smc_test = smc_test[
    [
        "origin_date",
        "forecast_date",
        "smc_probability",
        "smc_actual"
    ]
]

# ---------------------------------------------------------
# 7. PREPARE ENSEMBLE
# ---------------------------------------------------------

ensemble_test = ensemble_test.rename(
    columns={
        "ensemble_probability":
            "ensemble_probability",
        "actual_direction":
            "ensemble_actual"
    }
)

ensemble_test = ensemble_test[
    [
        "forecast_date",
        "ensemble_probability",
        "ensemble_actual"
    ]
]

# ---------------------------------------------------------
# 8. MERGE MODELS
# ---------------------------------------------------------

merged = bayesian_test.merge(
    ensemble_test,
    on="forecast_date",
    how="inner"
)

merged = merged.merge(
    smc_test,
    on="forecast_date",
    how="inner"
)

print(
    f"\nCommon model forecast dates: "
    f"{len(merged):,}"
)

# ---------------------------------------------------------
# 9. ADD CAUSAL HMM REGIME
# ---------------------------------------------------------

# IMPORTANT:
# Use the forecast ORIGIN date to attach regime.
# We do NOT use the future forecast date's regime.

merged = merged.merge(
    regimes[
        [
            "date",
            "regime",
            "regime_label",
            "regime_probability",
            "prob_low_volatility",
            "prob_medium_volatility",
            "prob_high_volatility"
        ]
    ],
    left_on="origin_date",
    right_on="date",
    how="inner"
)

merged = merged.drop(
    columns=["date"]
)

print(
    f"After causal regime alignment: "
    f"{len(merged):,}"
)

# ---------------------------------------------------------
# 10. CHECK ACTUAL LABEL CONSISTENCY
# ---------------------------------------------------------

actual_columns = [
    "bayesian_actual",
    "smc_actual",
    "ensemble_actual"
]

for column in actual_columns:
    merged[column] = pd.to_numeric(
        merged[column],
        errors="coerce"
    )

label_consistency = (
    merged[actual_columns]
    .nunique(axis=1)
    .eq(1)
).all()

print(
    f"Actual labels consistent: "
    f"{label_consistency}"
)

if not label_consistency:
    raise ValueError(
        "Actual direction labels do not match."
    )

merged["actual_direction"] = (
    merged["ensemble_actual"].astype(int)
)

# ---------------------------------------------------------
# 11. PROBABILITY VALIDATION
# ---------------------------------------------------------

probability_columns = [
    "bayesian_probability",
    "smc_probability",
    "ensemble_probability"
]

for column in probability_columns:
    merged[column] = pd.to_numeric(
        merged[column],
        errors="coerce"
    )

probability_array = merged[
    probability_columns
].to_numpy()

probabilities_valid = (
    np.isfinite(probability_array).all()
    and
    (probability_array >= 0).all()
    and
    (probability_array <= 1).all()
)

print(
    f"Model probabilities valid: "
    f"{probabilities_valid}"
)

if not probabilities_valid:
    raise ValueError(
        "Invalid probability values detected."
    )

# ---------------------------------------------------------
# 12. REGIME-WISE PERFORMANCE
# ---------------------------------------------------------

models = {
    "Bayesian": "bayesian_probability",
    "SMC": "smc_probability",
    "Ensemble": "ensemble_probability"
}

results = []

for regime_name, group in merged.groupby(
    "regime_label"
):

    actual = group[
        "actual_direction"
    ].to_numpy()

    for model_name, probability_column in models.items():

        probability = group[
            probability_column
        ].to_numpy()

        prediction = (
            probability >= 0.5
        ).astype(int)

        accuracy = np.mean(
            prediction == actual
        )

        brier = np.mean(
            (probability - actual) ** 2
        )

        results.append(
            {
                "regime": regime_name,
                "model": model_name,
                "observations": len(group),
                "accuracy": accuracy,
                "brier_score": brier,
                "actual_positive_rate":
                    np.mean(actual),
                "mean_probability":
                    np.mean(probability)
            }
        )

performance = pd.DataFrame(results)

# ---------------------------------------------------------
# 13. SAVE INTEGRATED DATA
# ---------------------------------------------------------

output_file = (
    DATA /
    "causal_regime_integrated_predictions.csv"
)

merged.to_csv(
    output_file,
    index=False
)

# ---------------------------------------------------------
# 14. SAVE PERFORMANCE
# ---------------------------------------------------------

performance_file = (
    REPORTS /
    "causal_regime_performance.csv"
)

performance.to_csv(
    performance_file,
    index=False
)

# ---------------------------------------------------------
# 15. CREATE REPORT
# ---------------------------------------------------------

report_file = (
    REPORTS /
    "causal_regime_integration.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "CAUSAL REGIME INTEGRATION REPORT\n"
    )

    file.write(
        "=" * 55 + "\n\n"
    )

    file.write(
        f"Common forecast dates: "
        f"{len(merged)}\n"
    )

    file.write(
        f"Actual labels consistent: "
        f"{label_consistency}\n"
    )

    file.write(
        f"Model probabilities valid: "
        f"{probabilities_valid}\n\n"
    )

    file.write(
        "REGIME-WISE PERFORMANCE\n"
    )

    file.write(
        "=" * 55 + "\n"
    )

    file.write(
        performance.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    file.write(
        "\n\nLIMITATION\n"
    )

    file.write(
        "This analysis evaluates model performance "
        "conditional on causal HMM regimes. It does "
        "not establish profitability or guarantee "
        "future predictive performance.\n"
    )

# ---------------------------------------------------------
# 16. FINAL OUTPUT
# ---------------------------------------------------------

print("\n" + "=" * 60)

print("REGIME-WISE PERFORMANCE")
print("=" * 60)

print(
    performance.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)

print("\nIntegrated predictions:")
print(output_file)

print("\nPerformance:")
print(performance_file)

print("\nReport:")
print(report_file)

print("\n" + "=" * 60)
print("STEP 16 COMPLETED")
print("=" * 60)