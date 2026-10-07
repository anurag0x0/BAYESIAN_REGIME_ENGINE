from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports" / "evaluation"

REPORTS.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("STEP 18: STATISTICAL SIGNIFICANCE & ROBUSTNESS")
print("=" * 60)


# ---------------------------------------------------------
# 1. LOAD WALK-FORWARD DATA
# ---------------------------------------------------------

file_path = DATA / "walk_forward_backtest.csv"

df = pd.read_csv(file_path)

if df.empty:
    raise ValueError("Walk-forward dataset is empty.")


print(f"Observations: {len(df):,}")


# ---------------------------------------------------------
# 2. DATE PREPARATION
# ---------------------------------------------------------

df["origin_date"] = pd.to_datetime(
    df["origin_date"]
)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
)

df = df.sort_values(
    "forecast_date"
).reset_index(drop=True)


# ---------------------------------------------------------
# 3. MODEL DEFINITIONS
# ---------------------------------------------------------

models = {
    "Bayesian": "bayesian_probability",
    "SMC": "smc_probability",
    "Ensemble": "ensemble_probability",
}


actual = (
    pd.to_numeric(
        df["actual_direction"],
        errors="coerce"
    )
)

if actual.isna().any():
    raise ValueError(
        "Missing actual direction values."
    )

actual = actual.astype(int).to_numpy()


# ---------------------------------------------------------
# 4. CREATE PREDICTIONS
# ---------------------------------------------------------

predictions = {}

probabilities = {}

for model_name, column in models.items():

    probability = (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
        .to_numpy()
    )

    if not np.isfinite(probability).all():
        raise ValueError(
            f"{model_name}: invalid probability values."
        )

    if (
        (probability < 0).any()
        or
        (probability > 1).any()
    ):
        raise ValueError(
            f"{model_name}: probability outside [0,1]."
        )

    probabilities[model_name] = probability

    predictions[model_name] = (
        probability >= 0.5
    ).astype(int)


# ---------------------------------------------------------
# 5. BASIC PERFORMANCE
# ---------------------------------------------------------

basic_results = []

for model_name in models:

    prediction = predictions[model_name]

    accuracy = np.mean(
        prediction == actual
    )

    brier = np.mean(
        (
            probabilities[model_name]
            - actual
        ) ** 2
    )

    basic_results.append(
        {
            "model": model_name,
            "observations": len(actual),
            "accuracy": accuracy,
            "brier_score": brier,
        }
    )


basic_df = pd.DataFrame(
    basic_results
)


# ---------------------------------------------------------
# 6. MCMARNEYS TEST
# ---------------------------------------------------------

def mcnemar_exact(
    prediction_a,
    prediction_b,
    actual_values
):
    """
    Exact McNemar test using discordant pairs.
    """

    correct_a = (
        prediction_a == actual_values
    )

    correct_b = (
        prediction_b == actual_values
    )

    b = np.sum(
        correct_a & ~correct_b
    )

    c = np.sum(
        ~correct_a & correct_b
    )

    discordant = b + c

    if discordant == 0:

        p_value = 1.0

    else:

        result = binomtest(
            min(b, c),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )

        p_value = result.pvalue

    return b, c, p_value


comparison_results = []


for model_a, model_b in combinations(
    models.keys(),
    2
):

    a = predictions[model_a]
    b = predictions[model_b]

    b_count, c_count, p_value = (
        mcnemar_exact(
            a,
            b,
            actual
        )
    )

    accuracy_a = np.mean(
        a == actual
    )

    accuracy_b = np.mean(
        b == actual
    )

    comparison_results.append(
        {
            "model_a": model_a,
            "model_b": model_b,
            "accuracy_a": accuracy_a,
            "accuracy_b": accuracy_b,
            "accuracy_difference":
                accuracy_a - accuracy_b,
            "discordant_a_correct":
                b_count,
            "discordant_b_correct":
                c_count,
            "mcnemar_p_value":
                p_value,
            "significant_at_0_05":
                p_value < 0.05,
        }
    )


comparison_df = pd.DataFrame(
    comparison_results
)


# ---------------------------------------------------------
# 7. BOOTSTRAP CONFIDENCE INTERVALS
# ---------------------------------------------------------

rng = np.random.default_rng(42)

bootstrap_iterations = 5000


def bootstrap_accuracy(
    prediction,
    actual_values
):

    n = len(actual_values)

    scores = np.empty(
        bootstrap_iterations
    )

    correct = (
        prediction == actual_values
    )

    for i in range(
        bootstrap_iterations
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        scores[i] = np.mean(
            correct[indices]
        )

    lower = np.percentile(
        scores,
        2.5
    )

    upper = np.percentile(
        scores,
        97.5
    )

    return lower, upper


bootstrap_results = []


for model_name in models:

    lower, upper = (
        bootstrap_accuracy(
            predictions[model_name],
            actual
        )
    )

    bootstrap_results.append(
        {
            "model": model_name,
            "accuracy":
                np.mean(
                    predictions[model_name]
                    == actual
                ),
            "bootstrap_ci_lower":
                lower,
            "bootstrap_ci_upper":
                upper,
            "confidence_level": 0.95,
        }
    )


bootstrap_df = pd.DataFrame(
    bootstrap_results
)


# ---------------------------------------------------------
# 8. REGIME ROBUSTNESS
# ---------------------------------------------------------

regime_results = []


if "regime_label" in df.columns:

    for regime, group in df.groupby(
        "regime_label"
    ):

        actual_regime = (
            group["actual_direction"]
            .astype(int)
            .to_numpy()
        )

        for model_name, column in models.items():

            probability = (
                group[column]
                .astype(float)
                .to_numpy()
            )

            prediction = (
                probability >= 0.5
            ).astype(int)

            regime_results.append(
                {
                    "regime": regime,
                    "model": model_name,
                    "observations":
                        len(group),
                    "accuracy":
                        np.mean(
                            prediction
                            == actual_regime
                        ),
                    "brier_score":
                        np.mean(
                            (
                                probability
                                - actual_regime
                            ) ** 2
                        ),
                }
            )


regime_df = pd.DataFrame(
    regime_results
)


# ---------------------------------------------------------
# 9. SAVE RESULTS
# ---------------------------------------------------------

basic_file = (
    REPORTS /
    "statistical_basic_performance.csv"
)

comparison_file = (
    REPORTS /
    "model_significance_tests.csv"
)

bootstrap_file = (
    REPORTS /
    "bootstrap_confidence_intervals.csv"
)

regime_file = (
    REPORTS /
    "statistical_regime_robustness.csv"
)


basic_df.to_csv(
    basic_file,
    index=False
)

comparison_df.to_csv(
    comparison_file,
    index=False
)

bootstrap_df.to_csv(
    bootstrap_file,
    index=False
)

regime_df.to_csv(
    regime_file,
    index=False
)


# ---------------------------------------------------------
# 10. FINAL REPORT
# ---------------------------------------------------------

report_file = (
    REPORTS /
    "statistical_robustness_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STATISTICAL SIGNIFICANCE & ROBUSTNESS REPORT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Observations: {len(df)}\n"
    )

    f.write(
        "Bootstrap iterations: "
        f"{bootstrap_iterations}\n\n"
    )

    f.write(
        "BASIC PERFORMANCE\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        basic_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    f.write(
        "\n\nPAIRWISE MCMARNEYS TEST\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        comparison_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    f.write(
        "\n\nBOOTSTRAP 95% CONFIDENCE INTERVALS\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        bootstrap_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    f.write(
        "\n\nREGIME ROBUSTNESS\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        regime_df.to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )

    f.write(
        "\n\nINTERPRETATION\n"
    )

    f.write(
        "=" * 60 + "\n"
    )

    f.write(
        "A p-value below 0.05 indicates evidence of a "
        "difference in paired classification outcomes "
        "under the exact McNemar test. It does not prove "
        "that one model will outperform in the future.\n\n"
    )

    f.write(
        "Bootstrap intervals quantify uncertainty around "
        "historical accuracy. Overlapping intervals should "
        "not be interpreted as a formal equivalence test.\n\n"
    )

    f.write(
        "Regime robustness is descriptive and should not "
        "be interpreted as proof of profitability.\n"
    )


# ---------------------------------------------------------
# 11. PRINT RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("BASIC PERFORMANCE")
print("=" * 60)

print(
    basic_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


print("\n" + "=" * 60)
print("PAIRWISE SIGNIFICANCE TESTS")
print("=" * 60)

print(
    comparison_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


print("\n" + "=" * 60)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS")
print("=" * 60)

print(
    bootstrap_df.to_string(
        index=False,
        float_format=lambda x:
        f"{x:.4f}"
    )
)


print("\nBasic performance:")
print(basic_file)

print("\nSignificance tests:")
print(comparison_file)

print("\nBootstrap:")
print(bootstrap_file)

print("\nRegime robustness:")
print(regime_file)

print("\nReport:")
print(report_file)


print("\n" + "=" * 60)
print("STEP 18 COMPLETED")
print("=" * 60)