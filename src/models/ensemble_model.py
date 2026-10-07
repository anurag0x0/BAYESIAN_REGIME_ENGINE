
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]

FILES = {
    "Bayesian": ROOT / "data/processed/bayesian_predictions.csv",
    "SMC": ROOT / "data/processed/smc_predictions.csv",
    "Chronos": ROOT / "data/processed/chronos_predictions.csv",
}

OUTPUT = ROOT / "data/processed/ensemble_predictions.csv"
REPORT = ROOT / "reports/ensemble/ensemble_summary.txt"


def normalize_direction(series, name):
    """Convert direction labels to binary values: Up=1, Down=0."""
    numeric = pd.to_numeric(series, errors="coerce")
    text = series.astype(str).str.strip().str.lower()

    mapped = text.map({
        "up": 1,
        "upward": 1,
        "bullish": 1,
        "positive": 1,
        "down": 0,
        "downward": 0,
        "bearish": 0,
        "negative": 0,
    })

    result = numeric.fillna(mapped)

    invalid = result.notna() & ~result.isin([0, 1])
    if invalid.any():
        raise ValueError(
            f"{name} contains unexpected direction values: "
            f"{series[invalid].unique().tolist()}"
        )

    return result


def load_predictions(name, path):
    if not path.exists():
        raise FileNotFoundError(
            f"{name} prediction file not found: {path}"
        )

    df = pd.read_csv(path)

    if name == "Bayesian":
        date_col = "prediction_for_date"
        prob_col = "probability_positive"
        actual_col = "actual_direction"
    elif name == "SMC":
        date_col = "date"
        prob_col = "probability_up"
        actual_col = "actual_next_day_direction"
    else:
        date_col = "forecast_for_date"
        prob_col = "probability_up"
        actual_col = "actual_next_day_direction"

    required = [date_col, prob_col, actual_col]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"{name} is missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    result = df[required].copy()
    result = result.rename(columns={
        date_col: "date",
        prob_col: f"{name.lower()}_prob",
        actual_col: f"{name.lower()}_actual",
    })

    result["date"] = pd.to_datetime(
        result["date"], errors="coerce"
    ).dt.normalize()

    prob_name = f"{name.lower()}_prob"
    actual_name = f"{name.lower()}_actual"

    result[prob_name] = pd.to_numeric(
        result[prob_name], errors="coerce"
    )
    result[actual_name] = normalize_direction(
        result[actual_name], actual_name
    )

    result = result.dropna(
        subset=["date", prob_name, actual_name]
    )

    result[prob_name] = result[prob_name].clip(0, 1)
    result = result.drop_duplicates(
        subset="date", keep="last"
    )

    return result


def main():
    print("Loading model predictions...")

    bayesian = load_predictions("Bayesian", FILES["Bayesian"])
    smc = load_predictions("SMC", FILES["SMC"])
    chronos = load_predictions("Chronos", FILES["Chronos"])

    print(f"Bayesian rows: {len(bayesian)}")
    print(f"SMC rows: {len(smc)}")
    print(f"Chronos rows: {len(chronos)}")

    # Align model outputs by forecast date.
    df = bayesian.merge(smc, on="date", how="inner")
    df = df.merge(chronos, on="date", how="inner")

    if df.empty:
        raise ValueError(
            "No common forecast dates found across all models. "
            "The SMC date may represent the forecast origin rather "
            "than the forecast-for date. Check the sample rows "
            "from each CSV before proceeding."
        )

    # Cross-check actual labels from each source.
    actual_cols = [
        "bayesian_actual",
        "smc_actual",
        "chronos_actual",
    ]

    mismatch = (
        df[actual_cols].nunique(axis=1, dropna=True) > 1
    )

    if mismatch.any():
        print("\nActual direction mismatch detected.")
        print(
            df.loc[mismatch, ["date"] + actual_cols]
            .head(10).to_string(index=False)
        )
        raise ValueError(
            "Actual direction labels do not match for some dates. "
            "Check the date convention in the prediction files."
        )

    df["actual_direction"] = df["bayesian_actual"].astype(int)

    # Equal-weight probability ensemble.
    prob_cols = [
        "bayesian_prob",
        "smc_prob",
        "chronos_prob",
    ]

    df["ensemble_probability_up"] = df[prob_cols].mean(axis=1)
    df["ensemble_direction"] = (
        df["ensemble_probability_up"] >= 0.5
    ).astype(int)

    y_true = df["actual_direction"]

    results = {}

    for model in ["Bayesian", "SMC", "Chronos"]:
        col = f"{model.lower()}_prob"
        predicted = (df[col] >= 0.5).astype(int)

        results[model] = {
            "accuracy": accuracy_score(y_true, predicted),
            "brier_score": brier_score_loss(y_true, df[col]),
        }

    results["Ensemble"] = {
        "accuracy": accuracy_score(
            y_true, df["ensemble_direction"]
        ),
        "brier_score": brier_score_loss(
            y_true, df["ensemble_probability_up"]
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(OUTPUT, index=False)

    with open(REPORT, "w", encoding="utf-8") as file:
        file.write("ENSEMBLE MODEL EVALUATION\n")
        file.write("=" * 40 + "\n")
        file.write(f"Matched predictions: {len(df)}\n")
        file.write(
            f"Date range: {df['date'].min().date()} to "
            f"{df['date'].max().date()}\n\n"
        )

        for model, metrics in results.items():
            file.write(f"{model}\n")
            file.write(
                f"  Accuracy: {metrics['accuracy']:.4f}\n"
            )
            file.write(
                f"  Brier Score: {metrics['brier_score']:.4f}\n\n"
            )

        file.write(
            "Equal-weight ensemble is an experimental benchmark, "
            "not a validated trading strategy. Historical metrics "
            "do not guarantee future performance.\n"
        )

    print("\nMODEL COMPARISON")
    print("-" * 50)

    for model, metrics in results.items():
        print(
            f"{model:10s} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Brier Score: {metrics['brier_score']:.4f}"
        )

    print(f"\nMatched predictions: {len(df)}")
    print(f"Predictions saved: {OUTPUT}")
    print(f"Summary saved: {REPORT}")


if __name__ == "__main__":
    main()
