from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "nifty50_features.csv"
REPORT_DIR = ROOT / "reports" / "evaluation"
REPORT_PATH = REPORT_DIR / "robustness_leakage_audit.txt"

TARGET = "target_next_day_direction"
DATE = "date"
RETURN = "daily_return"


def main():
    print("=" * 60)
    print("STEP 13: ROBUSTNESS & DATA LEAKAGE AUDIT")
    print("=" * 60)

    if not DATA_PATH.exists():
        print(f"Dataset not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    errors = []
    warnings = []

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # Check required columns
    required = [DATE, RETURN, TARGET]
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"Missing required columns: {missing}")
        return

    # Date checks
    df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
    df = df.sort_values(DATE).reset_index(drop=True)

    invalid_dates = df[DATE].isna().sum()
    duplicate_dates = df[DATE].duplicated().sum()

    print("\n[1] DATE QUALITY")
    print(f"Invalid dates: {invalid_dates}")
    print(f"Duplicate dates: {duplicate_dates}")

    if invalid_dates:
        errors.append(f"Invalid dates found: {invalid_dates}")

    if duplicate_dates:
        errors.append(f"Duplicate dates found: {duplicate_dates}")

    # Missing values
    print("\n[2] MISSING VALUES")
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if missing_values.empty:
        print("No missing values found.")
    else:
        print(missing_values.to_string())
        warnings.append("Missing values detected.")

    # Target alignment
    print("\n[3] TARGET ALIGNMENT")

    returns = pd.to_numeric(df[RETURN], errors="coerce")
    target = pd.to_numeric(df[TARGET], errors="coerce")

    expected = (returns.shift(-1) > 0).astype("float")
    valid = target.notna() & expected.notna()
    valid.iloc[-1] = False

    if valid.sum() > 0 and set(target[valid].unique()).issubset({0, 1}):
        match_rate = (
            target[valid].astype(int) == expected[valid].astype(int)
        ).mean()

        print(f"Alignment match rate: {match_rate:.2%}")

        if match_rate < 0.999:
            warnings.append(
                "Target alignment mismatch. Verify target creation logic."
            )
    else:
        print("Could not automatically verify target alignment.")
        warnings.append("Manual target verification required.")

    # Suspicious feature names
    print("\n[4] FUTURE-LOOKING COLUMNS")

    suspicious_terms = [
        "future", "forward", "lead_", "next_day",
        "nextday", "lookahead", "target", "label"
    ]

    features = [c for c in df.columns if c not in [DATE, TARGET]]

    suspicious = [
        c for c in features
        if any(term in c.lower() for term in suspicious_terms)
    ]

    if suspicious:
        print("Review these columns:")
        for col in suspicious:
            print(f"- {col}")
        warnings.append(f"Review suspicious columns: {suspicious}")
    else:
        print("No suspicious column names detected.")

    # Chronological split
    print("\n[5] CHRONOLOGICAL TRAIN-TEST SPLIT")

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")

    if not train.empty and not test.empty:
        chronological = train[DATE].max() < test[DATE].min()
        print(f"Chronological order valid: {chronological}")

        if not chronological:
            errors.append("Train-test split is not chronological.")

    # Duplicate rows
    print("\n[6] DUPLICATE ROWS")

    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    if duplicates:
        warnings.append(f"Duplicate rows found: {duplicates}")

    # Model implementation warnings
    warnings.extend([
        "Verify that scalers are fitted on training data only.",
        "Ensure target_next_day_direction is never used as a feature.",
        "Verify that all input features are available at forecast time.",
        "HMM model.predict on the full sequence may create retrospective "
        "regime labels. Use causal walk-forward filtering for live forecasts.",
        "This audit does not prove all model code is leakage-free."
    ])

    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = [
        "STEP 13: ROBUSTNESS & DATA LEAKAGE AUDIT",
        "=" * 60,
        f"Rows: {len(df)}",
        f"Columns: {len(df.columns)}",
        "",
        "ERRORS:"
    ]

    report += [f"- {e}" for e in errors] if errors else ["- None detected"]

    report += ["", "WARNINGS:"]
    report += [f"- {w}" for w in warnings] if warnings else ["- None"]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\n" + "=" * 60)

    if errors:
        print("AUDIT: ISSUES FOUND")
    elif warnings:
        print("AUDIT: MANUAL REVIEW REQUIRED")
    else:
        print("AUDIT: PASSED")

    print(f"Report saved: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()