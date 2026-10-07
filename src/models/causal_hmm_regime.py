
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp


# --------------------------------------------------
# PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "nifty50_features.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "nifty50_regimes_causal.csv"
REPORT_PATH = ROOT / "reports" / "evaluation" / "causal_hmm_summary.txt"

DATE_COL = "date"
RETURN_COL = "daily_return"
TARGET_COL = "target_next_day_direction"

N_STATES = 3
TRAIN_RATIO = 0.80
RANDOM_STATE = 42


# --------------------------------------------------
# FIND VOLATILITY FEATURE
# --------------------------------------------------

def find_volatility_column(columns):
    candidates = [
        col for col in columns
        if "volatility" in col.lower()
        or "rolling_vol" in col.lower()
        or "vol_" in col.lower()
    ]

    # Prefer a rolling volatility feature if multiple exist.
    candidates = sorted(
        candidates,
        key=lambda col: (
            "rolling" not in col.lower(),
            len(col)
        )
    )

    return candidates[0] if candidates else None


# --------------------------------------------------
# CAUSAL FORWARD FILTERING
# --------------------------------------------------

def forward_filter(model, X):
    """
    Calculate filtered probabilities using observations
    only up to and including each current row.

    Does not use future observations to infer past states.
    """
    log_emissions = model._compute_log_likelihood(X)

    start_prob = np.asarray(model.startprob_)
    transition = np.asarray(model.transmat_)

    n_rows, n_states = log_emissions.shape
    filtered = np.zeros((n_rows, n_states), dtype=float)

    # Initial observation
    log_alpha = (
        np.log(np.maximum(start_prob, 1e-300))
        + log_emissions[0]
    )
    log_alpha -= logsumexp(log_alpha)
    filtered[0] = np.exp(log_alpha)

    # Sequential forward recursion
    for t in range(1, n_rows):
        log_transition = np.log(
            np.maximum(transition, 1e-300)
        )

        log_alpha = (
            logsumexp(
                log_alpha[:, None] + log_transition,
                axis=0
            )
            + log_emissions[t]
        )

        log_alpha -= logsumexp(log_alpha)
        filtered[t] = np.exp(log_alpha)

    return filtered


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():
    print("=" * 65)
    print("STEP 14: CAUSAL HMM REGIME DETECTION")
    print("=" * 65)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = [DATE_COL, RETURN_COL]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL], errors="coerce"
    )

    df = (
        df.dropna(subset=[DATE_COL, RETURN_COL])
        .sort_values(DATE_COL)
        .drop_duplicates(subset=[DATE_COL])
        .reset_index(drop=True)
    )

    # Choose available volatility feature.
    volatility_col = find_volatility_column(df.columns)

    if volatility_col is None:
        volatility_col = "abs_daily_return"
        df[volatility_col] = pd.to_numeric(
            df[RETURN_COL], errors="coerce"
        ).abs()
        print(
            "No volatility column found. "
            "Using absolute daily return as volatility proxy."
        )

    print(f"Rows: {len(df):,}")
    print(f"Return feature: {RETURN_COL}")
    print(f"Volatility feature: {volatility_col}")

    # Only observable, non-target features are used.
    feature_cols = [RETURN_COL, volatility_col]
    X_df = df[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    valid_rows = X_df.notna().all(axis=1)
    df = df.loc[valid_rows].reset_index(drop=True)
    X_df = X_df.loc[valid_rows].reset_index(drop=True)

    X = X_df.to_numpy(dtype=float)

    if len(X) < 100:
        raise ValueError(
            "Not enough rows to fit the HMM reliably."
        )

    # Chronological training split.
    split_idx = int(len(X) * TRAIN_RATIO)

    X_train = X[:split_idx]

    if split_idx <= N_STATES:
        raise ValueError("Training dataset is too small.")

    # Fit scaler on training data only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform all rows using training-only scaler.
    X_scaled = scaler.transform(X)

    # Fit HMM on training period only.
    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_STATE,
        min_covar=1e-5
    )

    model.fit(X_train_scaled)

    # Filter sequentially across train and test periods.
    # Test observations update current state probabilities,
    # but do not refit model parameters.
    probabilities = forward_filter(model, X_scaled)

    # Determine each hidden state's volatility using
    # training-period filtered probabilities only.
    train_probs = probabilities[:split_idx]
    train_volatility = X[:split_idx, 1]

    state_volatility = (
        train_probs.T @ train_volatility
    ) / np.maximum(train_probs.sum(axis=0), 1e-12)

    # Sort states by estimated volatility.
    ordered_states = np.argsort(state_volatility)

    state_to_regime = {
        int(ordered_states[0]): "Low Volatility",
        int(ordered_states[1]): "Medium Volatility",
        int(ordered_states[2]): "High Volatility"
    }

    # Reorder posterior columns as low, medium, high.
    ordered_probs = probabilities[:, ordered_states]

    # Numeric regime is 0=Low, 1=Medium, 2=High.
    predicted_state = np.argmax(probabilities, axis=1)
    regime_numeric = np.array([
        int(np.where(ordered_states == state)[0][0])
        for state in predicted_state
    ])

    regime_labels = [
        state_to_regime[int(state)]
        for state in predicted_state
    ]

    # Output table.
    result = pd.DataFrame({
        DATE_COL: df[DATE_COL],
        "regime": regime_numeric,
        "regime_label": regime_labels,
        "regime_probability": probabilities.max(axis=1),
        "prob_low_volatility": ordered_probs[:, 0],
        "prob_medium_volatility": ordered_probs[:, 1],
        "prob_high_volatility": ordered_probs[:, 2],
        "daily_return": X[:, 0],
        "volatility_input": X[:, 1],
        "data_split": [
            "train" if i < split_idx else "test"
            for i in range(len(df))
        ]
    })

    # Save outputs.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    result.to_csv(OUTPUT_PATH, index=False)

    # Basic diagnostics.
    regime_counts = result["regime_label"].value_counts()
    test_result = result.iloc[split_idx:]

    report_lines = [
        "CAUSAL HMM REGIME DETECTION SUMMARY",
        "=" * 55,
        f"Total rows: {len(result)}",
        f"Training rows: {split_idx}",
        f"Testing rows: {len(result) - split_idx}",
        f"Features: {feature_cols}",
        f"Volatility feature: {volatility_col}",
        "",
        "Training-only estimated state volatility:"
    ]

    for state in ordered_states:
        label = state_to_regime[int(state)]
        report_lines.append(
            f"{label}: {state_volatility[state]:.8f}"
        )

    report_lines += [
        "",
        "Regime counts (all rows):",
        regime_counts.to_string(),
        "",
        "Test-period regime counts:",
        test_result["regime_label"].value_counts().to_string(),
        "",
        "Method: causal forward filtering",
        "HMM parameters fitted on training period only.",
        "Scaler fitted on training period only.",
        "Future target excluded from model inputs.",
        "",
        "Limitation:",
        "Filtered probabilities are causal conditional estimates,",
        "but hidden-state meanings and fitted parameters can drift",
        "over time. No trading profitability is implied."
    ]

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8"
    )

    print("\n[RESULT]")
    print(f"Training rows: {split_idx:,}")
    print(f"Testing rows: {len(result) - split_idx:,}")
    print("\nRegime distribution:")
    print(regime_counts.to_string())

    print("\n[OUTPUT]")
    print(f"Predictions: {OUTPUT_PATH}")
    print(f"Summary: {REPORT_PATH}")
    print("\nStep 14 completed.")


if __name__ == "__main__":
    main()
