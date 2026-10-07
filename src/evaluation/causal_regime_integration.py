
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "nifty50_features.csv"
PRED_PATH = ROOT / "data" / "processed" / "nifty50_regimes_causal.csv"
REPORT_PATH = ROOT / "reports" / "evaluation" / "causal_hmm_validation.txt"

DATE_COL = "date"
RETURN_COL = "daily_return"
TRAIN_RATIO = 0.80
N_STATES = 3
RANDOM_STATE = 42


def find_volatility_column(columns):
    candidates = [
        c for c in columns
        if "volatility" in c.lower()
        or "rolling_vol" in c.lower()
        or "vol_" in c.lower()
    ]
    candidates = sorted(
        candidates,
        key=lambda c: ("rolling" not in c.lower(), len(c))
    )
    return candidates[0] if candidates else None


def forward_filter(model, X):
    """Estimate hidden-state probabilities using only past/current rows."""
    log_emissions = model._compute_log_likelihood(X)
    log_transition = np.log(np.maximum(model.transmat_, 1e-300))
    log_alpha = (
        np.log(np.maximum(model.startprob_, 1e-300))
        + log_emissions[0]
    )
    log_alpha -= logsumexp(log_alpha)

    probabilities = np.zeros((len(X), model.n_components))
    probabilities[0] = np.exp(log_alpha)

    for t in range(1, len(X)):
        log_alpha = (
            logsumexp(log_alpha[:, None] + log_transition, axis=0)
            + log_emissions[t]
        )
        log_alpha -= logsumexp(log_alpha)
        probabilities[t] = np.exp(log_alpha)

    return probabilities


def main():
    print("=" * 60)
    print("STEP 15: CAUSAL HMM VALIDATION")
    print("=" * 60)

    for path in [DATA_PATH, PRED_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file missing: {path}")

    df = pd.read_csv(DATA_PATH)
    saved = pd.read_csv(PRED_PATH)

    df.columns = df.columns.str.strip()
    saved.columns = saved.columns.str.strip()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = (
        df.dropna(subset=["date", RETURN_COL])
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )

    vol_col = find_volatility_column(df.columns)
    if vol_col is None:
        vol_col = "abs_daily_return"
        df[vol_col] = pd.to_numeric(df[RETURN_COL], errors="coerce").abs()

    features = [RETURN_COL, vol_col]
    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    valid = X_df.notna().all(axis=1)
    df = df.loc[valid].reset_index(drop=True)
    X = X_df.loc[valid].to_numpy(dtype=float)

    saved["date"] = pd.to_datetime(saved["date"], errors="coerce")
    split_idx = int(len(X) * TRAIN_RATIO)

    print(f"Rows: {len(X):,}")
    print(f"Training rows: {split_idx:,}")
    print(f"Testing rows: {len(X) - split_idx:,}")
    print(f"Features: {features}")

    # Fit preprocessing and HMM on training period only.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[:split_idx])
    X_scaled = scaler.transform(X)

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="full",
        n_iter=200,
        random_state=RANDOM_STATE,
        min_covar=1e-5
    )
    model.fit(X_train)

    # Recompute causal filtered probabilities.
    probabilities = forward_filter(model, X_scaled)

    # Validate the saved prediction structure.
    print("\n[1] OUTPUT FILE CHECK")
    required_cols = {
        "date", "regime", "regime_label", "regime_probability",
        "prob_low_volatility", "prob_medium_volatility",
        "prob_high_volatility", "data_split"
    }
    missing_cols = required_cols - set(saved.columns)
    output_ok = not missing_cols and len(saved) == len(df)

    print(f"Saved rows: {len(saved):,}")
    print(f"Output structure valid: {output_ok}")

    if missing_cols:
        print(f"Missing columns: {sorted(missing_cols)}")

    if len(saved) == len(df):
        dates_match = saved["date"].equals(df["date"])
    else:
        dates_match = False
    print(f"Dates aligned: {dates_match}")

    # Check probability values and row-wise sums.
    print("\n[2] PROBABILITY CHECK")
    prob_cols = [
        "prob_low_volatility",
        "prob_medium_volatility",
        "prob_high_volatility"
    ]

    probability_ok = False
    if set(prob_cols).issubset(saved.columns):
        saved_probs = saved[prob_cols].to_numpy(dtype=float)
        sums = saved_probs.sum(axis=1)
        probability_ok = (
            np.isfinite(saved_probs).all()
            and (saved_probs >= -1e-8).all()
            and (saved_probs <= 1 + 1e-8).all()
            and np.allclose(sums, 1.0, atol=1e-5)
        )
        print(f"Probabilities finite and within [0,1]: "
              f"{np.isfinite(saved_probs).all() and (saved_probs >= -1e-8).all() and (saved_probs <= 1 + 1e-8).all()}")
        print(f"Probability sums approximately 1: "
              f"{np.allclose(sums, 1.0, atol=1e-5)}")
    else:
        print("Probability columns missing.")

    # Causality test: perturb future observations and ensure
    # probabilities up to each checkpoint do not change.
    print("\n[3] FUTURE-INDEPENDENCE TEST")
    rng = np.random.default_rng(RANDOM_STATE)
    checkpoints = np.linspace(
        split_idx, len(X) - 1, num=min(10, len(X) - split_idx),
        dtype=int
    )
    max_diffs = []

    for checkpoint in checkpoints:
        altered = X_scaled.copy()
        if checkpoint + 1 < len(altered):
            altered[checkpoint + 1:] = (
                altered[checkpoint + 1:]
                + rng.normal(0, 5, size=altered[checkpoint + 1:].shape)
            )

        changed_probs = forward_filter(model, altered)
        difference = np.max(
            np.abs(
                probabilities[:checkpoint + 1]
                - changed_probs[:checkpoint + 1]
            )
        )
        max_diffs.append(difference)

    max_diff = max(max_diffs, default=0.0)
    causal_ok = max_diff < 1e-10
    print(f"Checkpoints tested: {len(checkpoints)}")
    print(f"Maximum past-probability difference: {max_diff:.12g}")
    print(f"Future-independence test passed: {causal_ok}")

    # Verify output split labels.
    print("\n[4] TRAIN/TEST LABEL CHECK")
    split_ok = False
    if "data_split" in saved.columns and len(saved) == len(df):
        expected_split = np.array([
            "train" if i < split_idx else "test"
            for i in range(len(df))
        ])
        split_ok = np.array_equal(
            saved["data_split"].astype(str).to_numpy(),
            expected_split
        )
    print(f"Split labels correct: {split_ok}")

    # Write validation report.
    checks = {
        "Output file structure": output_ok,
        "Date alignment": dates_match,
        "Probability validity": probability_ok,
        "Future-independence": causal_ok,
        "Train/test labels": split_ok
    }
    overall = all(checks.values())

    lines = [
        "CAUSAL HMM VALIDATION REPORT",
        "=" * 50,
        f"Rows: {len(X)}",
        f"Training rows: {split_idx}",
        f"Testing rows: {len(X) - split_idx}",
        f"Features: {features}",
        "",
        "CHECK RESULTS:"
    ]
    lines.extend(
        f"{name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in checks.items()
    )
    lines += [
        "",
        f"Maximum past-probability difference: {max_diff:.12g}",
        "",
        "LIMITATION:",
        "This validates the forward-filtering causality property and",
        "saved output structure. It does not prove every downstream",
        "model or strategy is free from leakage."
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print("VALIDATION RESULT:", "PASSED" if overall else "REVIEW REQUIRED")
    print(f"Report: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
