
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import logsumexp, ndtr
from sklearn.metrics import accuracy_score, brier_score_loss


# ==========================================
# 1. PROJECT PATHS
# ==========================================
ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "nifty50_features.csv"

MODEL_DIR = ROOT / "models" / "smc"
REPORT_DIR = ROOT / "reports" / "smc"
OUTPUT_PATH = ROOT / "data" / "processed" / "smc_predictions.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. CONFIGURATION
# ==========================================
N_PARTICLES = 3000
TRAIN_RATIO = 0.80
RANDOM_SEED = 42
RESAMPLE_ESS_RATIO = 0.50


# ==========================================
# 3. LOAD DATA
# ==========================================
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"]
    )

    required_columns = [
        "date",
        "daily_return",
        "volatility_20d",
        "target_next_day_direction"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    df = df[required_columns].copy()

    df = df.sort_values("date")
    df = df.drop_duplicates("date")

    numeric_columns = [
        "daily_return",
        "volatility_20d",
        "target_next_day_direction"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=required_columns
    ).reset_index(drop=True)

    target_values = set(
        df["target_next_day_direction"].unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "Target must contain only 0 and 1."
        )

    if len(df) < 100:
        raise ValueError(
            "Not enough data to run Particle Filter."
        )

    return df


# ==========================================
# 4. EFFECTIVE SAMPLE SIZE
# ==========================================
def effective_sample_size(weights):

    return 1.0 / np.sum(weights ** 2)


# ==========================================
# 5. SYSTEMATIC RESAMPLING
# ==========================================
def systematic_resample(weights, rng):

    n = len(weights)

    positions = (
        rng.random() + np.arange(n)
    ) / n

    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0

    indexes = np.zeros(n, dtype=int)

    i = 0
    j = 0

    while i < n:

        if positions[i] < cumulative_sum[j]:

            indexes[i] = j
            i += 1

        else:
            j += 1

    return indexes


# ==========================================
# 6. PARTICLE FILTER
# ==========================================
def run_particle_filter(df, train_end):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # Training data only for initial parameters
    train_df = df.iloc[:train_end]

    train_returns = train_df[
        "daily_return"
    ].to_numpy(dtype=float)

    train_volatility = train_df[
        "volatility_20d"
    ].to_numpy(dtype=float)

    base_mean = float(
        np.mean(train_returns)
    )

    observation_sigma = float(
        np.std(train_returns, ddof=1)
    )

    median_volatility = float(
        np.median(train_volatility)
    )

    if (
        not np.isfinite(observation_sigma)
        or observation_sigma <= 0
    ):
        raise ValueError(
            "Could not estimate return volatility."
        )

    if (
        not np.isfinite(median_volatility)
        or median_volatility <= 0
    ):
        raise ValueError(
            "Could not estimate median volatility."
        )

    # Latent state random-walk noise
    process_sigma = max(
        observation_sigma * 0.05,
        1e-6
    )

    # Initialize particles
    particles = rng.normal(
        loc=base_mean,
        scale=max(
            observation_sigma * 0.25,
            1e-6
        ),
        size=N_PARTICLES
    )

    weights = np.full(
        N_PARTICLES,
        1.0 / N_PARTICLES
    )

    predictions = []
    resampling_count = 0

    returns = df[
        "daily_return"
    ].to_numpy(dtype=float)

    volatility = df[
        "volatility_20d"
    ].to_numpy(dtype=float)

    targets = df[
        "target_next_day_direction"
    ].to_numpy(dtype=int)

    # Process observations in chronological order
    for i in range(len(df)):

        # Use current day's known volatility
        vol_scale = np.clip(
            volatility[i] / median_volatility,
            0.5,
            2.0
        )

        sigma_t = max(
            observation_sigma * vol_scale,
            1e-8
        )

        # --------------------------
        # A. PREDICT
        # --------------------------
        particles = particles + rng.normal(
            loc=0.0,
            scale=process_sigma,
            size=N_PARTICLES
        )

        # --------------------------
        # B. UPDATE
        # --------------------------
        log_likelihood = (
            -0.5 * (
                (returns[i] - particles) / sigma_t
            ) ** 2
            - np.log(sigma_t)
            - 0.5 * np.log(2.0 * np.pi)
        )

        log_weights = np.log(
            np.maximum(weights, 1e-300)
        )

        log_weights += log_likelihood

        log_weights -= logsumexp(
            log_weights
        )

        weights = np.exp(log_weights)

        # --------------------------
        # C. ESTIMATE LATENT STATE
        # --------------------------
        state_mean = float(
            np.sum(particles * weights)
        )

        state_variance = float(
            np.sum(
                weights * (
                    particles - state_mean
                ) ** 2
            )
        )

        ess = effective_sample_size(
            weights
        )

        # --------------------------
        # D. NEXT-DAY PROBABILITY
        # --------------------------
        probability_up = float(
            np.sum(
                weights * ndtr(
                    particles / sigma_t
                )
            )
        )

        probability_up = float(
            np.clip(
                probability_up,
                0.0,
                1.0
            )
        )

        predicted_direction = int(
            probability_up >= 0.5
        )

        predictions.append({

            "date": df.loc[i, "date"],

            "daily_return": returns[i],

            "actual_next_day_direction": targets[i],

            "estimated_latent_return": state_mean,

            "latent_state_std": float(
                np.sqrt(
                    max(state_variance, 0.0)
                )
            ),

            "probability_up": probability_up,

            "predicted_direction": predicted_direction,

            "effective_sample_size": float(ess),

            "volatility_scale": float(vol_scale),

            "split": (
                "train"
                if i < train_end
                else "test"
            )
        })

        # --------------------------
        # E. RESAMPLING
        # --------------------------
        if ess < RESAMPLE_ESS_RATIO * N_PARTICLES:

            indexes = systematic_resample(
                weights,
                rng
            )

            particles = particles[indexes]

            weights.fill(
                1.0 / N_PARTICLES
            )

            resampling_count += 1

    predictions_df = pd.DataFrame(
        predictions
    )

    config = {
        "base_mean": base_mean,
        "observation_sigma": observation_sigma,
        "process_sigma": process_sigma,
        "median_volatility": median_volatility,
        "n_particles": N_PARTICLES,
        "resampling_count": resampling_count,
        "random_seed": RANDOM_SEED
    }

    return predictions_df, config


# ==========================================
# 7. MAIN
# ==========================================
def main():

    print("Loading feature dataset...")

    df = load_data()

    train_end = int(
        len(df) * TRAIN_RATIO
    )

    if train_end <= 0 or train_end >= len(df):
        raise ValueError(
            "Invalid train/test split."
        )

    print(f"Total records: {len(df)}")
    print(f"Training records: {train_end}")
    print(
        f"Testing records: {len(df) - train_end}"
    )
    print(f"Particles: {N_PARTICLES}")

    # Run Particle Filter
    predictions, config = run_particle_filter(
        df,
        train_end
    )

    # Test period only
    test = predictions[
        predictions["split"] == "test"
    ].copy()

    # Evaluation
    accuracy = accuracy_score(
        test["actual_next_day_direction"],
        test["predicted_direction"]
    )

    brier = brier_score_loss(
        test["actual_next_day_direction"],
        test["probability_up"]
    )

    # Save predictions
    predictions.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # Save configuration
    model_path = (
        MODEL_DIR / "particle_filter_config.pkl"
    )

    joblib.dump(
        config,
        model_path
    )

    # Save report
    report_path = (
        REPORT_DIR / "smc_summary.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "SEQUENTIAL MONTE CARLO - PARTICLE FILTER\n"
        )

        file.write(
            "========================================\n"
        )

        file.write(
            f"Total records: {len(df)}\n"
        )

        file.write(
            f"Training records: {train_end}\n"
        )

        file.write(
            f"Testing records: {len(test)}\n"
        )

        file.write(
            f"Particles: {N_PARTICLES}\n"
        )

        file.write(
            f"Process sigma: {config['process_sigma']:.8f}\n"
        )

        file.write(
            "Observation sigma: "
            f"{config['observation_sigma']:.8f}\n"
        )

        file.write(
            "Resampling events: "
            f"{config['resampling_count']}\n"
        )

        file.write(
            f"Test direction accuracy: {accuracy:.4f}\n"
        )

        file.write(
            f"Test Brier score: {brier:.4f}\n"
        )

        file.write(
            "\nNote: Experimental particle-filter baseline. "
            "Not a validated trading strategy. "
            "Further walk-forward evaluation and "
            "calibration are required.\n"
        )

    # Final output
    print(
        "\nParticle Filter completed successfully!"
    )

    print(
        f"Test direction accuracy: {accuracy:.4f}"
    )

    print(
        f"Test Brier score: {brier:.4f}"
    )

    print(
        "Resampling events: "
        f"{config['resampling_count']}"
    )

    print(
        f"Predictions saved: {OUTPUT_PATH}"
    )

    print(
        f"Model config saved: {model_path}"
    )

    print(
        f"Summary saved: {report_path}"
    )


if __name__ == "__main__":
    main()
