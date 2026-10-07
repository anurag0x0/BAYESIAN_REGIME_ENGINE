# ============================================================
# Bayesian Regime Detection Engine
# Project 1A - Bayesian HMM
# ============================================================
#
# File:
# C:\BAYESIAN_REGIME_ENGINE\src\models\bayesian_hmm.py
#
# Objective:
#   Detect latent market regimes using a Bayesian Hidden Markov Model
#   with posterior uncertainty and MCMC diagnostics.
#
# Regimes:
#   0 -> Risk-Off
#   1 -> Transitional
#   2 -> Risk-On
#   3 -> Late-Cycle
#   4 -> Post-Shock
#
# Input:
#   data/processed/nifty50_features.csv
#
# Main outputs:
#   data/processed/bayesian_hmm_regimes.csv
#   reports/bayesian_hmm/
#       posterior_summary.csv
#       diagnostics.txt
#       model_metadata.txt
#       trace.nc
#
# ============================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import pymc as pm
import arviz as az
import pytensor.tensor as pt
from pytensor.scan import scan

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data" / "processed" / "nifty50_features.csv"

OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "bayesian_hmm"

OUTPUT_FILE = OUTPUT_DIR / "bayesian_hmm_regimes.csv"
SUMMARY_FILE = REPORT_DIR / "posterior_summary.csv"
DIAGNOSTIC_FILE = REPORT_DIR / "diagnostics.txt"
METADATA_FILE = REPORT_DIR / "model_metadata.txt"
TRACE_FILE = REPORT_DIR / "trace.nc"


# ============================================================
# 2. CONFIGURATION
# ============================================================

N_STATES = 5

TRAIN_RATIO = 0.80

# MCMC configuration
DRAWS = 300
TUNE = 300
CHAINS = 2
CORES = 1

RANDOM_SEED = 42

# To keep MCMC computationally manageable,
# use the most recent training observations.
#
# None = use complete training data.
MAX_TRAIN_OBS = 1200


# ============================================================
# 3. REGIME DEFINITIONS
# ============================================================

REGIME_NAMES = {
    0: "Risk-Off",
    1: "Transitional",
    2: "Risk-On",
    3: "Late-Cycle",
    4: "Post-Shock",
}


# ============================================================
# 4. CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5. LOAD DATA
# ============================================================

def load_data():
    print("=" * 70)
    print("LOADING NIFTY 50 FEATURE DATA")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print(f"Input file: {INPUT_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # --------------------------------------------------------
    # Find date column
    # --------------------------------------------------------

    possible_date_columns = [
        "date",
        "Date",
        "DATE",
    ]

    date_column = None

    for col in possible_date_columns:
        if col in df.columns:
            date_column = col
            break

    if date_column is None:
        raise ValueError(
            "Could not find date column in feature dataset."
        )

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df = df.dropna(subset=[date_column])

    df = df.sort_values(date_column).reset_index(drop=True)

    # --------------------------------------------------------
    # Find daily return
    # --------------------------------------------------------

    return_candidates = [
        "daily_return",
        "log_return",
        "return",
    ]

    return_column = None

    for col in return_candidates:
        if col in df.columns:
            return_column = col
            break

    if return_column is None:
        raise ValueError(
            "Could not find a return column. "
            "Expected one of: daily_return, log_return, return."
        )

    # --------------------------------------------------------
    # Find volatility
    # --------------------------------------------------------

    volatility_candidates = [
        "volatility_5d",
        "rolling_volatility_5d",
        "vol_5d",
        "volatility_20d",
        "rolling_volatility_20d",
    ]

    volatility_column = None

    for col in volatility_candidates:
        if col in df.columns:
            volatility_column = col
            break

    if volatility_column is None:
        raise ValueError(
            "Could not find a volatility feature."
        )

    print(f"Date column: {date_column}")
    print(f"Return column: {return_column}")
    print(f"Volatility column: {volatility_column}")

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    work = df[
        [
            date_column,
            return_column,
            volatility_column,
        ]
    ].copy()

    work.columns = [
        "date",
        "daily_return",
        "volatility",
    ]

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    work["daily_return"] = pd.to_numeric(
        work["daily_return"],
        errors="coerce"
    )

    work["volatility"] = pd.to_numeric(
        work["volatility"],
        errors="coerce"
    )

    work = work.dropna().reset_index(drop=True)

    # --------------------------------------------------------
    # Remove infinite values
    # --------------------------------------------------------

    work = work[
        np.isfinite(work["daily_return"])
        &
        np.isfinite(work["volatility"])
    ].reset_index(drop=True)

    print(f"Usable observations: {len(work)}")

    if len(work) < 500:
        raise ValueError(
            "Not enough observations for Bayesian HMM."
        )

    return work


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

def create_split(df):
    print("\n" + "=" * 70)
    print("CREATING CHRONOLOGICAL TRAIN / TEST SPLIT")
    print("=" * 70)

    split_index = int(len(df) * TRAIN_RATIO)

    train = df.iloc[:split_index].copy()

    test = df.iloc[split_index:].copy()

    print(f"Total observations : {len(df)}")
    print(f"Training           : {len(train)}")
    print(f"Testing            : {len(test)}")

    print(
        f"Train period: "
        f"{train['date'].min().date()} -> "
        f"{train['date'].max().date()}"
    )

    print(
        f"Test period : "
        f"{test['date'].min().date()} -> "
        f"{test['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Optional training-size limit
    # --------------------------------------------------------

    if MAX_TRAIN_OBS is not None and len(train) > MAX_TRAIN_OBS:

        print(
            f"\nMCMC training window limited to "
            f"latest {MAX_TRAIN_OBS} observations."
        )

        train_for_mcmc = train.tail(MAX_TRAIN_OBS).copy()

    else:

        train_for_mcmc = train.copy()

    print(
        f"MCMC training observations: "
        f"{len(train_for_mcmc)}"
    )

    return train, test, train_for_mcmc


# ============================================================
# 7. STANDARDIZATION
# ============================================================

def standardize_features(train, full_df):
    print("\n" + "=" * 70)
    print("STANDARDIZING HMM OBSERVATIONS")
    print("=" * 70)

    train_return_mean = train["daily_return"].mean()
    train_return_std = train["daily_return"].std()

    train_vol_mean = train["volatility"].mean()
    train_vol_std = train["volatility"].std()

    if train_return_std == 0:
        raise ValueError(
            "Training return standard deviation is zero."
        )

    if train_vol_std == 0:
        raise ValueError(
            "Training volatility standard deviation is zero."
        )

    result = full_df.copy()

    result["return_scaled"] = (
        result["daily_return"] - train_return_mean
    ) / train_return_std

    result["volatility_scaled"] = (
        result["volatility"] - train_vol_mean
    ) / train_vol_std

    scaler = {
        "return_mean": float(train_return_mean),
        "return_std": float(train_return_std),
        "volatility_mean": float(train_vol_mean),
        "volatility_std": float(train_vol_std),
    }

    print(
        f"Return mean/std: "
        f"{train_return_mean:.8f} / "
        f"{train_return_std:.8f}"
    )

    print(
        f"Volatility mean/std: "
        f"{train_vol_mean:.8f} / "
        f"{train_vol_std:.8f}"
    )

    return result, scaler


# ============================================================
# 8. PREPARE OBSERVATION
# ============================================================

def create_observation_matrix(train_for_mcmc, scaler):
    """
    Create 2-dimensional observations:

        Feature 0 = standardized daily return
        Feature 1 = standardized volatility
    """

    x_return = (
        train_for_mcmc["daily_return"].values
        - scaler["return_mean"]
    ) / scaler["return_std"]

    x_volatility = (
        train_for_mcmc["volatility"].values
        - scaler["volatility_mean"]
    ) / scaler["volatility_std"]

    observations = np.column_stack(
        [
            x_return,
            x_volatility,
        ]
    )

    observations = np.asarray(
        observations,
        dtype=np.float64
    )

    return observations


# ============================================================
# 9. BUILD BAYESIAN HMM
# ============================================================

def build_bayesian_hmm(observations):
    """
    Bayesian HMM.

    Hidden state:
        z_t ∈ {0,1,2,3,4}

    Observation:
        x_t = [return_t, volatility_t]

    Each state has:
        mu[state, feature]
        sigma[state, feature]

    Transition matrix:
        P(z_t | z_(t-1))

    We marginalize the discrete hidden states using
    the forward algorithm instead of sampling the state
    sequence directly.

    This avoids problematic discrete-variable sampling
    inside NUTS.
    """

    n_observations = len(observations)

    print("\n" + "=" * 70)
    print("BUILDING BAYESIAN HMM")
    print("=" * 70)

    print(f"Observations: {n_observations}")
    print(f"States      : {N_STATES}")
    print("Features    : daily_return + volatility")

    # --------------------------------------------------------
    # Convert observation data to PyTensor
    # --------------------------------------------------------

    obs_tensor = pt.as_tensor_variable(observations)

    with pm.Model() as model:

        # ====================================================
        # INITIAL STATE PROBABILITY
        # ====================================================

        initial_probs = pm.Dirichlet(
            "initial_probs",
            a=np.ones(N_STATES)
        )

        # ====================================================
        # TRANSITION MATRIX
        # ====================================================

        transition_matrix = pm.Dirichlet(
            "transition_matrix",
            a=np.ones(N_STATES),
            shape=(N_STATES, N_STATES),
        )

        # ====================================================
        # STATE MEANS
        # ====================================================

        mu = pm.Normal(
            "mu",
            mu=0.0,
            sigma=1.5,
            shape=(N_STATES, 2),
        )

        # ====================================================
        # STATE STANDARD DEVIATIONS
        # ====================================================

        sigma = pm.HalfNormal(
            "sigma",
            sigma=1.0,
            shape=(N_STATES, 2),
        )

        # ----------------------------------------------------
        # Log emission probability
        #
        # Independent Gaussian emission for:
        #   return
        #   volatility
        # ----------------------------------------------------

        obs_expanded = obs_tensor[:, None, :]

        mu_expanded = mu[None, :, :]

        sigma_expanded = sigma[None, :, :]

        log_prob_each_feature = (
            -0.5
            * (
                (
                    obs_expanded
                    - mu_expanded
                )
                / sigma_expanded
            ) ** 2
            - pt.log(sigma_expanded)
            - 0.5 * np.log(2.0 * np.pi)
        )

        log_emission = pt.sum(
            log_prob_each_feature,
            axis=2
        )

        # ----------------------------------------------------
        # Forward algorithm
        # ----------------------------------------------------

        log_initial_probs = pt.log(
            initial_probs + 1e-12
        )

        log_transition_matrix = pt.log(
            transition_matrix + 1e-12
        )

        # First observation
        alpha_0 = (
            log_initial_probs
            + log_emission[0]
        )

        def forward_step(
            current_emission,
            previous_alpha,
            log_transition
        ):
            """
            Forward recursion:

            alpha_t(j)
              =
            emission_t(j)
              +
            logsumexp_i(
                alpha_(t-1)(i)
                +
                transition(i,j)
            )
            """

            transition_scores = (
                previous_alpha[:, None]
                + log_transition
            )

            next_alpha = (
                current_emission
                + pt.logsumexp(
                    transition_scores,
                    axis=0
                )
            )

            return next_alpha

        # ----------------------------------------------------
        # Run forward recursion
        # ----------------------------------------------------

        alpha_sequence, _ = scan(
            fn=forward_step,
            sequences=log_emission[1:],
            outputs_info=alpha_0,
            non_sequences=log_transition_matrix,
        )

        final_alpha = alpha_sequence[-1]

        log_likelihood = pt.logsumexp(
            final_alpha
        )

        # ----------------------------------------------------
        # Add model likelihood
        # ----------------------------------------------------

        pm.Potential(
            "hmm_log_likelihood",
            log_likelihood
        )

    return model


# ============================================================
# 10. RUN MCMC
# ============================================================

def sample_posterior(model):
    print("\n" + "=" * 70)
    print("RUNNING BAYESIAN POSTERIOR SAMPLING")
    print("=" * 70)

    print(
        f"Draws : {DRAWS}\n"
        f"Tune  : {TUNE}\n"
        f"Chains: {CHAINS}\n"
        f"Cores : {CORES}"
    )

    print(
        "\nIMPORTANT: Bayesian HMM sampling can take "
        "several minutes depending on CPU."
    )

    with model:

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            random_seed=RANDOM_SEED,
            target_accept=0.90,
            init="adapt_diag",
            return_inferencedata=True,
            progressbar=True,
        )

    return trace


# ============================================================
# 11. POSTERIOR SUMMARY
# ============================================================

def save_posterior_summary(trace):
    print("\n" + "=" * 70)
    print("CREATING POSTERIOR SUMMARY")
    print("=" * 70)

    summary = az.summary(
        trace,
        var_names=[
            "initial_probs",
            "transition_matrix",
            "mu",
            "sigma",
        ],
        round_to=6,
    )

    summary.to_csv(
        SUMMARY_FILE
    )

    print(
        f"Posterior summary saved:\n"
        f"{SUMMARY_FILE}"
    )

    print("\nPosterior summary preview:")

    print(
        summary[
            [
                "mean",
                "sd",
                "hdi_3%",
                "hdi_97%",
                "ess_bulk",
                "r_hat",
            ]
        ].head(20)
    )

    return summary


# ============================================================
# 12. DIAGNOSTICS
# ============================================================

def create_diagnostics(trace):
    print("\n" + "=" * 70)
    print("BAYESIAN MODEL DIAGNOSTICS")
    print("=" * 70)

    summary = az.summary(
        trace,
        var_names=[
            "initial_probs",
            "transition_matrix",
            "mu",
            "sigma",
        ],
        round_to=6,
    )

    # --------------------------------------------------------
    # R-hat
    # --------------------------------------------------------

    rhat_values = summary["r_hat"].dropna()

    max_rhat = (
        float(rhat_values.max())
        if len(rhat_values) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # ESS
    # --------------------------------------------------------

    ess_values = summary["ess_bulk"].dropna()

    min_ess = (
        float(ess_values.min())
        if len(ess_values) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Divergences
    # --------------------------------------------------------

    divergences = 0

    if "diverging" in trace.sample_stats:

        divergences = int(
            trace.sample_stats["diverging"]
            .values
            .sum()
        )

    # --------------------------------------------------------
    # Number of posterior draws
    # --------------------------------------------------------

    posterior_size = (
        trace.posterior.sizes.get(
            "draw",
            0
        )
    )

    chains = (
        trace.posterior.sizes.get(
            "chain",
            0
        )
    )

    # --------------------------------------------------------
    # Diagnostic status
    # --------------------------------------------------------

    rhat_ok = (
        np.isfinite(max_rhat)
        and max_rhat < 1.01
    )

    ess_ok = (
        np.isfinite(min_ess)
        and min_ess >= 100
    )

    divergence_ok = (
        divergences == 0
    )

    overall_ok = (
        rhat_ok
        and ess_ok
        and divergence_ok
    )

    status = (
        "PASS"
        if overall_ok
        else "REVIEW REQUIRED"
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "BAYESIAN HMM DIAGNOSTIC REPORT"
    )

    report.append(
        "=" * 60
    )

    report.append(
        f"Chains: {chains}"
    )

    report.append(
        f"Posterior draws per chain: {posterior_size}"
    )

    report.append(
        f"Maximum R-hat: {max_rhat:.6f}"
    )

    report.append(
        f"Minimum bulk ESS: {min_ess:.2f}"
    )

    report.append(
        f"Total divergences: {divergences}"
    )

    report.append("")
    report.append(
        "Diagnostic thresholds:"
    )

    report.append(
        "R-hat < 1.01"
    )

    report.append(
        "Bulk ESS >= 100"
    )

    report.append(
        "Divergences = 0"
    )

    report.append("")
    report.append(
        f"R-hat check: "
        f"{'PASS' if rhat_ok else 'FAIL'}"
    )

    report.append(
        f"ESS check: "
        f"{'PASS' if ess_ok else 'FAIL'}"
    )

    report.append(
        f"Divergence check: "
        f"{'PASS' if divergence_ok else 'FAIL'}"
    )

    report.append("")
    report.append(
        f"OVERALL STATUS: {status}"
    )

    DIAGNOSTIC_FILE.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    print("\n".join(report))

    return {
        "max_rhat": max_rhat,
        "min_ess": min_ess,
        "divergences": divergences,
        "status": status,
    }


# ============================================================
# 13. EXTRACT POSTERIOR PARAMETERS
# ============================================================

def extract_posterior_parameters(trace):
    print("\n" + "=" * 70)
    print("EXTRACTING POSTERIOR PARAMETERS")
    print("=" * 70)

    # --------------------------------------------------------
    # Posterior mean of mu
    # --------------------------------------------------------

    posterior_mu = (
        trace.posterior["mu"]
        .mean(
            dim=[
                "chain",
                "draw",
            ]
        )
        .values
    )

    # Shape:
    # (N_STATES, 2)

    posterior_sigma = (
        trace.posterior["sigma"]
        .mean(
            dim=[
                "chain",
                "draw",
            ]
        )
        .values
    )

    posterior_transition = (
        trace.posterior["transition_matrix"]
        .mean(
            dim=[
                "chain",
                "draw",
            ]
        )
        .values
    )

    posterior_initial = (
        trace.posterior["initial_probs"]
        .mean(
            dim=[
                "chain",
                "draw",
            ]
        )
        .values
    )

    print("\nPosterior state means:")

    for state in range(N_STATES):

        print(
            f"State {state}: "
            f"return_mu={posterior_mu[state, 0]:.4f}, "
            f"vol_mu={posterior_mu[state, 1]:.4f}"
        )

    return (
        posterior_mu,
        posterior_sigma,
        posterior_transition,
        posterior_initial,
    )


# ============================================================
# 14. FORWARD FILTERING
# ============================================================

def calculate_filtered_probabilities(
    observations,
    posterior_mu,
    posterior_sigma,
    posterior_transition,
    posterior_initial,
):
    """
    Calculate filtered P(z_t | x_1:t).

    This uses posterior-mean parameters.

    Important:
    This is a filtered probability estimate, not a
    posterior state sample from every MCMC draw.
    """

    n = len(observations)

    log_emission = np.zeros(
        (n, N_STATES),
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Calculate emission log likelihood
    # --------------------------------------------------------

    for state in range(N_STATES):

        mu_state = posterior_mu[state]

        sigma_state = (
            posterior_sigma[state]
            + 1e-8
        )

        standardized = (
            observations
            - mu_state
        ) / sigma_state

        log_pdf = (
            -0.5 * standardized ** 2
            - np.log(sigma_state)
            - 0.5 * np.log(2 * np.pi)
        )

        log_emission[:, state] = (
            log_pdf.sum(axis=1)
        )

    # --------------------------------------------------------
    # Stable log-space forward filtering
    # --------------------------------------------------------

    log_alpha = np.zeros(
        (n, N_STATES),
        dtype=np.float64
    )

    log_initial = np.log(
        posterior_initial + 1e-12
    )

    log_transition = np.log(
        posterior_transition + 1e-12
    )

    # First observation
    log_alpha[0] = (
        log_initial
        + log_emission[0]
    )

    # Normalize
    max_value = np.max(
        log_alpha[0]
    )

    probabilities = (
        np.exp(
            log_alpha[0]
            - max_value
        )
    )

    probabilities /= probabilities.sum()

    filtered = np.zeros(
        (n, N_STATES),
        dtype=np.float64
    )

    filtered[0] = probabilities

    # --------------------------------------------------------
    # Sequential filtering
    # --------------------------------------------------------

    for t in range(1, n):

        previous = log_alpha[t - 1]

        transition_scores = (
            previous[:, None]
            + log_transition
        )

        next_alpha = (
            log_emission[t]
            + logsumexp_numpy(
                transition_scores,
                axis=0
            )
        )

        log_alpha[t] = next_alpha

        max_value = np.max(
            next_alpha
        )

        probabilities = np.exp(
            next_alpha
            - max_value
        )

        total = probabilities.sum()

        if total <= 0 or not np.isfinite(total):

            probabilities = (
                np.ones(N_STATES)
                / N_STATES
            )

        else:

            probabilities /= total

        filtered[t] = probabilities

    return filtered


# ============================================================
# 15. NUMPY LOGSUMEXP
# ============================================================

def logsumexp_numpy(values, axis=None):
    """
    Numerically stable logsumexp.
    """

    max_value = np.max(
        values,
        axis=axis,
        keepdims=True
    )

    stabilized = (
        values
        - max_value
    )

    result = (
        max_value
        + np.log(
            np.sum(
                np.exp(stabilized),
                axis=axis,
                keepdims=True
            )
        )
    )

    if axis is not None:

        result = np.squeeze(
            result,
            axis=axis
        )

    return result


# ============================================================
# 16. MAP REGIME LABELS
# ============================================================

def assign_regime_names(
    probabilities,
    posterior_mu,
    posterior_sigma,
):
    """
    Convert latent states into interpretable regime labels.

    Because HMM state numbers themselves have no intrinsic
    economic meaning, states are mapped using posterior
    return/volatility characteristics.

    Mapping logic:

      High positive return + lower volatility -> Risk-On
      Negative return + high volatility       -> Risk-Off
      Very high volatility                    -> Post-Shock
      Positive return + high volatility       -> Late-Cycle
      Remaining state                         -> Transitional
    """

    state_statistics = []

    for state in range(N_STATES):

        return_mu = posterior_mu[state, 0]

        volatility_mu = posterior_mu[state, 1]

        return_sigma = posterior_sigma[state, 0]

        volatility_sigma = posterior_sigma[state, 1]

        state_statistics.append(
            {
                "state": state,
                "return_mu": return_mu,
                "volatility_mu": volatility_mu,
                "return_sigma": return_sigma,
                "volatility_sigma": volatility_sigma,
            }
        )

    stats = pd.DataFrame(
        state_statistics
    )

    # --------------------------------------------------------
    # Rank states
    # --------------------------------------------------------

    high_vol_state = int(
        stats["volatility_mu"]
        .idxmax()
    )

    low_vol_state = int(
        stats["volatility_mu"]
        .idxmin()
    )

    high_return_state = int(
        stats["return_mu"]
        .idxmax()
    )

    low_return_state = int(
        stats["return_mu"]
        .idxmin()
    )

    mapping = {}

    # --------------------------------------------------------
    # Post-Shock
    # --------------------------------------------------------

    mapping[high_vol_state] = "Post-Shock"

    # --------------------------------------------------------
    # Risk-Off
    # --------------------------------------------------------

    remaining = [
        s for s in range(N_STATES)
        if s not in mapping
    ]

    if remaining:

        risk_off_state = min(
            remaining,
            key=lambda s: stats.loc[
                s,
                "return_mu"
            ]
        )

        mapping[risk_off_state] = "Risk-Off"

    # --------------------------------------------------------
    # Risk-On
    # --------------------------------------------------------

    remaining = [
        s for s in range(N_STATES)
        if s not in mapping
    ]

    if remaining:

        risk_on_state = max(
            remaining,
            key=lambda s: stats.loc[
                s,
                "return_mu"
            ]
        )

        mapping[risk_on_state] = "Risk-On"

    # --------------------------------------------------------
    # Late-Cycle
    # --------------------------------------------------------

    remaining = [
        s for s in range(N_STATES)
        if s not in mapping
    ]

    if remaining:

        late_cycle_state = max(
            remaining,
            key=lambda s: stats.loc[
                s,
                "volatility_mu"
            ]
        )

        mapping[late_cycle_state] = "Late-Cycle"

    # --------------------------------------------------------
    # Transitional
    # --------------------------------------------------------

    remaining = [
        s for s in range(N_STATES)
        if s not in mapping
    ]

    for state in remaining:

        mapping[state] = "Transitional"

    # --------------------------------------------------------
    # Ensure every state has unique name
    # --------------------------------------------------------

    used_names = set()

    final_mapping = {}

    for state in range(N_STATES):

        name = mapping.get(
            state,
            "Transitional"
        )

        if name in used_names:

            name = (
                f"Transitional-{state}"
            )

        used_names.add(name)

        final_mapping[state] = name

    print("\nState interpretation:")

    for state in range(N_STATES):

        print(
            f"State {state} -> "
            f"{final_mapping[state]}"
        )

    return final_mapping


# ============================================================
# 17. CREATE FINAL REGIME DATASET
# ============================================================

def create_regime_dataset(
    df,
    filtered_probabilities,
    state_mapping,
):
    print("\n" + "=" * 70)
    print("CREATING BAYESIAN HMM REGIME DATASET")
    print("=" * 70)

    result = df.copy()

    # --------------------------------------------------------
    # Add state probabilities
    # --------------------------------------------------------

    for state in range(N_STATES):

        regime_name = state_mapping[
            state
        ]

        safe_name = (
            regime_name
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        result[
            f"prob_{safe_name}"
        ] = filtered_probabilities[
            :, state
        ]

    # --------------------------------------------------------
    # Most probable state
    # --------------------------------------------------------

    result[
        "bayesian_hmm_state"
    ] = np.argmax(
        filtered_probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Regime label
    # --------------------------------------------------------

    result[
        "bayesian_hmm_regime"
    ] = result[
        "bayesian_hmm_state"
    ].map(
        state_mapping
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    result[
        "bayesian_hmm_confidence"
    ] = np.max(
        filtered_probabilities,
        axis=1
    )

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    probability_safe = np.clip(
        filtered_probabilities,
        1e-12,
        1.0
    )

    entropy = -np.sum(
        probability_safe
        * np.log(probability_safe),
        axis=1
    )

    result[
        "bayesian_hmm_entropy"
    ] = entropy

    # --------------------------------------------------------
    # Regime change
    # --------------------------------------------------------

    result[
        "regime_change"
    ] = (
        result[
            "bayesian_hmm_state"
        ]
        .ne(
            result[
                "bayesian_hmm_state"
            ].shift(1)
        )
        .astype(int)
    )

    # First row is not a change
    result.loc[
        result.index[0],
        "regime_change"
    ] = 0

    return result


# ============================================================
# 18. SAVE TRACE
# ============================================================

def save_trace(trace):
    print("\n" + "=" * 70)
    print("SAVING BAYESIAN TRACE")
    print("=" * 70)

    trace.to_netcdf(
        TRACE_FILE
    )

    print(
        f"Trace saved:\n"
        f"{TRACE_FILE}"
    )


# ============================================================
# 19. SAVE METADATA
# ============================================================

def save_metadata(
    df,
    train,
    test,
    train_for_mcmc,
    scaler,
    state_mapping,
    diagnostics,
):
    print("\n" + "=" * 70)
    print("SAVING MODEL METADATA")
    print("=" * 70)

    metadata = {
        "project": (
            "Bayesian Regime Detection Engine "
            "for Equity Direction Forecasting"
        ),
        "model": "Bayesian Hidden Markov Model",
        "asset": "NIFTY 50",
        "states": N_STATES,
        "state_labels": state_mapping,
        "features": [
            "daily_return",
            "volatility",
        ],
        "train_ratio": TRAIN_RATIO,
        "total_observations": len(df),
        "train_observations": len(train),
        "test_observations": len(test),
        "mcmc_training_observations": len(
            train_for_mcmc
        ),
        "draws": DRAWS,
        "tune": TUNE,
        "chains": CHAINS,
        "cores": CORES,
        "random_seed": RANDOM_SEED,
        "target_accept": 0.90,
        "scaler": scaler,
        "diagnostics": diagnostics,
        "date_range": {
            "start": str(
                df["date"].min().date()
            ),
            "end": str(
                df["date"].max().date()
            ),
        },
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=4,
            default=str
        ),
        encoding="utf-8"
    )

    print(
        f"Metadata saved:\n"
        f"{METADATA_FILE}"
    )


# ============================================================
# 20. PRINT REGIME SUMMARY
# ============================================================

def print_regime_summary(result):
    print("\n" + "=" * 70)
    print("BAYESIAN HMM REGIME SUMMARY")
    print("=" * 70)

    counts = (
        result[
            "bayesian_hmm_regime"
        ]
        .value_counts()
    )

    percentages = (
        result[
            "bayesian_hmm_regime"
        ]
        .value_counts(
            normalize=True
        )
        * 100
    )

    summary = pd.DataFrame(
        {
            "observations": counts,
            "share_pct": percentages,
        }
    )

    summary["share_pct"] = (
        summary["share_pct"]
        .round(2)
    )

    print(summary)

    print("\nLatest regime:")

    latest = result.iloc[-1]

    print(
        f"Date       : "
        f"{latest['date'].date()}"
    )

    print(
        f"Regime     : "
        f"{latest['bayesian_hmm_regime']}"
    )

    print(
        f"Confidence : "
        f"{latest['bayesian_hmm_confidence']:.4f}"
    )

    print(
        f"Entropy    : "
        f"{latest['bayesian_hmm_entropy']:.4f}"
    )


# ============================================================
# 21. MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print(
        "PROJECT 1A - BAYESIAN REGIME DETECTION ENGINE"
    )
    print(
        "STEP 21 - BAYESIAN HMM"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    train, test, train_for_mcmc = (
        create_split(df)
    )

    # --------------------------------------------------------
    # Standardization
    #
    # Fit scaler using training data only.
    # --------------------------------------------------------

    standardized_df, scaler = (
        standardize_features(
            train,
            df
        )
    )

    standardized_train_for_mcmc = (
        standardized_df[
            standardized_df["date"].isin(
                train_for_mcmc["date"]
            )
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Observation matrix
    # --------------------------------------------------------

    observations = (
        create_observation_matrix(
            standardized_train_for_mcmc,
            scaler
        )
    )

    print(
        f"\nObservation matrix shape: "
        f"{observations.shape}"
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_bayesian_hmm(
        observations
    )

    # --------------------------------------------------------
    # Sample posterior
    # --------------------------------------------------------

    trace = sample_posterior(
        model
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_posterior_summary(
        trace
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    diagnostics = create_diagnostics(
        trace
    )

    # --------------------------------------------------------
    # Posterior parameters
    # --------------------------------------------------------

    (
        posterior_mu,
        posterior_sigma,
        posterior_transition,
        posterior_initial,
    ) = extract_posterior_parameters(
        trace
    )

    # --------------------------------------------------------
    # Create observations for ALL data
    # --------------------------------------------------------

    all_observations = (
        create_observation_matrix(
            standardized_df,
            scaler
        )
    )

    # --------------------------------------------------------
    # Filtered probabilities
    # --------------------------------------------------------

    filtered_probabilities = (
        calculate_filtered_probabilities(
            all_observations,
            posterior_mu,
            posterior_sigma,
            posterior_transition,
            posterior_initial,
        )
    )

    # --------------------------------------------------------
    # State interpretation
    # --------------------------------------------------------

    state_mapping = (
        assign_regime_names(
            filtered_probabilities,
            posterior_mu,
            posterior_sigma,
        )
    )

    # --------------------------------------------------------
    # Final dataset
    # --------------------------------------------------------

    result = create_regime_dataset(
        standardized_df,
        filtered_probabilities,
        state_mapping,
    )

    # --------------------------------------------------------
    # Save final CSV
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nBayesian HMM regime data saved:\n"
        f"{OUTPUT_FILE}"
    )

    # --------------------------------------------------------
    # Save trace
    # --------------------------------------------------------

    save_trace(
        trace
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        df,
        train,
        test,
        train_for_mcmc,
        scaler,
        state_mapping,
        diagnostics,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_regime_summary(
        result
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    probability_columns = [
        col
        for col in result.columns
        if col.startswith("prob_")
    ]

    probability_sum = (
        result[
            probability_columns
        ]
        .sum(axis=1)
    )

    probability_valid = (
        np.allclose(
            probability_sum.values,
            1.0,
            atol=1e-5
        )
    )

    finite_probabilities = (
        np.isfinite(
            result[
                probability_columns
            ].values
        ).all()
    )

    dates_sorted = (
        result["date"]
        .is_monotonic_increasing
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    print(
        f"Probability sum == 1 : "
        f"{probability_valid}"
    )

    print(
        f"Probabilities finite : "
        f"{finite_probabilities}"
    )

    print(
        f"Dates sorted         : "
        f"{dates_sorted}"
    )

    print(
        f"Rows generated       : "
        f"{len(result)}"
    )

    print("\n" + "=" * 70)
    print("STEP 21 COMPLETED")
    print("=" * 70)

    print(
        "\nGenerated files:"
    )

    print(
        f"1. {OUTPUT_FILE}"
    )

    print(
        f"2. {SUMMARY_FILE}"
    )

    print(
        f"3. {DIAGNOSTIC_FILE}"
    )

    print(
        f"4. {METADATA_FILE}"
    )

    print(
        f"5. {TRACE_FILE}"
    )

    print(
        "\nBayesian HMM posterior inference completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()