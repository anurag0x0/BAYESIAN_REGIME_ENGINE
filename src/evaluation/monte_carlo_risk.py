from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "monte_carlo"
MODEL_DIR = ROOT / "models" / "monte_carlo"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT FILES
# ============================================================

FEATURE_FILE = DATA_DIR / "nifty50_features.csv"

REGIME_FILE = (
    DATA_DIR / "causal_regime_integrated_predictions.csv"
)


# ============================================================
# OUTPUT FILES
# ============================================================

SIMULATION_FILE = (
    DATA_DIR / "monte_carlo_simulations.csv"
)

RISK_METRICS_FILE = (
    REPORT_DIR / "risk_metrics.csv"
)

REGIME_METRICS_FILE = (
    REPORT_DIR / "regime_risk_metrics.csv"
)

CRISIS_METRICS_FILE = (
    REPORT_DIR / "crisis_scenarios.csv"
)

SUMMARY_FILE = (
    REPORT_DIR / "summary.txt"
)

SIMULATION_PLOT = (
    REPORT_DIR / "simulation_paths.png"
)

RETURN_DISTRIBUTION_PLOT = (
    REPORT_DIR / "return_distribution.png"
)

REGIME_RISK_PLOT = (
    REPORT_DIR / "regime_risk_comparison.png"
)

CONFIG_FILE = (
    MODEL_DIR / "monte_carlo_config.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

N_SIMULATIONS = 10000

HORIZON_DAYS = 20

VAR_LEVELS = [0.90, 0.95, 0.99]

MIN_REGIME_OBSERVATIONS = 30


# ============================================================
# HELPER
# ============================================================

def find_column(df, candidates, required=True):
    """
    Return the first matching column.
    """

    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise ValueError(
            f"None of these columns were found: {candidates}"
        )

    return None


def clean_returns(series):
    """
    Convert return values to numeric and
    remove invalid values.
    """

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    )

    values = values.dropna()

    return values.astype(float)


def calculate_var(returns, confidence):
    """
    Empirical Value at Risk.

    Returned as a positive loss percentage.
    """

    values = np.asarray(
        returns,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    lower_quantile = np.quantile(
        values,
        1.0 - confidence
    )

    return float(
        max(
            -lower_quantile,
            0.0
        )
    )


def calculate_cvar(returns, confidence):
    """
    Conditional VaR / Expected Shortfall.

    Average loss in the tail beyond VaR.
    """

    values = np.asarray(
        returns,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    threshold = np.quantile(
        values,
        1.0 - confidence
    )

    tail = values[
        values <= threshold
    ]

    if len(tail) == 0:
        return float(
            max(
                -threshold,
                0.0
            )
        )

    return float(
        max(
            -np.mean(tail),
            0.0
        )
    )


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
    )

    print()
    print(
        "Feature file columns:"
    )

    print(
        df.columns.tolist()
    )

    date_column = find_column(
        df,
        [
            "date",
            "Date",
        ]
    )

    return_column = find_column(
        df,
        [
            "daily_return",
            "return",
            "log_return",
        ]
    )

    df["date_clean"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["market_return"] = pd.to_numeric(
        df[return_column],
        errors="coerce"
    )

    df = df[
        [
            "date_clean",
            "market_return",
        ]
    ].copy()

    df = (
        df
        .dropna()
        .sort_values("date_clean")
        .drop_duplicates(
            subset=["date_clean"]
        )
        .reset_index(drop=True)
    )

    df.rename(
        columns={
            "date_clean": "date"
        },
        inplace=True
    )

    return df


# ============================================================
# LOAD REGIME DATA
# ============================================================

def load_regime_data():

    if not REGIME_FILE.exists():
        raise FileNotFoundError(
            f"Regime file not found:\n{REGIME_FILE}"
        )

    df = pd.read_csv(
        REGIME_FILE
    )

    print()
    print(
        "Regime file columns:"
    )

    print(
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Actual Step 16 file contains:
    #
    # feature_date
    # forecast_date
    # regime
    # regime_label
    # regime_probability
    #
    # Therefore forecast_date is used as
    # the regime forecast date.
    # --------------------------------------------------------

    date_column = find_column(
        df,
        [
            "forecast_date",
            "feature_date",
            "date",
            "prediction_for_date",
            "forecast_for_date",
        ]
    )

    regime_column = find_column(
        df,
        [
            "regime_label",
            "regime",
            "regime_state",
        ]
    )

    probability_column = find_column(
        df,
        [
            "regime_probability",
            "probability",
            "regime_prob",
        ],
        required=False
    )

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["regime_label_clean"] = (
        df[regime_column]
        .astype(str)
        .str.strip()
    )

    if probability_column is not None:

        df["regime_probability_clean"] = (
            pd.to_numeric(
                df[probability_column],
                errors="coerce"
            )
        )

    else:

        df["regime_probability_clean"] = np.nan

    df = df[
        [
            "date",
            "regime_label_clean",
            "regime_probability_clean",
        ]
    ].copy()

    df = (
        df
        .dropna(
            subset=["date"]
        )
        .sort_values("date")
        .drop_duplicates(
            subset=["date"]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# MERGE MARKET + REGIME
# ============================================================

def prepare_regime_returns():

    market = load_market_data()

    regime = load_regime_data()

    print()
    print(
        "=" * 70
    )

    print(
        "DATA ALIGNMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"Market rows: {len(market)}"
    )

    print(
        f"Regime rows: {len(regime)}"
    )

    merged = market.merge(
        regime,
        on="date",
        how="inner"
    )

    merged = (
        merged
        .dropna(
            subset=[
                "market_return",
                "regime_label_clean",
            ]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Aligned rows: {len(merged)}"
    )

    if len(merged) < 100:
        raise ValueError(
            "Too few aligned market/regime observations."
        )

    return merged


# ============================================================
# NORMALIZE REGIMES
# ============================================================

def normalize_regime_name(value):

    text = str(
        value
    ).lower().strip()

    if (
        "high" in text
        or "risk-off" in text
        or "risk off" in text
    ):
        return "High_Volatility"

    if (
        "low" in text
        or "risk-on" in text
        or "risk on" in text
    ):
        return "Low_Volatility"

    if "medium" in text:
        return "Medium_Volatility"

    return "Other"


# ============================================================
# REGIME STATISTICS
# ============================================================

def calculate_regime_statistics(df):

    working = df.copy()

    working["risk_regime"] = (
        working["regime_label_clean"]
        .apply(
            normalize_regime_name
        )
    )

    rows = []

    for regime_name, group in (
        working.groupby("risk_regime")
    ):

        returns = clean_returns(
            group["market_return"]
        )

        if len(returns) < 2:
            continue

        rows.append(
            {
                "regime":
                    regime_name,

                "observations":
                    len(returns),

                "mean_daily_return":
                    returns.mean(),

                "median_daily_return":
                    returns.median(),

                "daily_volatility":
                    returns.std(
                        ddof=1
                    ),

                "annualized_volatility":
                    returns.std(
                        ddof=1
                    ) * np.sqrt(252),

                "minimum_daily_return":
                    returns.min(),

                "maximum_daily_return":
                    returns.max(),

                "negative_return_rate":
                    np.mean(
                        returns < 0
                    ),

                "VaR_90":
                    calculate_var(
                        returns,
                        0.90
                    ),

                "VaR_95":
                    calculate_var(
                        returns,
                        0.95
                    ),

                "VaR_99":
                    calculate_var(
                        returns,
                        0.99
                    ),

                "CVaR_90":
                    calculate_cvar(
                        returns,
                        0.90
                    ),

                "CVaR_95":
                    calculate_cvar(
                        returns,
                        0.95
                    ),

                "CVaR_99":
                    calculate_cvar(
                        returns,
                        0.99
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def simulate_regime(
    returns,
    n_simulations,
    horizon_days,
    seed
):

    values = clean_returns(
        returns
    ).values

    if len(values) < 2:
        raise ValueError(
            "Insufficient returns for simulation."
        )

    rng = np.random.default_rng(
        seed
    )

    sampled_returns = rng.choice(
        values,
        size=(
            n_simulations,
            horizon_days
        ),
        replace=True
    )

    cumulative_returns = (
        np.prod(
            1.0 + sampled_returns,
            axis=1
        )
        - 1.0
    )

    return (
        sampled_returns,
        cumulative_returns
    )


# ============================================================
# SIMULATION RISK METRICS
# ============================================================

def simulation_risk_metrics(
    cumulative_returns
):

    values = np.asarray(
        cumulative_returns,
        dtype=float
    )

    metrics = {}

    metrics["mean_return"] = float(
        np.mean(values)
    )

    metrics["median_return"] = float(
        np.median(values)
    )

    metrics["std_return"] = float(
        np.std(
            values,
            ddof=1
        )
    )

    metrics["probability_of_loss"] = float(
        np.mean(
            values < 0
        )
    )

    metrics["probability_loss_gt_5pct"] = float(
        np.mean(
            values <= -0.05
        )
    )

    metrics["probability_loss_gt_10pct"] = float(
        np.mean(
            values <= -0.10
        )
    )

    metrics["probability_gain_gt_5pct"] = float(
        np.mean(
            values >= 0.05
        )
    )

    metrics["VaR_90"] = calculate_var(
        values,
        0.90
    )

    metrics["VaR_95"] = calculate_var(
        values,
        0.95
    )

    metrics["VaR_99"] = calculate_var(
        values,
        0.99
    )

    metrics["CVaR_90"] = calculate_cvar(
        values,
        0.90
    )

    metrics["CVaR_95"] = calculate_cvar(
        values,
        0.95
    )

    metrics["CVaR_99"] = calculate_cvar(
        values,
        0.99
    )

    return metrics


# ============================================================
# CRISIS SCENARIOS
# ============================================================

def create_crisis_scenarios(
    historical_returns
):

    returns = clean_returns(
        historical_returns
    ).values

    q01 = np.quantile(
        returns,
        0.01
    )

    q05 = np.quantile(
        returns,
        0.05
    )

    q10 = np.quantile(
        returns,
        0.10
    )

    worst_return = np.min(
        returns
    )

    scenarios = [
        (
            "Historical_Worst_Day",
            worst_return
        ),

        (
            "Historical_1pct_Tail",
            q01
        ),

        (
            "Historical_5pct_Tail",
            q05
        ),

        (
            "Historical_10pct_Tail",
            q10
        ),
    ]

    rows = []

    for scenario, shock in scenarios:

        rows.append(
            {
                "scenario":
                    scenario,

                "shock_return":
                    shock,

                "one_day_loss":
                    max(
                        -shock,
                        0.0
                    ),

                "remaining_value":
                    1.0 + shock,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# CREATE VISUAL PATHS
# ============================================================

def create_visual_paths(
    sampled_returns,
    max_paths=40
):

    n_paths = min(
        max_paths,
        sampled_returns.shape[0]
    )

    selected = sampled_returns[
        :n_paths
    ]

    paths = np.cumprod(
        1.0 + selected,
        axis=1
    )

    return paths


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STEP 27 - REGIME-CONDITIONED "
        "MONTE CARLO RISK ENGINE"
    )
    print("=" * 70)

    np.random.seed(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # 1. Load and align data
    # --------------------------------------------------------

    df = prepare_regime_returns()

    # --------------------------------------------------------
    # 2. Normalize regime labels
    # --------------------------------------------------------

    df["risk_regime"] = (
        df["regime_label_clean"]
        .apply(
            normalize_regime_name
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "REGIME DISTRIBUTION"
    )

    print(
        "=" * 70
    )

    print(
        df["risk_regime"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # 3. Historical regime statistics
    # --------------------------------------------------------

    regime_metrics = (
        calculate_regime_statistics(
            df
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "REGIME RISK STATISTICS"
    )

    print(
        "=" * 70
    )

    if len(regime_metrics) > 0:

        print(
            regime_metrics.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # 4. Monte Carlo by regime
    # --------------------------------------------------------

    regime_names = [
        "Low_Volatility",
        "Medium_Volatility",
        "High_Volatility",
    ]

    all_simulation_rows = []

    regime_results = {}

    for index, regime_name in enumerate(
        regime_names
    ):

        regime_data = df[
            df["risk_regime"]
            == regime_name
        ]

        returns = clean_returns(
            regime_data[
                "market_return"
            ]
        )

        print()
        print(
            "-" * 70
        )

        print(
            f"REGIME: {regime_name}"
        )

        print(
            f"Historical observations: "
            f"{len(returns)}"
        )

        if len(returns) < MIN_REGIME_OBSERVATIONS:

            print(
                "WARNING: Insufficient observations. "
                "Skipping this regime."
            )

            continue

        (
            sampled_returns,
            cumulative_returns
        ) = simulate_regime(
            returns,
            N_SIMULATIONS,
            HORIZON_DAYS,
            RANDOM_SEED + index + 1
        )

        risk = simulation_risk_metrics(
            cumulative_returns
        )

        regime_results[
            regime_name
        ] = {
            "sampled_returns":
                sampled_returns,

            "cumulative_returns":
                cumulative_returns,

            "risk":
                risk,

            "observations":
                len(returns),
        }

        print(
            f"Mean {HORIZON_DAYS}-day return: "
            f"{risk['mean_return']:.4%}"
        )

        print(
            f"Median {HORIZON_DAYS}-day return: "
            f"{risk['median_return']:.4%}"
        )

        print(
            f"Probability of loss: "
            f"{risk['probability_of_loss']:.2%}"
        )

        print(
            f"VaR 90%: "
            f"{risk['VaR_90']:.4%}"
        )

        print(
            f"VaR 95%: "
            f"{risk['VaR_95']:.4%}"
        )

        print(
            f"VaR 99%: "
            f"{risk['VaR_99']:.4%}"
        )

        print(
            f"CVaR 95%: "
            f"{risk['CVaR_95']:.4%}"
        )

        # Store first 1,000 simulations
        # for an auditable CSV artifact.

        for simulation_id in range(
            min(
                1000,
                N_SIMULATIONS
            )
        ):

            all_simulation_rows.append(
                {
                    "regime":
                        regime_name,

                    "simulation_id":
                        simulation_id,

                    "horizon_days":
                        HORIZON_DAYS,

                    "simulated_return":
                        cumulative_returns[
                            simulation_id
                        ],
                }
            )

    # --------------------------------------------------------
    # 5. Overall Monte Carlo
    # --------------------------------------------------------

    overall_returns = clean_returns(
        df["market_return"]
    )

    (
        overall_sampled,
        overall_cumulative
    ) = simulate_regime(
        overall_returns,
        N_SIMULATIONS,
        HORIZON_DAYS,
        RANDOM_SEED
    )

    overall_risk = (
        simulation_risk_metrics(
            overall_cumulative
        )
    )

    print()
    print(
        "=" * 70
    )

    print(
        "OVERALL MONTE CARLO RISK"
    )

    print(
        "=" * 70
    )

    print(
        f"Mean {HORIZON_DAYS}-day return: "
        f"{overall_risk['mean_return']:.4%}"
    )

    print(
        f"Median {HORIZON_DAYS}-day return: "
        f"{overall_risk['median_return']:.4%}"
    )

    print(
        f"Probability of loss: "
        f"{overall_risk['probability_of_loss']:.2%}"
    )

    print(
        f"Probability loss > 5%: "
        f"{overall_risk['probability_loss_gt_5pct']:.2%}"
    )

    print(
        f"Probability loss > 10%: "
        f"{overall_risk['probability_loss_gt_10pct']:.2%}"
    )

    print(
        f"VaR 90%: "
        f"{overall_risk['VaR_90']:.4%}"
    )

    print(
        f"VaR 95%: "
        f"{overall_risk['VaR_95']:.4%}"
    )

    print(
        f"VaR 99%: "
        f"{overall_risk['VaR_99']:.4%}"
    )

    print(
        f"CVaR 90%: "
        f"{overall_risk['CVaR_90']:.4%}"
    )

    print(
        f"CVaR 95%: "
        f"{overall_risk['CVaR_95']:.4%}"
    )

    print(
        f"CVaR 99%: "
        f"{overall_risk['CVaR_99']:.4%}"
    )

    # --------------------------------------------------------
    # 6. Crisis scenarios
    # --------------------------------------------------------

    crisis_df = create_crisis_scenarios(
        overall_returns
    )

    print()
    print(
        "=" * 70
    )

    print(
        "HISTORICAL CRISIS / TAIL SCENARIOS"
    )

    print(
        "=" * 70
    )

    print(
        crisis_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 7. Risk metrics dataframe
    # --------------------------------------------------------

    risk_rows = []

    risk_rows.append(
        {
            "scenario":
                "Overall_Historical",

            "horizon_days":
                HORIZON_DAYS,

            "observations":
                len(overall_returns),

            **overall_risk,
        }
    )

    for regime_name, result in (
        regime_results.items()
    ):

        risk_rows.append(
            {
                "scenario":
                    regime_name,

                "horizon_days":
                    HORIZON_DAYS,

                "observations":
                    result[
                        "observations"
                    ],

                **result["risk"],
            }
        )

    risk_metrics_df = pd.DataFrame(
        risk_rows
    )

    # --------------------------------------------------------
    # 8. Save CSV outputs
    # --------------------------------------------------------

    simulation_df = pd.DataFrame(
        all_simulation_rows
    )

    simulation_df.to_csv(
        SIMULATION_FILE,
        index=False
    )

    risk_metrics_df.to_csv(
        RISK_METRICS_FILE,
        index=False
    )

    regime_metrics.to_csv(
        REGIME_METRICS_FILE,
        index=False
    )

    crisis_df.to_csv(
        CRISIS_METRICS_FILE,
        index=False
    )

    # ========================================================
    # 9. SIMULATION PATH PLOT
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    for regime_name, result in (
        regime_results.items()
    ):

        paths = create_visual_paths(
            result[
                "sampled_returns"
            ]
        )

        for path in paths:

            plt.plot(
                range(
                    1,
                    HORIZON_DAYS + 1
                ),
                path,
                alpha=0.10
            )

    plt.axhline(
        1.0,
        linestyle="--",
        label="Initial Value"
    )

    plt.xlabel(
        "Trading Day"
    )

    plt.ylabel(
        "Simulated Wealth Index"
    )

    plt.title(
        "Regime-Conditioned Monte Carlo Paths"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        SIMULATION_PLOT,
        dpi=150
    )

    plt.close()

    # ========================================================
    # 10. RETURN DISTRIBUTION
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    plt.hist(
        overall_cumulative,
        bins=80,
        density=True,
        alpha=0.75
    )

    plt.axvline(
        -overall_risk["VaR_95"],
        linestyle="--",
        label="VaR 95%"
    )

    plt.axvline(
        -overall_risk["CVaR_95"],
        linestyle="--",
        label="CVaR 95%"
    )

    plt.xlabel(
        f"{HORIZON_DAYS}-Day Simulated Return"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Monte Carlo Return Distribution"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        RETURN_DISTRIBUTION_PLOT,
        dpi=150
    )

    plt.close()

    # ========================================================
    # 11. REGIME RISK COMPARISON
    # ========================================================

    regime_plot_df = risk_metrics_df[
        risk_metrics_df[
            "scenario"
        ].isin(
            regime_results.keys()
        )
    ].copy()

    if len(regime_plot_df) > 0:

        plt.figure(
            figsize=(10, 6)
        )

        plt.bar(
            regime_plot_df[
                "scenario"
            ],
            regime_plot_df[
                "VaR_95"
            ]
        )

        plt.xlabel(
            "Market Regime"
        )

        plt.ylabel(
            "95% VaR"
        )

        plt.title(
            "Regime-Conditioned 95% VaR"
        )

        plt.grid(
            axis="y",
            alpha=0.3
        )

        plt.tight_layout()

        plt.savefig(
            REGIME_RISK_PLOT,
            dpi=150
        )

        plt.close()

    # ========================================================
    # 12. SUMMARY REPORT
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 27 - MONTE CARLO RISK ENGINE\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "Purpose:\n"
        )

        f.write(
            "Regime-conditioned Monte Carlo simulation "
            "and downside risk analysis for the "
            "Indian equity market.\n\n"
        )

        f.write(
            f"Simulation count: "
            f"{N_SIMULATIONS}\n"
        )

        f.write(
            f"Simulation horizon: "
            f"{HORIZON_DAYS} trading days\n"
        )

        f.write(
            "Sampling method: "
            "regime-conditioned empirical bootstrap\n"
        )

        f.write(
            f"Random seed: "
            f"{RANDOM_SEED}\n\n"
        )

        f.write(
            "OVERALL RISK\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for key, value in (
            overall_risk.items()
        ):

            f.write(
                f"{key}: {value:.8f}\n"
            )

        f.write(
            "\nREGIME RISK\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            risk_metrics_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nHISTORICAL TAIL SCENARIOS\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            crisis_df.to_string(
                index=False
            )
        )

        f.write(
            "\n\nMETHODOLOGICAL NOTES\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "1. Returns are sampled from observed "
            "historical returns.\n"
        )

        f.write(
            "2. Regime-specific simulations use "
            "empirical regime-conditioned returns.\n"
        )

        f.write(
            "3. No future returns are used to generate "
            "the simulated paths.\n"
        )

        f.write(
            "4. VaR is an empirical lower-tail loss "
            "quantile.\n"
        )

        f.write(
            "5. CVaR is the average simulated loss "
            "beyond the VaR threshold.\n"
        )

        f.write(
            "6. Historical tail scenarios are based "
            "on observed return quantiles.\n"
        )

        f.write(
            "7. Monte Carlo simulation is a risk-analysis "
            "framework and does not guarantee future "
            "market outcomes.\n"
        )

    # ========================================================
    # 13. CONFIG
    # ========================================================

    config = {

        "step":
            27,

        "project":
            "Bayesian Regime Detection Engine",

        "input_market_file":
            str(
                FEATURE_FILE.relative_to(ROOT)
            ),

        "input_regime_file":
            str(
                REGIME_FILE.relative_to(ROOT)
            ),

        "date_column_used":
            "forecast_date",

        "regime_column_used":
            "regime_label",

        "simulation_method":
            "Regime-conditioned empirical bootstrap",

        "n_simulations":
            N_SIMULATIONS,

        "horizon_days":
            HORIZON_DAYS,

        "random_seed":
            RANDOM_SEED,

        "var_levels":
            VAR_LEVELS,

        "min_regime_observations":
            MIN_REGIME_OBSERVATIONS,

        "regimes":
            list(
                regime_results.keys()
            ),

        "risk_metrics":
            [
                "mean_return",
                "median_return",
                "std_return",
                "probability_of_loss",
                "probability_loss_gt_5pct",
                "probability_loss_gt_10pct",
                "probability_gain_gt_5pct",
                "VaR_90",
                "VaR_95",
                "VaR_99",
                "CVaR_90",
                "CVaR_95",
                "CVaR_99",
            ],

        "crisis_scenarios":
            [
                "Historical_Worst_Day",
                "Historical_1pct_Tail",
                "Historical_5pct_Tail",
                "Historical_10pct_Tail",
            ],
    }

    with open(
        CONFIG_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    # ========================================================
    # 14. VALIDATION
    # ========================================================

    validation = {}

    validation[
        "feature_file_exists"
    ] = FEATURE_FILE.exists()

    validation[
        "regime_file_exists"
    ] = REGIME_FILE.exists()

    validation[
        "simulation_file_exists"
    ] = SIMULATION_FILE.exists()

    validation[
        "risk_metrics_exists"
    ] = RISK_METRICS_FILE.exists()

    validation[
        "regime_metrics_exists"
    ] = REGIME_METRICS_FILE.exists()

    validation[
        "crisis_metrics_exists"
    ] = CRISIS_METRICS_FILE.exists()

    validation[
        "summary_exists"
    ] = SUMMARY_FILE.exists()

    validation[
        "simulation_plot_exists"
    ] = SIMULATION_PLOT.exists()

    validation[
        "return_distribution_plot_exists"
    ] = RETURN_DISTRIBUTION_PLOT.exists()

    validation[
        "regime_risk_plot_exists"
    ] = (
        REGIME_RISK_PLOT.exists()
    )

    validation[
        "config_exists"
    ] = CONFIG_FILE.exists()

    validation[
        "market_dates_sorted"
    ] = bool(
        df[
            "date"
        ].is_monotonic_increasing
    )

    validation[
        "market_dates_unique"
    ] = bool(
        df[
            "date"
        ].duplicated().sum()
        == 0
    )

    validation[
        "returns_finite"
    ] = bool(
        np.isfinite(
            df[
                "market_return"
            ].values
        ).all()
    )

    validation[
        "simulation_count_valid"
    ] = bool(
        N_SIMULATIONS >= 1000
    )

    validation[
        "horizon_valid"
    ] = bool(
        HORIZON_DAYS >= 1
    )

    validation[
        "overall_simulation_valid"
    ] = bool(
        len(
            overall_cumulative
        )
        == N_SIMULATIONS
    )

    validation[
        "simulation_values_finite"
    ] = bool(
        np.isfinite(
            overall_cumulative
        ).all()
    )

    validation[
        "risk_metrics_finite"
    ] = bool(
        np.isfinite(
            list(
                overall_risk.values()
            )
        ).all()
    )

    validation[
        "var_cvar_nonnegative"
    ] = bool(
        all(
            overall_risk[key] >= 0
            for key in [
                "VaR_90",
                "VaR_95",
                "VaR_99",
                "CVaR_90",
                "CVaR_95",
                "CVaR_99",
            ]
        )
    )

    validation[
        "probabilities_valid"
    ] = bool(
        all(
            0.0
            <= overall_risk[key]
            <= 1.0
            for key in [
                "probability_of_loss",
                "probability_loss_gt_5pct",
                "probability_loss_gt_10pct",
                "probability_gain_gt_5pct",
            ]
        )
    )

    validation[
        "regime_metrics_nonempty"
    ] = bool(
        len(
            regime_metrics
        ) > 0
    )

    validation[
        "crisis_metrics_nonempty"
    ] = bool(
        len(
            crisis_df
        ) >= 4
    )

    validation[
        "all_required_outputs"
    ] = all(
        [
            SIMULATION_FILE.exists(),
            RISK_METRICS_FILE.exists(),
            REGIME_METRICS_FILE.exists(),
            CRISIS_METRICS_FILE.exists(),
            SUMMARY_FILE.exists(),
            SIMULATION_PLOT.exists(),
            RETURN_DISTRIBUTION_PLOT.exists(),
            CONFIG_FILE.exists(),
        ]
    )

    all_passed = all(
        validation.values()
    )

    # ========================================================
    # VALIDATION OUTPUT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "STEP 27 VALIDATION"
    )

    print("=" * 70)

    for key, value in (
        validation.items()
    ):

        print(
            f"{key}: {value}"
        )

    if not all_passed:

        print()
        print(
            "STEP 27 VALIDATION: FAILED"
        )

        failed = [
            key
            for key, value
            in validation.items()
            if not value
        ]

        print(
            "Failed checks:"
        )

        for item in failed:
            print(
                f"  - {item}"
            )

        raise RuntimeError(
            "Step 27 validation failed."
        )

    print()
    print(
        "STEP 27 VALIDATION: PASSED"
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 70)

    print(
        "STEP 27 COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print()
    print(
        "Generated files:"
    )

    print(
        f"1. {SIMULATION_FILE}"
    )

    print(
        f"2. {RISK_METRICS_FILE}"
    )

    print(
        f"3. {REGIME_METRICS_FILE}"
    )

    print(
        f"4. {CRISIS_METRICS_FILE}"
    )

    print(
        f"5. {SUMMARY_FILE}"
    )

    print(
        f"6. {SIMULATION_PLOT}"
    )

    print(
        f"7. {RETURN_DISTRIBUTION_PLOT}"
    )

    print(
        f"8. {REGIME_RISK_PLOT}"
    )

    print(
        f"9. {CONFIG_FILE}"
    )


if __name__ == "__main__":
    main()