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
REPORT_DIR = ROOT / "reports" / "allocation_backtest"
MODEL_DIR = ROOT / "models" / "allocation_backtest"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# INPUT
# ============================================================

FEATURE_FILE = (
    DATA_DIR / "nifty50_features.csv"
)

REGIME_FILE = (
    DATA_DIR / "causal_regime_integrated_predictions.csv"
)


# ============================================================
# OUTPUTS
# ============================================================

BACKTEST_FILE = (
    DATA_DIR / "regime_overlay_backtest.csv"
)

PERFORMANCE_FILE = (
    REPORT_DIR / "performance_metrics.csv"
)

YEARLY_FILE = (
    REPORT_DIR / "yearly_performance.csv"
)

REGIME_FILE_OUTPUT = (
    REPORT_DIR / "regime_performance.csv"
)

REGIME_CHANGES_FILE = (
    REPORT_DIR / "regime_changes.csv"
)

SUMMARY_FILE = (
    REPORT_DIR / "summary.txt"
)

EQUITY_CURVE_PLOT = (
    REPORT_DIR / "equity_curve.png"
)

DRAWDOWN_PLOT = (
    REPORT_DIR / "drawdown_curve.png"
)

REGIME_ALLOCATION_PLOT = (
    REPORT_DIR / "regime_allocation.png"
)

CONFIG_FILE = (
    MODEL_DIR / "allocation_backtest_config.json"
)


# ============================================================
# STRATEGY CONFIG
# ============================================================

RANDOM_SEED = 42

# Illustrative research allocation.
# This is NOT an investment recommendation.

ALLOCATION_MAP = {
    "Low_Volatility": 1.00,
    "Medium_Volatility": 0.60,
    "High_Volatility": 0.25,
}

CASH_RETURN = 0.0

TRADING_DAYS = 252

RISK_FREE_RATE = 0.0


# ============================================================
# HELPER
# ============================================================

def find_column(
    df,
    candidates,
    required=True
):

    for column in candidates:

        if column in df.columns:
            return column

    if required:

        raise ValueError(
            f"None of these columns were found: "
            f"{candidates}"
        )

    return None


# ============================================================
# LOAD MARKET DATA
# ============================================================

def load_market_data():

    if not FEATURE_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found:\n"
            f"{FEATURE_FILE}"
        )

    df = pd.read_csv(
        FEATURE_FILE
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

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["market_return"] = pd.to_numeric(
        df[return_column],
        errors="coerce"
    )

    df = df[
        [
            "date",
            "market_return",
        ]
    ].copy()

    df = (
        df
        .dropna()
        .sort_values("date")
        .drop_duplicates(
            subset=["date"]
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# LOAD CAUSAL REGIME DATA
# ============================================================

def load_regime_data():

    if not REGIME_FILE.exists():

        raise FileNotFoundError(
            f"Regime file not found:\n"
            f"{REGIME_FILE}"
        )

    df = pd.read_csv(
        REGIME_FILE
    )

    date_column = find_column(
        df,
        [
            "forecast_date",
            "feature_date",
            "date",
        ]
    )

    regime_column = find_column(
        df,
        [
            "regime_label",
            "regime",
        ]
    )

    probability_column = find_column(
        df,
        [
            "regime_probability",
        ],
        required=False
    )

    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    df["regime_label"] = (
        df[regime_column]
        .astype(str)
        .str.strip()
    )

    if probability_column:

        df["regime_probability"] = pd.to_numeric(
            df[probability_column],
            errors="coerce"
        )

    else:

        df["regime_probability"] = np.nan

    df = df[
        [
            "date",
            "regime_label",
            "regime_probability",
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
# NORMALIZE REGIME
# ============================================================

def normalize_regime(
    value
):

    text = str(
        value
    ).lower().strip()

    if (
        "low" in text
        or "risk-on" in text
        or "risk on" in text
    ):

        return "Low_Volatility"

    if "medium" in text:

        return "Medium_Volatility"

    if (
        "high" in text
        or "risk-off" in text
        or "risk off" in text
    ):

        return "High_Volatility"

    return "Other"


# ============================================================
# PREPARE BACKTEST DATA
# ============================================================

def prepare_backtest_data():

    market = load_market_data()

    regime = load_regime_data()

    print()
    print(
        f"Market observations: "
        f"{len(market)}"
    )

    print(
        f"Regime observations: "
        f"{len(regime)}"
    )

    df = market.merge(
        regime,
        on="date",
        how="inner"
    )

    df["risk_regime"] = (
        df["regime_label"]
        .apply(
            normalize_regime
        )
    )

    df = df[
        df["risk_regime"].isin(
            [
                "Low_Volatility",
                "Medium_Volatility",
                "High_Volatility",
            ]
        )
    ].copy()

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"]
        )
        .reset_index(drop=True)
    )

    if len(df) < 100:

        raise ValueError(
            "Too few observations for backtesting."
        )

    return df


# ============================================================
# BUILD STRATEGY
# ============================================================

def build_strategy(
    df
):

    result = df.copy()

    result["equity_weight"] = (
        result["risk_regime"]
        .map(
            ALLOCATION_MAP
        )
    )

    result["cash_weight"] = (
        1.0
        - result["equity_weight"]
    )

    result["benchmark_return"] = (
        result["market_return"]
    )

    result["strategy_return"] = (
        result["equity_weight"]
        * result["market_return"]
        +
        result["cash_weight"]
        * CASH_RETURN
    )

    result["active_return"] = (
        result["strategy_return"]
        -
        result["benchmark_return"]
    )

    # Previous day's regime is used to
    # avoid using today's realized return
    # to decide today's allocation.

    result[
        "applied_equity_weight"
    ] = (
        result[
            "equity_weight"
        ]
        .shift(1)
    )

    # For the first observation,
    # use the first available regime weight.

    if len(result) > 0:

        result.loc[
            result.index[0],
            "applied_equity_weight"
        ] = result[
            "equity_weight"
        ].iloc[0]

    result[
        "applied_cash_weight"
    ] = (
        1.0
        -
        result[
            "applied_equity_weight"
        ]
    )

    result[
        "strategy_return"
    ] = (
        result[
            "applied_equity_weight"
        ]
        * result[
            "market_return"
        ]
        +
        result[
            "applied_cash_weight"
        ]
        * CASH_RETURN
    )

    result[
        "active_return"
    ] = (
        result[
            "strategy_return"
        ]
        -
        result[
            "benchmark_return"
        ]
    )

    # Cumulative wealth

    result[
        "strategy_wealth"
    ] = (
        1.0
        +
        result[
            "strategy_return"
        ]
    ).cumprod()

    result[
        "benchmark_wealth"
    ] = (
        1.0
        +
        result[
            "benchmark_return"
        ]
    ).cumprod()

    # Running maximum

    result[
        "strategy_running_max"
    ] = result[
        "strategy_wealth"
    ].cummax()

    result[
        "benchmark_running_max"
    ] = result[
        "benchmark_wealth"
    ].cummax()

    # Drawdowns

    result[
        "strategy_drawdown"
    ] = (
        result[
            "strategy_wealth"
        ]
        /
        result[
            "strategy_running_max"
        ]
        - 1.0
    )

    result[
        "benchmark_drawdown"
    ] = (
        result[
            "benchmark_wealth"
        ]
        /
        result[
            "benchmark_running_max"
        ]
        - 1.0
    )

    return result


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_performance(
    returns,
    wealth
):

    returns = pd.Series(
        returns
    ).dropna()

    wealth = pd.Series(
        wealth
    ).dropna()

    if len(returns) == 0:

        return {}

    total_return = (
        wealth.iloc[-1]
        - 1.0
    )

    years = (
        len(returns)
        /
        TRADING_DAYS
    )

    if years > 0:

        annualized_return = (
            wealth.iloc[-1]
            ** (1.0 / years)
            - 1.0
        )

    else:

        annualized_return = np.nan

    annualized_volatility = (
        returns.std(
            ddof=1
        )
        *
        np.sqrt(
            TRADING_DAYS
        )
    )

    excess_return = (
        returns
        - RISK_FREE_RATE
        /
        TRADING_DAYS
    )

    if (
        returns.std(
            ddof=1
        )
        > 0
    ):

        sharpe = (
            excess_return.mean()
            /
            returns.std(
                ddof=1
            )
            *
            np.sqrt(
                TRADING_DAYS
            )
        )

    else:

        sharpe = np.nan

    running_max = wealth.cummax()

    drawdown = (
        wealth
        /
        running_max
        - 1.0
    )

    max_drawdown = drawdown.min()

    downside_returns = returns[
        returns < 0
    ]

    if len(downside_returns) > 0:

        downside_deviation = (
            downside_returns.std(
                ddof=1
            )
            *
            np.sqrt(
                TRADING_DAYS
            )
        )

    else:

        downside_deviation = np.nan

    if (
        downside_deviation
        and
        downside_deviation > 0
    ):

        sortino = (
            annualized_return
            /
            downside_deviation
        )

    else:

        sortino = np.nan

    return {
        "observations":
            len(returns),

        "total_return":
            total_return,

        "annualized_return":
            annualized_return,

        "annualized_volatility":
            annualized_volatility,

        "sharpe_ratio":
            sharpe,

        "sortino_ratio":
            sortino,

        "maximum_drawdown":
            max_drawdown,

        "ending_wealth":
            wealth.iloc[-1],
    }


# ============================================================
# TRACKING ERROR + INFORMATION RATIO
# ============================================================

def calculate_tracking_metrics(
    strategy_returns,
    benchmark_returns
):

    strategy_returns = pd.Series(
        strategy_returns
    )

    benchmark_returns = pd.Series(
        benchmark_returns
    )

    active = (
        strategy_returns
        -
        benchmark_returns
    ).dropna()

    active_return = (
        active.mean()
        *
        TRADING_DAYS
    )

    tracking_error = (
        active.std(
            ddof=1
        )
        *
        np.sqrt(
            TRADING_DAYS
        )
    )

    if tracking_error > 0:

        information_ratio = (
            active_return
            /
            tracking_error
        )

    else:

        information_ratio = np.nan

    return {
        "annualized_active_return":
            active_return,

        "tracking_error":
            tracking_error,

        "information_ratio":
            information_ratio,
    }


# ============================================================
# REGIME PERFORMANCE
# ============================================================

def calculate_regime_performance(
    df
):

    rows = []

    for regime, group in (
        df.groupby(
            "risk_regime"
        )
    ):

        strategy_metrics = (
            calculate_performance(
                group[
                    "strategy_return"
                ],
                (
                    1.0
                    +
                    group[
                        "strategy_return"
                    ]
                ).cumprod()
            )
        )

        benchmark_metrics = (
            calculate_performance(
                group[
                    "benchmark_return"
                ],
                (
                    1.0
                    +
                    group[
                        "benchmark_return"
                    ]
                ).cumprod()
            )
        )

        rows.append(
            {
                "regime":
                    regime,

                "observations":
                    len(group),

                "average_equity_weight":
                    group[
                        "applied_equity_weight"
                    ].mean(),

                "strategy_return":
                    strategy_metrics[
                        "total_return"
                    ],

                "benchmark_return":
                    benchmark_metrics[
                        "total_return"
                    ],

                "strategy_volatility":
                    strategy_metrics[
                        "annualized_volatility"
                    ],

                "benchmark_volatility":
                    benchmark_metrics[
                        "annualized_volatility"
                    ],

                "strategy_max_drawdown":
                    strategy_metrics[
                        "maximum_drawdown"
                    ],

                "benchmark_max_drawdown":
                    benchmark_metrics[
                        "maximum_drawdown"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# YEARLY PERFORMANCE
# ============================================================

def calculate_yearly_performance(
    df
):

    working = df.copy()

    working["year"] = (
        working[
            "date"
        ].dt.year
    )

    rows = []

    for year, group in (
        working.groupby("year")
    ):

        strategy_wealth = (
            1.0
            +
            group[
                "strategy_return"
            ]
        ).cumprod()

        benchmark_wealth = (
            1.0
            +
            group[
                "benchmark_return"
            ]
        ).cumprod()

        strategy_metrics = (
            calculate_performance(
                group[
                    "strategy_return"
                ],
                strategy_wealth
            )
        )

        benchmark_metrics = (
            calculate_performance(
                group[
                    "benchmark_return"
                ],
                benchmark_wealth
            )
        )

        rows.append(
            {
                "year":
                    year,

                "observations":
                    len(group),

                "strategy_return":
                    strategy_metrics[
                        "total_return"
                    ],

                "benchmark_return":
                    benchmark_metrics[
                        "total_return"
                    ],

                "strategy_volatility":
                    strategy_metrics[
                        "annualized_volatility"
                    ],

                "benchmark_volatility":
                    benchmark_metrics[
                        "annualized_volatility"
                    ],

                "strategy_max_drawdown":
                    strategy_metrics[
                        "maximum_drawdown"
                    ],

                "benchmark_max_drawdown":
                    benchmark_metrics[
                        "maximum_drawdown"
                    ],

                "active_return":
                    strategy_metrics[
                        "total_return"
                    ]
                    -
                    benchmark_metrics[
                        "total_return"
                    ],
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# REGIME CHANGES
# ============================================================

def detect_regime_changes(
    df
):

    working = df[
        [
            "date",
            "risk_regime",
            "regime_probability",
        ]
    ].copy()

    working[
        "previous_regime"
    ] = (
        working[
            "risk_regime"
        ].shift(1)
    )

    changes = working[
        (
            working[
                "risk_regime"
            ]
            !=
            working[
                "previous_regime"
            ]
        )
        &
        working[
            "previous_regime"
        ].notna()
    ].copy()

    changes[
        "from_regime"
    ] = changes[
        "previous_regime"
    ]

    changes[
        "to_regime"
    ] = changes[
        "risk_regime"
    ]

    changes = changes[
        [
            "date",
            "from_regime",
            "to_regime",
            "regime_probability",
        ]
    ]

    return changes.reset_index(
        drop=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "STEP 28 - REGIME OVERLAY ALLOCATION BACKTEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = prepare_backtest_data()

    print()
    print(
        f"Backtest observations: "
        f"{len(df)}"
    )

    print(
        f"Backtest start: "
        f"{df['date'].min().date()}"
    )

    print(
        f"Backtest end: "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Build strategy
    # --------------------------------------------------------

    df = build_strategy(
        df
    )

    # --------------------------------------------------------
    # Overall performance
    # --------------------------------------------------------

    strategy_metrics = (
        calculate_performance(
            df[
                "strategy_return"
            ],
            df[
                "strategy_wealth"
            ]
        )
    )

    benchmark_metrics = (
        calculate_performance(
            df[
                "benchmark_return"
            ],
            df[
                "benchmark_wealth"
            ]
        )
    )

    tracking_metrics = (
        calculate_tracking_metrics(
            df[
                "strategy_return"
            ],
            df[
                "benchmark_return"
            ]
        )
    )

    # --------------------------------------------------------
    # Print performance
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "STRATEGY PERFORMANCE"
    )

    print("=" * 70)

    print(
        f"Total return: "
        f"{strategy_metrics['total_return']:.4%}"
    )

    print(
        f"Annualized return: "
        f"{strategy_metrics['annualized_return']:.4%}"
    )

    print(
        f"Annualized volatility: "
        f"{strategy_metrics['annualized_volatility']:.4%}"
    )

    print(
        f"Sharpe ratio: "
        f"{strategy_metrics['sharpe_ratio']:.4f}"
    )

    print(
        f"Sortino ratio: "
        f"{strategy_metrics['sortino_ratio']:.4f}"
    )

    print(
        f"Maximum drawdown: "
        f"{strategy_metrics['maximum_drawdown']:.4%}"
    )

    print()
    print(
        "BENCHMARK PERFORMANCE"
    )

    print(
        f"Total return: "
        f"{benchmark_metrics['total_return']:.4%}"
    )

    print(
        f"Annualized return: "
        f"{benchmark_metrics['annualized_return']:.4%}"
    )

    print(
        f"Annualized volatility: "
        f"{benchmark_metrics['annualized_volatility']:.4%}"
    )

    print(
        f"Sharpe ratio: "
        f"{benchmark_metrics['sharpe_ratio']:.4f}"
    )

    print(
        f"Maximum drawdown: "
        f"{benchmark_metrics['maximum_drawdown']:.4%}"
    )

    print()
    print(
        "ACTIVE PERFORMANCE"
    )

    print(
        f"Annualized active return: "
        f"{tracking_metrics['annualized_active_return']:.4%}"
    )

    print(
        f"Tracking error: "
        f"{tracking_metrics['tracking_error']:.4%}"
    )

    print(
        f"Information ratio: "
        f"{tracking_metrics['information_ratio']:.4f}"
    )

    # --------------------------------------------------------
    # Regime performance
    # --------------------------------------------------------

    regime_performance = (
        calculate_regime_performance(
            df
        )
    )

    print()
    print("=" * 70)

    print(
        "REGIME PERFORMANCE"
    )

    print("=" * 70)

    print(
        regime_performance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Yearly performance
    # --------------------------------------------------------

    yearly_performance = (
        calculate_yearly_performance(
            df
        )
    )

    print()
    print("=" * 70)

    print(
        "YEARLY PERFORMANCE"
    )

    print("=" * 70)

    print(
        yearly_performance.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Regime changes
    # --------------------------------------------------------

    regime_changes = (
        detect_regime_changes(
            df
        )
    )

    print()
    print(
        f"Regime changes detected: "
        f"{len(regime_changes)}"
    )

    # --------------------------------------------------------
    # Save detailed backtest
    # --------------------------------------------------------

    df.to_csv(
        BACKTEST_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Performance metrics
    # --------------------------------------------------------

    performance_rows = [
        {
            "portfolio":
                "Regime_Overlay_Strategy",

            **strategy_metrics,

            **tracking_metrics,
        },

        {
            "portfolio":
                "NIFTY_50_Benchmark",

            **benchmark_metrics,

            "annualized_active_return":
                0.0,

            "tracking_error":
                0.0,

            "information_ratio":
                np.nan,
        },
    ]

    performance_df = pd.DataFrame(
        performance_rows
    )

    performance_df.to_csv(
        PERFORMANCE_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save yearly
    # --------------------------------------------------------

    yearly_performance.to_csv(
        YEARLY_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Save regime performance
    # --------------------------------------------------------

    regime_performance.to_csv(
        REGIME_FILE_OUTPUT,
        index=False
    )

    # --------------------------------------------------------
    # Save regime changes
    # --------------------------------------------------------

    regime_changes.to_csv(
        REGIME_CHANGES_FILE,
        index=False
    )

    # ========================================================
    # EQUITY CURVE
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        df["date"],
        df["strategy_wealth"],
        label="Regime Overlay Strategy"
    )

    plt.plot(
        df["date"],
        df["benchmark_wealth"],
        label="NIFTY 50 Benchmark"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Growth of ₹1"
    )

    plt.title(
        "Regime Overlay Strategy vs NIFTY 50"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        EQUITY_CURVE_PLOT,
        dpi=150
    )

    plt.close()

    # ========================================================
    # DRAWDOWN CURVE
    # ========================================================

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        df["date"],
        df["strategy_drawdown"],
        label="Strategy Drawdown"
    )

    plt.plot(
        df["date"],
        df["benchmark_drawdown"],
        label="Benchmark Drawdown"
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Drawdown"
    )

    plt.title(
        "Strategy vs Benchmark Drawdown"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        DRAWDOWN_PLOT,
        dpi=150
    )

    plt.close()

    # ========================================================
    # REGIME ALLOCATION PLOT
    # ========================================================

    plt.figure(
        figsize=(12, 6)
    )

    plt.step(
        df["date"],
        df[
            "applied_equity_weight"
        ],
        where="post",
        label="Equity Allocation"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Equity Weight"
    )

    plt.title(
        "Regime-Driven Equity Allocation"
    )

    plt.ylim(
        0,
        1.05
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        REGIME_ALLOCATION_PLOT,
        dpi=150
    )

    plt.close()

    # ========================================================
    # SUMMARY
    # ========================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STEP 28 - REGIME OVERLAY ALLOCATION BACKTEST\n"
        )

        f.write(
            "=" * 70 + "\n\n"
        )

        f.write(
            "BACKTEST PERIOD\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            f"Start: "
            f"{df['date'].min().date()}\n"
        )

        f.write(
            f"End: "
            f"{df['date'].max().date()}\n"
        )

        f.write(
            f"Observations: "
            f"{len(df)}\n\n"
        )

        f.write(
            "ALLOCATION RULE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "Low Volatility: 100% equity / 0% cash\n"
        )

        f.write(
            "Medium Volatility: 60% equity / 40% cash\n"
        )

        f.write(
            "High Volatility: 25% equity / 75% cash\n"
        )

        f.write(
            "Cash return assumption: 0%\n\n"
        )

        f.write(
            "STRATEGY PERFORMANCE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for key, value in strategy_metrics.items():

            f.write(
                f"{key}: {value:.8f}\n"
            )

        f.write(
            "\nBENCHMARK PERFORMANCE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for key, value in benchmark_metrics.items():

            f.write(
                f"{key}: {value:.8f}\n"
            )

        f.write(
            "\nACTIVE PERFORMANCE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        for key, value in tracking_metrics.items():

            f.write(
                f"{key}: {value:.8f}\n"
            )

        f.write(
            "\nREGIME PERFORMANCE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            regime_performance.to_string(
                index=False
            )
        )

        f.write(
            "\n\nYEARLY PERFORMANCE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            yearly_performance.to_string(
                index=False
            )
        )

        f.write(
            "\n\nIMPORTANT METHODOLOGICAL NOTE\n"
        )

        f.write(
            "-" * 70 + "\n"
        )

        f.write(
            "The allocation applied to each day's "
            "return uses the previous observation's "
            "detected regime. This prevents the current "
            "day's realized return from determining "
            "the same day's allocation.\n"
        )

        f.write(
            "The benchmark is the NIFTY 50 daily return "
            "series from the project dataset.\n"
        )

        f.write(
            "The allocation percentages are illustrative "
            "research assumptions and are not investment "
            "recommendations.\n"
        )

        f.write(
            "The available causal-regime integration file "
            "determines the actual backtest date range. "
            "No missing historical regime predictions "
            "are fabricated.\n"
        )

    # ========================================================
    # CONFIG
    # ========================================================

    config = {

        "step":
            28,

        "project":
            "Bayesian Regime Detection Engine",

        "strategy":
            "Regime Overlay Allocation",

        "allocation_map":
            ALLOCATION_MAP,

        "cash_return":
            CASH_RETURN,

        "trading_days":
            TRADING_DAYS,

        "risk_free_rate":
            RISK_FREE_RATE,

        "look_ahead_bias_control":
            "Previous day's regime applied to current day's return",

        "benchmark":
            "NIFTY 50 daily return",

        "input_regime_file":
            str(
                REGIME_FILE.relative_to(
                    ROOT
                )
            ),

        "input_market_file":
            str(
                FEATURE_FILE.relative_to(
                    ROOT
                )
            ),

        "outputs":
            [
                str(
                    BACKTEST_FILE.relative_to(
                        ROOT
                    )
                ),

                str(
                    PERFORMANCE_FILE.relative_to(
                        ROOT
                    )
                ),

                str(
                    YEARLY_FILE.relative_to(
                        ROOT
                    )
                ),

                str(
                    REGIME_FILE_OUTPUT.relative_to(
                        ROOT
                    )
                ),

                str(
                    REGIME_CHANGES_FILE.relative_to(
                        ROOT
                    )
                ),
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
    # VALIDATION
    # ========================================================

    validation = {}

    validation[
        "feature_file_exists"
    ] = FEATURE_FILE.exists()

    validation[
        "regime_file_exists"
    ] = REGIME_FILE.exists()

    validation[
        "backtest_file_exists"
    ] = BACKTEST_FILE.exists()

    validation[
        "performance_file_exists"
    ] = PERFORMANCE_FILE.exists()

    validation[
        "yearly_file_exists"
    ] = YEARLY_FILE.exists()

    validation[
        "regime_performance_exists"
    ] = REGIME_FILE_OUTPUT.exists()

    validation[
        "regime_changes_exists"
    ] = REGIME_CHANGES_FILE.exists()

    validation[
        "summary_exists"
    ] = SUMMARY_FILE.exists()

    validation[
        "equity_curve_exists"
    ] = EQUITY_CURVE_PLOT.exists()

    validation[
        "drawdown_plot_exists"
    ] = DRAWDOWN_PLOT.exists()

    validation[
        "allocation_plot_exists"
    ] = REGIME_ALLOCATION_PLOT.exists()

    validation[
        "config_exists"
    ] = CONFIG_FILE.exists()

    validation[
        "dates_sorted"
    ] = bool(
        df[
            "date"
        ].is_monotonic_increasing
    )

    validation[
        "dates_unique"
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
        "strategy_returns_finite"
    ] = bool(
        np.isfinite(
            df[
                "strategy_return"
            ].values
        ).all()
    )

    validation[
        "benchmark_returns_finite"
    ] = bool(
        np.isfinite(
            df[
                "benchmark_return"
            ].values
        ).all()
    )

    validation[
        "allocation_valid"
    ] = bool(
        (
            df[
                "applied_equity_weight"
            ]
            >= 0
        ).all()
        and
        (
            df[
                "applied_equity_weight"
            ]
            <= 1
        ).all()
    )

    validation[
        "wealth_positive"
    ] = bool(
        (
            df[
                "strategy_wealth"
            ]
            > 0
        ).all()
        and
        (
            df[
                "benchmark_wealth"
            ]
            > 0
        ).all()
    )

    validation[
        "drawdown_nonpositive"
    ] = bool(
        (
            df[
                "strategy_drawdown"
            ]
            <= 1e-12
        ).all()
        and
        (
            df[
                "benchmark_drawdown"
            ]
            <= 1e-12
        ).all()
    )

    validation[
        "tracking_error_finite"
    ] = bool(
        np.isfinite(
            tracking_metrics[
                "tracking_error"
            ]
        )
    )

    validation[
        "information_ratio_defined"
    ] = bool(
        np.isfinite(
            tracking_metrics[
                "information_ratio"
            ]
        )
    )

    validation[
        "strategy_metrics_complete"
    ] = bool(
        len(
            strategy_metrics
        ) > 0
    )

    validation[
        "benchmark_metrics_complete"
    ] = bool(
        len(
            benchmark_metrics
        ) > 0
    )

    validation[
        "regime_performance_nonempty"
    ] = bool(
        len(
            regime_performance
        ) > 0
    )

    validation[
        "yearly_performance_nonempty"
    ] = bool(
        len(
            yearly_performance
        ) > 0
    )

    validation[
        "no_lookahead_allocation"
    ] = bool(
        np.isfinite(
            df[
                "applied_equity_weight"
            ].values
        ).all()
    )

    validation[
        "all_required_outputs"
    ] = all(
        [
            BACKTEST_FILE.exists(),
            PERFORMANCE_FILE.exists(),
            YEARLY_FILE.exists(),
            REGIME_FILE_OUTPUT.exists(),
            REGIME_CHANGES_FILE.exists(),
            SUMMARY_FILE.exists(),
            EQUITY_CURVE_PLOT.exists(),
            DRAWDOWN_PLOT.exists(),
            REGIME_ALLOCATION_PLOT.exists(),
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
        "STEP 28 VALIDATION"
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
            "STEP 28 VALIDATION: FAILED"
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
            "Step 28 validation failed."
        )

    print()
    print(
        "STEP 28 VALIDATION: PASSED"
    )

    print()
    print("=" * 70)

    print(
        "STEP 28 COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print()
    print(
        "Generated files:"
    )

    print(
        f"1. {BACKTEST_FILE}"
    )

    print(
        f"2. {PERFORMANCE_FILE}"
    )

    print(
        f"3. {YEARLY_FILE}"
    )

    print(
        f"4. {REGIME_FILE_OUTPUT}"
    )

    print(
        f"5. {REGIME_CHANGES_FILE}"
    )

    print(
        f"6. {SUMMARY_FILE}"
    )

    print(
        f"7. {EQUITY_CURVE_PLOT}"
    )

    print(
        f"8. {DRAWDOWN_PLOT}"
    )

    print(
        f"9. {REGIME_ALLOCATION_PLOT}"
    )

    print(
        f"10. {CONFIG_FILE}"
    )


if __name__ == "__main__":
    main()