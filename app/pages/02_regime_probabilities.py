from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# BAYESIAN REGIME DETECTION ENGINE
# ADVANCED REGIME INTELLIGENCE
# ============================================================

st.set_page_config(
    page_title="Regime Intelligence | Bayesian Regime Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports" / "evaluation"


# ============================================================
# DATA LOADER
# ============================================================

@st.cache_data
def load_csv(file_path):

    if not file_path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(file_path)

    except Exception:
        return pd.DataFrame()


# ============================================================
# LOAD PROJECT DATA
# ============================================================

regimes = load_csv(
    DATA_DIR / "nifty50_regimes_causal.csv"
)

regime_performance = load_csv(
    REPORT_DIR / "causal_regime_performance.csv"
)

integrated_predictions = load_csv(
    DATA_DIR / "causal_regime_integrated_predictions.csv"
)


# ============================================================
# HELPER
# ============================================================

def find_column(df, candidates):

    for column in candidates:

        if column in df.columns:
            return column

    return None


# ============================================================
# DATA VALIDATION
# ============================================================

if regimes.empty:

    st.error(
        "Causal HMM dataset could not be loaded."
    )

    st.info(
        "Expected file:"
    )

    st.code(
        "data/processed/nifty50_regimes_causal.csv"
    )

    st.stop()


# ============================================================
# IDENTIFY COLUMNS
# ============================================================

date_col = find_column(
    regimes,
    [
        "date",
        "Date",
        "feature_date",
        "timestamp",
    ],
)


regime_col = find_column(
    regimes,
    [
        "regime_label",
        "regime",
        "state_label",
        "state",
        "predicted_regime",
        "causal_regime",
    ],
)


if regime_col is None:

    st.error(
        "Regime column was not found."
    )

    st.write(
        "Available columns:"
    )

    st.write(
        list(regimes.columns)
    )

    st.stop()


# ============================================================
# DATE PROCESSING
# ============================================================

if date_col:

    regimes[date_col] = pd.to_datetime(
        regimes[date_col],
        errors="coerce",
    )

    regimes = (
        regimes
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .reset_index(drop=True)
    )


# ============================================================
# REGIME SERIES
# ============================================================

regime_series = (
    regimes[regime_col]
    .astype(str)
)


states = sorted(
    regime_series.unique()
)


regime_counts = (
    regime_series
    .value_counts()
)


regime_share = (
    regime_counts
    / regime_counts.sum()
)


# ============================================================
# CURRENT REGIME
# ============================================================

current_regime = (
    regime_series.iloc[-1]
)


latest_date = (

    regimes.iloc[-1][date_col]

    if date_col

    else None
)


# ============================================================
# REGIME CHANGE COUNT
# ============================================================

regime_change_count = int(
    (
        regime_series
        != regime_series.shift()
    )
    .sum()
    - 1
)


# ============================================================
# CURRENT REGIME DURATION
# ============================================================

current_duration = 1


for i in range(
    len(regime_series) - 2,
    -1,
    -1
):

    if (
        regime_series.iloc[i]
        == current_regime
    ):

        current_duration += 1

    else:

        break


# ============================================================
# REGIME PERSISTENCE
# ============================================================

runs = []


if len(regime_series) > 0:

    start = 0

    for i in range(
        1,
        len(regime_series)
    ):

        if (
            regime_series.iloc[i]
            != regime_series.iloc[i - 1]
        ):

            runs.append(
                {
                    "Regime":
                        regime_series.iloc[start],

                    "Duration":
                        i - start,
                }
            )

            start = i


    runs.append(
        {
            "Regime":
                regime_series.iloc[start],

            "Duration":
                len(regime_series) - start,
        }
    )


runs_df = pd.DataFrame(
    runs
)


if not runs_df.empty:

    persistence = (
        runs_df
        .groupby("Regime")["Duration"]
        .agg(
            Episodes="count",
            Average_Duration="mean",
            Median_Duration="median",
            Maximum_Duration="max",
        )
        .round(2)
    )

else:

    persistence = pd.DataFrame()


# ============================================================
# TRANSITION MATRIX
# ============================================================

transition_counts = pd.DataFrame(
    0,
    index=states,
    columns=states,
    dtype=float,
)


for previous, current in zip(
    regime_series.iloc[:-1],
    regime_series.iloc[1:]
):

    transition_counts.loc[
        previous,
        current
    ] += 1


transition_probabilities = (
    transition_counts
    .div(
        transition_counts
        .sum(axis=1)
        .replace(0, np.nan),
        axis=0
    )
)


# ============================================================
# FIND PROBABILITY COLUMNS
# ============================================================

numeric_columns = [
    column
    for column in regimes.columns
    if pd.api.types.is_numeric_dtype(
        regimes[column]
    )
]


probability_columns = []


for column in numeric_columns:

    name = str(
        column
    ).lower()


    if not any(
        word in name
        for word in [
            "prob",
            "posterior",
            "filtered",
            "state_prob",
            "regime_prob",
        ]
    ):

        continue


    values = pd.to_numeric(
        regimes[column],
        errors="coerce"
    )


    if len(values.dropna()) == 0:

        continue


    if (
        values.min() >= -0.000001
        and values.max() <= 1.000001
    ):

        probability_columns.append(
            column
        )


# ============================================================
# PROBABILITY DATA
# ============================================================

probability_data = pd.DataFrame()


if probability_columns:

    for column in probability_columns:

        probability_data[column] = pd.to_numeric(
            regimes[column],
            errors="coerce"
        )


    row_sum = (
        probability_data.sum(axis=1)
    )


    valid_rows = (
        row_sum > 0
    )


    probability_data.loc[
        valid_rows
    ] = (
        probability_data.loc[
            valid_rows
        ]
        .div(
            row_sum[valid_rows],
            axis=0
        )
    )


# ============================================================
# ENTROPY
# ============================================================

entropy_data = None


if probability_columns:

    valid_probability_data = (
        probability_data
        .dropna()
    )


    if not valid_probability_data.empty:

        p = (
            valid_probability_data
            .clip(
                lower=0.000000001,
                upper=1
            )
        )


        entropy = (
            -(
                p
                * np.log(p)
            )
            .sum(axis=1)
        )


        max_entropy = np.log(
            len(probability_columns)
        )


        if max_entropy > 0:

            entropy_data = (
                entropy
                / max_entropy
            )


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    """
    <div style="
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 100%
        );
        padding: 32px;
        border-radius: 20px;
        margin-bottom: 25px;
        border: 1px solid #334155;
    ">

        <div style="
            color: #93c5fd;
            font-size: 11px;
            font-weight: 800;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            margin-bottom: 8px;
        ">
            CAUSAL HMM • SEQUENTIAL FILTERING • MARKET STATE INTELLIGENCE
        </div>

        <div style="
            color: white;
            font-size: 32px;
            font-weight: 800;
            margin-bottom: 10px;
        ">
            Regime Probability Terminal
        </div>

        <div style="
            color: #cbd5e1;
            font-size: 14px;
            line-height: 1.7;
            max-width: 950px;
        ">
            Advanced quantitative analysis of the causal Hidden Markov
            Model used to classify the NIFTY 50 market into volatility
            regimes. Analyze regime distribution, persistence,
            transition behaviour, uncertainty and regime-wise
            forecasting performance.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CURRENT STATE
# ============================================================

st.subheader(
    "Current Regime State"
)


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.metric(
        "Current Regime",
        current_regime
    )


with k2:

    st.metric(
        "Latest Date",
        (
            latest_date.strftime(
                "%d %b %Y"
            )
            if latest_date is not None
            else "N/A"
        )
    )


with k3:

    st.metric(
        "Detected States",
        len(states)
    )


with k4:

    st.metric(
        "Regime Changes",
        regime_change_count
    )


with k5:

    st.metric(
        "Current Run",
        f"{current_duration} days"
    )


# ============================================================
# REGIME DISTRIBUTION
# ============================================================

st.divider()

st.subheader(
    "1. Historical Regime Distribution"
)


left, right = st.columns(
    [1, 1.6]
)


with left:

    distribution_table = pd.DataFrame(
        {
            "Observations":
                regime_counts,

            "Market Share":
                [
                    f"{value:.2%}"
                    for value
                    in regime_share
                ],
        }
    )


    st.dataframe(
        distribution_table,
        use_container_width=True,
    )


with right:

    st.bar_chart(
        regime_counts,
        height=350,
    )


# ============================================================
# POSTERIOR PROBABILITY
# ============================================================

st.divider()

st.subheader(
    "2. Posterior Regime Probabilities"
)


if probability_columns:

    latest_probabilities = (
        probability_data
        .iloc[-1]
    )


    probability_cards = st.columns(
        min(
            len(
                probability_columns
            ),
            4,
        )
    )


    for index, column in enumerate(
        probability_columns[:4]
    ):

        with probability_cards[index]:

            st.metric(
                str(column),
                f"{latest_probabilities[column]:.2%}"
            )


    chart_probability = (
        probability_data
        .copy()
    )


    if date_col:

        chart_probability.index = (
            regimes[date_col]
        )


    st.line_chart(
        chart_probability,
        height=400,
    )


else:

    st.warning(
        "No explicit posterior probability columns "
        "were found in nifty50_regimes_causal.csv."
    )


    st.info(
        """
        The current causal HMM output stores the detected regime
        labels but does not appear to store the full posterior
        probability vector.

        The dashboard intentionally does not create artificial
        probabilities from hard regime labels.

        For genuine Low / Medium / High posterior probability
        curves, the causal HMM model must save the filtered
        probability vector during inference.
        """
    )


# ============================================================
# REGIME UNCERTAINTY
# ============================================================

st.divider()

st.subheader(
    "3. Regime Uncertainty"
)


if entropy_data is not None:

    uncertainty_df = pd.DataFrame(
        {
            "Normalized Entropy":
                entropy_data
        }
    )


    if date_col:

        uncertainty_df.index = (
            regimes.loc[
                uncertainty_df.index,
                date_col
            ]
        )


    st.line_chart(
        uncertainty_df,
        height=300,
    )


    st.caption(
        "Higher entropy means the probability mass is spread "
        "more evenly across competing regimes."
    )

else:

    st.info(
        "Entropy is available only when posterior regime "
        "probabilities are stored."
    )


# ============================================================
# TRANSITION MATRIX
# ============================================================

st.divider()

st.subheader(
    "4. Regime Transition Matrix"
)


st.caption(
    "Rows = current regime | Columns = next regime"
)


st.dataframe(
    transition_probabilities.style.format(
        "{:.2%}"
    ),
    use_container_width=True,
)


# ============================================================
# REGIME PERSISTENCE
# ============================================================

st.divider()

st.subheader(
    "5. Regime Persistence"
)


if not persistence.empty:

    st.dataframe(
        persistence,
        use_container_width=True,
    )


    st.bar_chart(
        persistence[
            ["Average_Duration"]
        ],
        height=300,
    )


else:

    st.info(
        "Regime persistence could not be calculated."
    )


# ============================================================
# REGIME TIMELINE
# ============================================================

st.divider()

st.subheader(
    "6. Causal Regime Timeline"
)


timeline = pd.DataFrame(
    {
        "Regime":
            regime_series,
    }
)


timeline["Regime_ID"] = (
    pd.Categorical(
        timeline["Regime"],
        categories=states,
        ordered=True,
    )
    .codes
)


if date_col:

    timeline.index = (
        regimes[date_col]
    )


st.line_chart(
    timeline[
        ["Regime_ID"]
    ],
    height=350,
)


st.caption(
    "The numeric state ID is only used to visualize "
    "the chronological regime sequence. It is not a "
    "continuous market score."
)


# ============================================================
# MARKET CONDITIONS
# ============================================================

st.divider()

st.subheader(
    "7. Regime vs Market Conditions"
)


return_col = find_column(
    regimes,
    [
        "daily_return",
        "return",
        "log_return",
    ],
)


volatility_col = find_column(
    regimes,
    [
        "volatility_5d",
        "volatility_20d",
        "rolling_volatility",
    ],
)


market_columns = []


if return_col:

    market_columns.append(
        return_col
    )


if volatility_col:

    market_columns.append(
        volatility_col
    )


if market_columns:

    market_context = (
        regimes[
            market_columns
        ]
        .copy()
    )


    if date_col:

        market_context.index = (
            regimes[date_col]
        )


    st.line_chart(
        market_context,
        height=350,
    )


else:

    st.info(
        "Market return / volatility columns "
        "were not found."
    )


# ============================================================
# REGIME-WISE MODEL PERFORMANCE
# ============================================================

if not regime_performance.empty:

    st.divider()

    st.subheader(
        "8. Forecasting Performance by Regime"
    )


    st.dataframe(
        regime_performance,
        use_container_width=True,
        hide_index=True,
    )


    if (
        "regime"
        in regime_performance.columns
        and
        "model"
        in regime_performance.columns
        and
        "accuracy"
        in regime_performance.columns
    ):

        performance_chart = (
            regime_performance
            .pivot(
                index="regime",
                columns="model",
                values="accuracy",
            )
        )


        st.bar_chart(
            performance_chart,
            height=350,
        )


# ============================================================
# QUANTITATIVE INTERPRETATION
# ============================================================

st.divider()

st.subheader(
    "9. Quantitative Interpretation"
)


current_lower = (
    current_regime
    .lower()
)


if "high" in current_lower:

    st.warning(
        """
        **High-Volatility Regime**

        The causal HMM currently classifies the latest observation
        as a high-volatility state.

        This indicates a historically more variable market
        environment. Forecast uncertainty should be evaluated
        alongside the regime rather than treating the regime as
        a direct directional signal.
        """
    )


elif "medium" in current_lower:

    st.info(
        """
        **Medium-Volatility Regime**

        The latest observation belongs to the medium-volatility
        state learned by the causal HMM.

        This represents an intermediate historical volatility
        environment between the low- and high-volatility states.
        """
    )


elif "low" in current_lower:

    st.success(
        """
        **Low-Volatility Regime**

        The latest observation belongs to the low-volatility
        state learned by the causal HMM.

        This represents a historically calmer market environment
        according to the model's learned state structure.
        """
    )


else:

    st.info(
        f"""
        **Detected State: {current_regime}**

        The causal HMM assigned the latest observation to this
        model-defined historical state.

        This classification is not a guaranteed prediction of
        future market direction.
        """
    )


# ============================================================
# MODEL METHODOLOGY
# ============================================================

st.divider()

with st.expander(
    "🔬 Causal HMM Methodology"
):

    st.markdown(
        """
        ### Causal Regime Detection

        The project uses a Hidden Markov Model to classify market
        observations into discrete volatility states.

        **Pipeline**

        1. Historical market data is prepared chronologically.
        2. The training period is separated from the test period.
        3. Model parameters are estimated using the training data.
        4. Features are standardized using training information.
        5. Observations are processed sequentially using forward filtering.
        6. A causal regime state is assigned to each observation.
        7. The resulting sequence is validated for future-information leakage.
        8. Regimes are aligned with the forecasting models.

        ### Three regime states

        • Low Volatility  
        • Medium Volatility  
        • High Volatility

        ### Important distinction

        A regime classification describes the estimated market state.
        It is not itself a prediction that the next trading day will
        rise or fall.

        Posterior probabilities are required to quantify uncertainty
        between competing HMM states.
        """
    )


# ============================================================
# DATA QUALITY
# ============================================================

st.divider()

st.subheader(
    "10. Regime Engine Data Quality"
)


q1, q2, q3, q4 = st.columns(4)


with q1:

    st.metric(
        "Rows",
        f"{len(regimes):,}"
    )


with q2:

    st.metric(
        "Columns",
        f"{len(regimes.columns):,}"
    )


with q3:

    st.metric(
        "Missing Values",
        int(
            regimes
            .isna()
            .sum()
            .sum()
        )
    )


with q4:

    st.metric(
        "Duplicate Rows",
        int(
            regimes
            .duplicated()
            .sum()
        )
    )


# ============================================================
# DATA EXPLORER
# ============================================================

with st.expander(
    "🔎 Explore Causal HMM Dataset"
):

    st.write(
        "Available columns:"
    )

    st.code(
        ", ".join(
            map(
                str,
                regimes.columns
            )
        )
    )


    st.dataframe(
        regimes.tail(150),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# DOWNLOAD
# ============================================================

st.download_button(
    label="⬇️ Download Causal Regime Dataset",
    data=(
        regimes
        .to_csv(index=False)
        .encode("utf-8")
    ),
    file_name="causal_regime_analysis.csv",
    mime="text/csv",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bayesian Regime Detection Engine • "
    "Advanced Regime Intelligence • "
    "Historical Quantitative Research Only • "
    "Not Financial Advice"
)