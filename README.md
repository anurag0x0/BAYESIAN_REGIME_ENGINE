# Bayesian Regime Detection Engine for Equity Direction Forecasting

## Research-Oriented Machine Learning System for Indian Equity Markets

The **Bayesian Regime Detection Engine** is a research-oriented quantitative forecasting system designed to identify latent market regimes and evaluate next-day equity market direction under uncertainty.

The system combines:

- Bayesian regression
- Hidden Markov Models
- Causal regime filtering
- Sequential Monte Carlo
- Particle filtering
- Time-series foundation model benchmarking
- Ensemble forecasting
- Walk-forward evaluation
- Statistical significance testing
- Bootstrap confidence intervals
- Regime-wise performance analysis
- Forecast uncertainty analysis
- Leakage and robustness auditing
- Interactive Streamlit analytics dashboard

The primary market used in this project is the **NIFTY 50 index**.

---

# 1. Project Objective

Financial markets are non-stationary.

A forecasting model that performs well during a low-volatility period may behave differently during:

- High-volatility periods
- Market shocks
- Transitional regimes
- Momentum periods
- Post-crisis recovery periods

Therefore, instead of assuming that the market follows a single stable data-generating process, this project attempts to answer two questions:

### Question 1

> What latent market regime is the market currently experiencing?

### Question 2

> Given the current market regime and available information, how confidently can we forecast the next day's market direction?

The project therefore treats **market regime + directional forecasting + uncertainty** as interconnected components.

---

# 2. Research Architecture

```text
                    ┌──────────────────────┐
                    │   NIFTY 50 Data      │
                    │      ^NSEI           │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Data Cleaning         │
                    │ Validation            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Feature Engineering  │
                    │ Returns              │
                    │ Volatility           │
                    │ Momentum             │
                    │ OHLC Features        │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       ┌────────────┐   ┌──────────────┐  ┌──────────────┐
       │ Bayesian   │   │ Causal HMM   │  │ SMC /        │
       │ Forecast   │   │ Regime Model │  │ Particle     │
       │            │   │              │  │ Filter       │
       └─────┬──────┘   └──────┬───────┘  └──────┬───────┘
             │                 │                 │
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Chronos Benchmark    │
                    │ Time-Series Model     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Ensemble Forecasting │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Regime Integration   │
                    └──────────┬───────────┘
                               │
                               ▼
              ┌────────────────┴────────────────┐
              │                                 │
              ▼                                 ▼
    ┌────────────────────┐          ┌────────────────────┐
    │ Walk-Forward        │          │ Statistical        │
    │ Evaluation          │          │ Robustness         │
    └──────────┬─────────┘          └──────────┬─────────┘
               │                               │
               └──────────────┬────────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────────────────┘