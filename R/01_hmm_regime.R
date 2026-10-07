# ============================================================
# STEP 29C - R HMM REGIME VALIDATION
# Bayesian Regime Detection Engine
# ============================================================

PROJECT_ROOT <- "C:/BAYESIAN_REGIME_ENGINE"

INPUT_FILE <- file.path(
  PROJECT_ROOT,
  "data",
  "processed",
  "nifty50_features.csv"
)

PYTHON_REGIME_FILE <- file.path(
  PROJECT_ROOT,
  "data",
  "processed",
  "nifty50_regimes_causal.csv"
)

OUTPUT_DIR <- file.path(
  PROJECT_ROOT,
  "reports",
  "r_validation"
)

OUTPUT_FILE <- file.path(
  OUTPUT_DIR,
  "r_hmm_regime_validation.csv"
)

SUMMARY_FILE <- file.path(
  OUTPUT_DIR,
  "r_hmm_summary.txt"
)

PLOT_FILE <- file.path(
  OUTPUT_DIR,
  "r_hmm_regimes.png"
)

COMPARISON_FILE <- file.path(
  OUTPUT_DIR,
  "r_python_regime_comparison.csv"
)

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

# ============================================================
# 1. Packages
# ============================================================

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(depmixS4)
})

cat("\n============================================\n")
cat("STEP 29C - R HMM REGIME VALIDATION\n")
cat("============================================\n")

# ============================================================
# 2. Load data
# ============================================================

if (!file.exists(INPUT_FILE)) {
  stop(
    paste("Input file not found:", INPUT_FILE)
  )
}

dt <- fread(INPUT_FILE)

cat(
  "Loaded feature dataset:",
  nrow(dt),
  "rows x",
  ncol(dt),
  "columns\n"
)

required_columns <- c(
  "date",
  "daily_return"
)

missing_columns <- setdiff(
  required_columns,
  names(dt)
)

if (length(missing_columns) > 0) {
  stop(
    paste(
      "Missing required columns:",
      paste(missing_columns, collapse = ", ")
    )
  )
}

# ============================================================
# 3. Prepare data
# ============================================================

dt[, date := as.IDate(date)]

setorder(dt, date)

dt[, daily_return := as.numeric(daily_return)]

dt <- dt[
  !is.na(daily_return)
]

dt <- dt[
  is.finite(daily_return)
]

if (anyDuplicated(dt$date) > 0) {
  stop("Duplicate dates detected.")
}

cat(
  "Usable observations:",
  nrow(dt),
  "\n"
)

# ============================================================
# 4. Chronological split
# ============================================================

n <- nrow(dt)

train_n <- floor(
  0.80 * n
)

dt[, data_split := "test"]

dt[1:train_n, data_split := "train"]

cat(
  "Training observations:",
  train_n,
  "\n"
)

cat(
  "Test observations:",
  n - train_n,
  "\n"
)

# ============================================================
# 5. Train-only standardization
# ============================================================

train_returns <- dt[
  data_split == "train",
  daily_return
]

train_mean <- mean(
  train_returns
)

train_sd <- sd(
  train_returns
)

if (!is.finite(train_sd) || train_sd <= 0) {
  stop("Invalid training standard deviation.")
}

dt[
  ,
  return_scaled :=
    (daily_return - train_mean) / train_sd
]

# ============================================================
# 6. Fit Gaussian HMM
# ============================================================

cat("\n")
cat("Fitting 3-state Gaussian HMM...\n")

set.seed(42)

hmm_model <- depmix(
  response = return_scaled ~ 1,
  data = dt,
  nstates = 3,
  family = gaussian()
)

hmm_fit <- fit(
  hmm_model,
  verbose = FALSE
)

cat("HMM fitting completed.\n")

# ============================================================
# 7. Extract posterior/filtering information
# ============================================================

posterior_data <- posterior(
  hmm_fit,
  type = "filtering"
)

cat("\nPosterior object information:\n")
cat(
  "Rows:",
  nrow(posterior_data),
  "\n"
)

cat(
  "Columns:",
  ncol(posterior_data),
  "\n"
)

cat(
  "Column names:",
  paste(
    names(posterior_data),
    collapse = ", "
  ),
  "\n"
)

# ============================================================
# 8. Robustly extract state probabilities
# ============================================================

# depmixS4 versions can expose posterior state
# probabilities differently. First look for numeric
# columns containing probability-like values.

numeric_columns <- names(
  posterior_data
)[
  sapply(
    posterior_data,
    is.numeric
  )
]

candidate_columns <- c()

for (col in numeric_columns) {

  values <- posterior_data[[col]]

  if (
    length(values) == nrow(posterior_data) &&
    all(
      is.finite(values)
    ) &&
    all(
      values >= -1e-8 &
      values <= 1 + 1e-8
    )
  ) {

    candidate_columns <- c(
      candidate_columns,
      col
    )
  }
}

# Remove obvious non-probability columns.
candidate_columns <- candidate_columns[
  !candidate_columns %in%
    c(
      "logLik",
      "response",
      "return_scaled",
      "daily_return",
      "date",
      "time",
      "state"
    )
]

# If exactly three probability columns are found,
# use them.
if (length(candidate_columns) >= 3) {

  # Prefer columns whose names contain "posterior"
  posterior_named <- candidate_columns[
    grepl(
      "posterior|state|S[0-9]",
      candidate_columns,
      ignore.case = TRUE
    )
  ]

  if (length(posterior_named) >= 3) {
    state_columns <- posterior_named[1:3]
  } else {
    state_columns <- candidate_columns[1:3]
  }

} else {

  # ========================================================
  # Fallback:
  # Use Viterbi state sequence when filtering
  # probabilities are not exposed by the installed
  # depmixS4 version.
  # ========================================================

  cat("\n")
  cat(
    "Filtering probabilities were not exposed as columns.\n"
  )

  cat(
    "Using Viterbi state sequence as R HMM state validation.\n"
  )

  posterior_data <- posterior(
    hmm_fit
  )

  if (!"state" %in% names(posterior_data)) {

    # Try viterbi()
    viterbi_result <- viterbi(
      hmm_fit
    )

    if ("state" %in% names(viterbi_result)) {

      posterior_data$state <-
        viterbi_result$state

    } else {

      stop(
        paste(
          "Unable to extract HMM state sequence.",
          "Available columns:",
          paste(
            names(posterior_data),
            collapse = ", "
          )
        )
      )
    }
  }

  r_regime <- as.integer(
    posterior_data$state
  )

  # Create one-hot probabilities.
  prob_matrix <- matrix(
    0,
    nrow = length(r_regime),
    ncol = 3
  )

  for (i in seq_along(r_regime)) {

    state <- r_regime[i]

    if (
      is.finite(state) &&
      state >= 1 &&
      state <= 3
    ) {

      prob_matrix[
        i,
        state
      ] <- 1
    }
  }

  colnames(prob_matrix) <- c(
    "r_prob_state_1",
    "r_prob_state_2",
    "r_prob_state_3"
  )

}

# ============================================================
# 9. If probability columns were found
# ============================================================

if (exists("state_columns")) {

  prob_matrix <- as.matrix(
    posterior_data[
      ,
      state_columns,
      drop = FALSE
    ]
  )

  prob_matrix <- apply(
    prob_matrix,
    2,
    as.numeric
  )

  prob_matrix <- as.matrix(
    prob_matrix
  )

  colnames(prob_matrix) <- c(
    "r_prob_state_1",
    "r_prob_state_2",
    "r_prob_state_3"
  )

  r_regime <- max.col(
    prob_matrix,
    ties.method = "first"
  )
}

# ============================================================
# 10. Ensure correct row count
# ============================================================

if (
  nrow(prob_matrix) != nrow(dt)
) {

  stop(
    paste(
      "Probability/state rows do not match input rows.",
      "Input:",
      nrow(dt),
      "HMM:",
      nrow(prob_matrix)
    )
  )
}

# ============================================================
# 11. State volatility statistics
# ============================================================

state_mean <- numeric(3)
state_sd <- numeric(3)

for (s in 1:3) {

  idx <- r_regime == s

  if (sum(idx) > 1) {

    state_mean[s] <- mean(
      dt$daily_return[idx]
    )

    state_sd[s] <- sd(
      dt$daily_return[idx]
    )

  } else {

    state_mean[s] <- NA_real_
    state_sd[s] <- NA_real_
  }
}

# Rank states by volatility
vol_rank <- rank(
  state_sd,
  ties.method = "first"
)

regime_label <- rep(
  NA_character_,
  length(r_regime)
)

regime_label[
  vol_rank[r_regime] == 1
] <- "Low Volatility"

regime_label[
  vol_rank[r_regime] == 2
] <- "Medium Volatility"

regime_label[
  vol_rank[r_regime] == 3
] <- "High Volatility"

# ============================================================
# 12. Build final R result
# ============================================================

result <- data.table(
  date = dt$date,

  daily_return =
    dt$daily_return,

  data_split =
    dt$data_split,

  r_state =
    r_regime,

  r_prob_state_1 =
    prob_matrix[, 1],

  r_prob_state_2 =
    prob_matrix[, 2],

  r_prob_state_3 =
    prob_matrix[, 3],

  r_regime_label =
    regime_label
)

# ============================================================
# 13. Probability validation
# ============================================================

prob_sum <- rowSums(
  result[
    ,
    .(
      r_prob_state_1,
      r_prob_state_2,
      r_prob_state_3
    )
  ]
)

probability_valid <- all(
  is.finite(prob_sum) &
  abs(prob_sum - 1) < 1e-6
)

cat(
  "\nProbability validation:",
  probability_valid,
  "\n"
)

# ============================================================
# 14. R regime statistics
# ============================================================

state_summary <- result[
  ,
  .(
    observations = .N,
    mean_return =
      mean(daily_return),
    volatility =
      sd(daily_return)
  ),
  by = r_regime_label
]

cat("\nR regime statistics:\n")
print(state_summary)

# ============================================================
# 15. Python comparison
# ============================================================

python_comparison_available <-
  file.exists(
    PYTHON_REGIME_FILE
  )

python_match_rate <- NA_real_

if (python_comparison_available) {

  cat("\n")
  cat(
    "Loading Python causal HMM output...\n"
  )

  py <- fread(
    PYTHON_REGIME_FILE
  )

  if (
    "date" %in% names(py) &&
    "regime_label" %in% names(py)
  ) {

    py[, date := as.IDate(date)]

    comparison <- merge(
      result[
        ,
        .(
          date,
          r_regime_label
        )
      ],
      py[
        ,
        .(
          date,
          python_regime_label =
            regime_label
        )
      ],
      by = "date",
      all = FALSE
    )

    if (nrow(comparison) > 0) {

      comparison[
        ,
        exact_label_match :=
          r_regime_label ==
          python_regime_label
      ]

      python_match_rate <-
        mean(
          comparison$exact_label_match
        )

      fwrite(
        comparison,
        COMPARISON_FILE
      )

      cat(
        "Python/R common dates:",
        nrow(comparison),
        "\n"
      )

      cat(
        "Exact regime-label match rate:",
        round(
          python_match_rate,
          4
        ),
        "\n"
      )
    }
  }
}

# ============================================================
# 16. Save result
# ============================================================

fwrite(
  result,
  OUTPUT_FILE
)

# ============================================================
# 17. Plot
# ============================================================

p <- ggplot(
  result,
  aes(
    x = date,
    y = daily_return,
    color = r_regime_label
  )
) +
  geom_point(
    size = 0.6,
    alpha = 0.65
  ) +
  labs(
    title =
      "R HMM Market Regime Validation",
    subtitle =
      "NIFTY 50 daily returns",
    x = "Date",
    y = "Daily Return",
    color = "R Regime"
  ) +
  theme_minimal()

ggsave(
  PLOT_FILE,
  p,
  width = 12,
  height = 6,
  dpi = 150
)

# ============================================================
# 18. Summary
# ============================================================

summary_lines <- c(

  "STEP 29C - R HMM REGIME VALIDATION",
  "============================================",

  paste(
    "Input rows:",
    nrow(dt)
  ),

  paste(
    "Training rows:",
    train_n
  ),

  paste(
    "Test rows:",
    nrow(dt) - train_n
  ),

  paste(
    "Training mean return:",
    format(
      train_mean,
      scientific = TRUE
    )
  ),

  paste(
    "Training return SD:",
    format(
      train_sd,
      scientific = TRUE
    )
  ),

  paste(
    "Probability validation:",
    probability_valid
  ),

  paste(
    "Python comparison available:",
    python_comparison_available
  ),

  paste(
    "Python/R exact label match rate:",
    ifelse(
      is.na(python_match_rate),
      "NA",
      round(
        python_match_rate,
        4
      )
    )
  ),

  "",
  "R state statistics:",
  capture.output(
    print(state_summary)
  ),

  "",
  "Output files:",
  OUTPUT_FILE,
  PLOT_FILE
)

writeLines(
  summary_lines,
  SUMMARY_FILE
)

# ============================================================
# 19. Final validation
# ============================================================

validation <- c(

  input_exists =
    file.exists(INPUT_FILE),

  output_exists =
    file.exists(OUTPUT_FILE),

  summary_exists =
    file.exists(SUMMARY_FILE),

  plot_exists =
    file.exists(PLOT_FILE),

  rows_positive =
    nrow(result) > 0,

  dates_sorted =
    all(
      diff(
        as.numeric(result$date)
      ) >= 0
    ),

  duplicate_dates =
    anyDuplicated(
      result$date
    ) == 0,

  probabilities_valid =
    probability_valid,

  regimes_valid =
    all(
      result$r_regime_label %in%
        c(
          "Low Volatility",
          "Medium Volatility",
          "High Volatility"
        )
    )
)

cat("\n")
cat("STEP 29C VALIDATION\n")
print(validation)

all_passed <- all(
  validation
)

cat("\n")

if (all_passed) {

  cat(
    "STEP 29C VALIDATION: PASSED\n"
  )

  cat(
    "STEP 29C COMPLETED SUCCESSFULLY\n"
  )

} else {

  cat(
    "STEP 29C VALIDATION: FAILED\n"
  )

  quit(
    status = 1
  )
}