# ============================================================
# STEP 29D
# BAYESIAN REGIME VALIDATION IN R
#
# Model:
# Bayesian Logistic Regression using rstanarm
#
# Purpose:
# Independent R validation layer for Bayesian regime modelling
#
# Target:
# High-volatility regime
#
# IMPORTANT:
# - Chronological 80/20 split
# - Training-only volatility threshold
# - Training-only feature standardization
# - Bayesian posterior probability
# - Posterior uncertainty
# - Honest handling of undefined precision/F1
# ============================================================


# ============================================================
# 1. PACKAGES
# ============================================================

required_packages <- c(
  "data.table",
  "ggplot2",
  "rstanarm"
)

for (pkg in required_packages) {

  if (!requireNamespace(pkg, quietly = TRUE)) {

    install.packages(
      pkg,
      repos = "https://cloud.r-project.org"
    )
  }
}

library(data.table)
library(ggplot2)
library(rstanarm)

options(mc.cores = 1)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

PROJECT_ROOT <- "C:/BAYESIAN_REGIME_ENGINE"

INPUT_FILE <- file.path(
  PROJECT_ROOT,
  "data",
  "processed",
  "nifty50_features.csv"
)

OUTPUT_DIR <- file.path(
  PROJECT_ROOT,
  "reports",
  "r_validation"
)

MODEL_DIR <- file.path(
  PROJECT_ROOT,
  "models",
  "r_bayesian"
)

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

dir.create(
  MODEL_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)


# Output files

OUTPUT_FILE <- file.path(
  OUTPUT_DIR,
  "r_bayesian_regime.csv"
)

SUMMARY_FILE <- file.path(
  OUTPUT_DIR,
  "r_bayesian_summary.txt"
)

DIAGNOSTICS_FILE <- file.path(
  OUTPUT_DIR,
  "r_bayesian_diagnostics.txt"
)

CALIBRATION_FILE <- file.path(
  OUTPUT_DIR,
  "r_bayesian_calibration.csv"
)

METRICS_FILE <- file.path(
  OUTPUT_DIR,
  "r_bayesian_metrics.csv"
)

MODEL_FILE <- file.path(
  MODEL_DIR,
  "r_bayesian_regime_model.rds"
)


# ============================================================
# 3. HEADER
# ============================================================

cat("\n")
cat("============================================\n")
cat("STEP 29D - R BAYESIAN REGIME MODEL\n")
cat("============================================\n")
cat("\n")


# ============================================================
# 4. LOAD DATA
# ============================================================

if (!file.exists(INPUT_FILE)) {

  stop(
    paste(
      "Input file not found:",
      INPUT_FILE
    )
  )
}

dt <- fread(
  INPUT_FILE
)

cat(
  "Loaded feature dataset:",
  nrow(dt),
  "rows x",
  ncol(dt),
  "columns\n"
)


# ============================================================
# 5. REQUIRED COLUMNS
# ============================================================

required_columns <- c(
  "date",
  "daily_return",
  "volatility_20d"
)

missing_columns <- setdiff(
  required_columns,
  names(dt)
)

if (length(missing_columns) > 0) {

  stop(
    paste(
      "Missing required columns:",
      paste(
        missing_columns,
        collapse = ", "
      )
    )
  )
}


# ============================================================
# 6. DATE CLEANING
# ============================================================

dt[, date := as.IDate(date)]

setorder(
  dt,
  date
)


# ============================================================
# 7. REMOVE INVALID OBSERVATIONS
# ============================================================

dt <- dt[
  !is.na(date) &
  !is.na(daily_return) &
  !is.na(volatility_20d)
]

cat(
  "Usable observations:",
  nrow(dt),
  "\n"
)


# ============================================================
# 8. CHRONOLOGICAL TRAIN / TEST SPLIT
# ============================================================

n <- nrow(dt)

train_n <- floor(
  n * 0.80
)

train <- dt[
  1:train_n
]

test <- dt[
  (train_n + 1):n
]

cat(
  "Training rows:",
  nrow(train),
  "\n"
)

cat(
  "Test rows:",
  nrow(test),
  "\n"
)


# ============================================================
# 9. TRAINING-ONLY HIGH VOLATILITY THRESHOLD
# ============================================================

high_vol_threshold <- quantile(
  train$volatility_20d,
  probs = 0.75,
  na.rm = TRUE
)

cat(
  "High-volatility threshold:",
  round(
    high_vol_threshold,
    9
  ),
  "\n"
)


# ============================================================
# 10. CREATE TARGET
# ============================================================

train[
  ,
  high_volatility :=
    as.integer(
      volatility_20d >= high_vol_threshold
    )
]

test[
  ,
  high_volatility :=
    as.integer(
      volatility_20d >= high_vol_threshold
    )
]


# ============================================================
# 11. TRAINING-ONLY STANDARDIZATION
# ============================================================

return_mean <- mean(
  train$daily_return,
  na.rm = TRUE
)

return_sd <- sd(
  train$daily_return,
  na.rm = TRUE
)

if (
  !is.finite(return_sd) ||
  return_sd == 0
) {

  stop(
    "Training standard deviation is invalid."
  )
}

train[
  ,
  daily_return_scaled :=
    (daily_return - return_mean) /
    return_sd
]

test[
  ,
  daily_return_scaled :=
    (daily_return - return_mean) /
    return_sd
]


# ============================================================
# 12. BAYESIAN LOGISTIC REGRESSION
# ============================================================

cat("\n")
cat("--------------------------------------------\n")
cat("Starting Bayesian logistic regression...\n")
cat("--------------------------------------------\n")
cat("\n")


bayesian_model <- stan_glm(

  high_volatility ~ daily_return_scaled,

  data = train,

  family = binomial(
    link = "logit"
  ),

  prior = normal(
    location = 0,
    scale = 1,
    autoscale = FALSE
  ),

  prior_intercept = normal(
    location = 0,
    scale = 1,
    autoscale = FALSE
  ),

  chains = 2,

  iter = 1000,

  warmup = 500,

  cores = 1,

  seed = 42,

  refresh = 100
)


cat(
  "\nBayesian model fitting completed.\n"
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

saveRDS(
  bayesian_model,
  MODEL_FILE
)

cat(
  "Model saved:",
  MODEL_FILE,
  "\n"
)


# ============================================================
# 14. POSTERIOR DIAGNOSTICS
# ============================================================

cat("\n")
cat("--------------------------------------------\n")
cat("Posterior diagnostics\n")
cat("--------------------------------------------\n")


model_summary <- capture.output(
  print(
    summary(
      bayesian_model
    )
  )
)

writeLines(
  model_summary,
  DIAGNOSTICS_FILE
)


# ============================================================
# 15. POSTERIOR DRAWS
# ============================================================

posterior_draws <- as.data.table(
  as.matrix(
    bayesian_model
  )
)

coefficient_names <- names(
  posterior_draws
)

intercept_name <- "(Intercept)"

slope_name <- "daily_return_scaled"


if (
  !(intercept_name %in% coefficient_names) ||
  !(slope_name %in% coefficient_names)
) {

  stop(
    paste(
      "Could not find expected posterior coefficients.",
      "\nAvailable coefficients:",
      paste(
        coefficient_names,
        collapse = ", "
      )
    )
  )
}


intercept_draws <- posterior_draws[
  ,
  get(intercept_name)
]

slope_draws <- posterior_draws[
  ,
  get(slope_name)
]


# ============================================================
# 16. POSTERIOR PREDICTIVE PROBABILITIES
# ============================================================

x_test <- test$daily_return_scaled


probability_matrix <- outer(

  x_test,

  slope_draws,

  function(x, beta) {

    plogis(
      intercept_draws +
      beta * x
    )
  }
)


posterior_probability <- rowMeans(
  probability_matrix,
  na.rm = TRUE
)


posterior_sd <- apply(
  probability_matrix,
  1,
  sd,
  na.rm = TRUE
)


# ============================================================
# 17. CLASS PREDICTIONS
# ============================================================

predicted_class <- as.integer(
  posterior_probability >= 0.50
)

actual_class <- test$high_volatility


# ============================================================
# 18. CORE PROBABILISTIC METRICS
# ============================================================

accuracy <- mean(
  predicted_class == actual_class
)


brier_score <- mean(
  (
    posterior_probability -
    actual_class
  )^2
)


log_loss <- -mean(

  actual_class *
    log(
      pmax(
        posterior_probability,
        1e-15
      )
    )

  +

  (1 - actual_class) *
    log(
      pmax(
        1 - posterior_probability,
        1e-15
      )
    )
)


mean_uncertainty <- mean(
  posterior_sd,
  na.rm = TRUE
)


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

tp <- sum(
  predicted_class == 1 &
  actual_class == 1
)

tn <- sum(
  predicted_class == 0 &
  actual_class == 0
)

fp <- sum(
  predicted_class == 1 &
  actual_class == 0
)

fn <- sum(
  predicted_class == 0 &
  actual_class == 1
)


# ============================================================
# 20. CLASSIFICATION METRICS
# ============================================================

# Precision
#
# If there are zero positive predictions,
# precision is mathematically undefined.
#
# We DO NOT replace it with fake zero.

precision <- if (
  (tp + fp) > 0
) {

  tp / (tp + fp)

} else {

  NA_real_
}


# Recall

recall <- if (
  (tp + fn) > 0
) {

  tp / (tp + fn)

} else {

  NA_real_
}


# F1

f1 <- if (
  is.finite(precision) &&
  is.finite(recall) &&
  (precision + recall) > 0
) {

  2 * precision * recall /
    (precision + recall)

} else {

  NA_real_
}


# ============================================================
# 21. PREDICTION CLASS COUNTS
# ============================================================

predicted_positive_count <- sum(
  predicted_class == 1
)

predicted_negative_count <- sum(
  predicted_class == 0
)

actual_positive_count <- sum(
  actual_class == 1
)

actual_negative_count <- sum(
  actual_class == 0
)


# ============================================================
# 22. RESULT DATASET
# ============================================================

result <- data.table(

  date = test$date,

  daily_return = test$daily_return,

  volatility_20d =
    test$volatility_20d,

  actual_high_volatility =
    actual_class,

  posterior_probability =
    posterior_probability,

  posterior_uncertainty =
    posterior_sd,

  predicted_high_volatility =
    predicted_class
)


# ============================================================
# 23. SAVE PREDICTIONS
# ============================================================

fwrite(
  result,
  OUTPUT_FILE
)

cat(
  "\nPrediction output saved:",
  OUTPUT_FILE,
  "\n"
)


# ============================================================
# 24. CALIBRATION TABLE
# ============================================================

result[
  ,
  probability_bin :=
    cut(
      posterior_probability,
      breaks = seq(
        0,
        1,
        by = 0.1
      ),
      include.lowest = TRUE
    )
]


calibration <- result[
  ,
  .(
    observations = .N,

    mean_predicted_probability =
      mean(
        posterior_probability,
        na.rm = TRUE
      ),

    observed_frequency =
      mean(
        actual_high_volatility,
        na.rm = TRUE
      )
  ),
  by = probability_bin
]


fwrite(
  calibration,
  CALIBRATION_FILE
)

cat(
  "Calibration table saved:",
  CALIBRATION_FILE,
  "\n"
)


# ============================================================
# 25. METRICS TABLE
# ============================================================

metrics_table <- data.table(

  metric = c(

    "accuracy",

    "brier_score",

    "log_loss",

    "precision",

    "recall",

    "f1",

    "mean_posterior_uncertainty",

    "true_negative",

    "false_positive",

    "false_negative",

    "true_positive",

    "predicted_positive_count",

    "predicted_negative_count",

    "actual_positive_count",

    "actual_negative_count"
  ),

  value = c(

    accuracy,

    brier_score,

    log_loss,

    precision,

    recall,

    f1,

    mean_uncertainty,

    tn,

    fp,

    fn,

    tp,

    predicted_positive_count,

    predicted_negative_count,

    actual_positive_count,

    actual_negative_count
  )
)


fwrite(
  metrics_table,
  METRICS_FILE
)


# ============================================================
# 26. SUMMARY REPORT
# ============================================================

precision_text <- if (
  is.finite(precision)
) {

  as.character(
    round(
      precision,
      6
    )
  )

} else {

  "NA - no positive predictions"
}


f1_text <- if (
  is.finite(f1)
) {

  as.character(
    round(
      f1,
      6
    )
  )

} else {

  "NA - no positive predictions"
}


summary_lines <- c(

  "============================================================",

  "R BAYESIAN REGIME MODEL SUMMARY",

  "============================================================",

  "",

  paste(
    "Input rows:",
    nrow(dt)
  ),

  paste(
    "Training rows:",
    nrow(train)
  ),

  paste(
    "Test rows:",
    nrow(test)
  ),

  paste(
    "High volatility threshold:",
    format(
      high_vol_threshold,
      digits = 10
    )
  ),

  "",

  "MODEL:",

  "Bayesian Logistic Regression using rstanarm",

  "",

  "TARGET:",

  "High volatility regime",

  "",

  "PREDICTOR:",

  "Training-standardized daily return",

  "",

  "METRICS:",

  paste(
    "Accuracy:",
    round(
      accuracy,
      6
    )
  ),

  paste(
    "Brier Score:",
    round(
      brier_score,
      6
    )
  ),

  paste(
    "Log Loss:",
    round(
      log_loss,
      6
    )
  ),

  paste(
    "Precision:",
    precision_text
  ),

  paste(
    "Recall:",
    round(
      recall,
      6
    )
  ),

  paste(
    "F1 Score:",
    f1_text
  ),

  paste(
    "Mean posterior uncertainty:",
    round(
      mean_uncertainty,
      6
    )
  ),

  "",

  "PREDICTION DISTRIBUTION:",

  paste(
    "Predicted positive:",
    predicted_positive_count
  ),

  paste(
    "Predicted negative:",
    predicted_negative_count
  ),

  paste(
    "Actual positive:",
    actual_positive_count
  ),

  paste(
    "Actual negative:",
    actual_negative_count
  ),

  "",

  "CONFUSION MATRIX:",

  paste(
    "True Negative:",
    tn
  ),

  paste(
    "False Positive:",
    fp
  ),

  paste(
    "False Negative:",
    fn
  ),

  paste(
    "True Positive:",
    tp
  ),

  "",

  "POSTERIOR MODEL:",

  paste(
    "Posterior draws:",
    nrow(posterior_draws)
  ),

  "",

  "OUTPUT FILES:",

  OUTPUT_FILE,

  CALIBRATION_FILE,

  METRICS_FILE,

  DIAGNOSTICS_FILE,

  MODEL_FILE,

  "",

  "NOTE:",

  "Precision and F1 are undefined when the model produces zero positive predictions.",

  "They are intentionally reported as NA rather than being artificially converted to zero.",

  "",

  "============================================================"
)


writeLines(
  summary_lines,
  SUMMARY_FILE
)


# ============================================================
# 27. CONSOLE PERFORMANCE
# ============================================================

cat("\n")
cat("============================================\n")
cat("MODEL PERFORMANCE\n")
cat("============================================\n")


cat(
  "Accuracy:",
  round(
    accuracy,
    4
  ),
  "\n"
)


cat(
  "Brier Score:",
  round(
    brier_score,
    4
  ),
  "\n"
)


cat(
  "Log Loss:",
  round(
    log_loss,
    4
  ),
  "\n"
)


cat(
  "Precision:",
  ifelse(
    is.finite(precision),
    round(
      precision,
      4
    ),
    "NA"
  ),
  "\n"
)


cat(
  "Recall:",
  round(
    recall,
    4
  ),
  "\n"
)


cat(
  "F1:",
  ifelse(
    is.finite(f1),
    round(
      f1,
      4
    ),
    "NA"
  ),
  "\n"
)


cat(
  "Mean posterior uncertainty:",
  round(
    mean_uncertainty,
    4
  ),
  "\n"
)


cat(
  "Predicted positive observations:",
  predicted_positive_count,
  "\n"
)


cat(
  "Actual positive observations:",
  actual_positive_count,
  "\n"
)


# ============================================================
# 28. VALIDATION
# ============================================================

cat("\n")
cat("============================================\n")
cat("STEP 29D VALIDATION\n")
cat("============================================\n")


input_exists <- file.exists(
  INPUT_FILE
)

output_exists <- file.exists(
  OUTPUT_FILE
)

summary_exists <- file.exists(
  SUMMARY_FILE
)

diagnostics_exists <- file.exists(
  DIAGNOSTICS_FILE
)

model_exists <- file.exists(
  MODEL_FILE
)

calibration_exists <- file.exists(
  CALIBRATION_FILE
)

metrics_file_exists <- file.exists(
  METRICS_FILE
)

rows_positive <- nrow(
  result
) > 0


probabilities_valid <- all(

  is.finite(
    result$posterior_probability
  ) &

  result$posterior_probability >= 0 &

  result$posterior_probability <= 1
)


uncertainty_valid <- all(
  is.finite(
    result$posterior_uncertainty
  )
)


dates_sorted <- all(
  diff(
    as.numeric(
      result$date
    )
  ) >= 0
)


duplicate_dates <- anyDuplicated(
  result$date
) == 0


# IMPORTANT:
#
# Only core probabilistic metrics are required
# to be finite.
#
# Precision/F1 may legitimately be NA when
# there are zero positive predictions.

metrics_valid <- all(
  is.finite(
    c(
      accuracy,
      brier_score,
      log_loss,
      mean_uncertainty
    )
  )
)


validation_table <- data.table(

  check = c(

    "input_exists",

    "output_exists",

    "summary_exists",

    "diagnostics_exists",

    "model_exists",

    "calibration_exists",

    "metrics_file_exists",

    "rows_positive",

    "probabilities_valid",

    "uncertainty_valid",

    "dates_sorted",

    "duplicate_dates",

    "metrics_valid"
  ),

  passed = c(

    input_exists,

    output_exists,

    summary_exists,

    diagnostics_exists,

    model_exists,

    calibration_exists,

    metrics_file_exists,

    rows_positive,

    probabilities_valid,

    uncertainty_valid,

    dates_sorted,

    duplicate_dates,

    metrics_valid
  )
)


print(
  validation_table
)


all_passed <- all(
  validation_table$passed
)


# ============================================================
# 29. FINAL VALIDATION RESULT
# ============================================================

if (all_passed) {

  cat("\n")
  cat(
    "STEP 29D VALIDATION: PASSED\n"
  )

} else {

  cat("\n")
  cat(
    "STEP 29D VALIDATION: FAILED\n"
  )

  failed_checks <- validation_table[
    passed == FALSE,
    check
  ]

  cat(
    "Failed checks:",
    paste(
      failed_checks,
      collapse = ", "
    ),
    "\n"
  )

  stop(
    "Step 29D validation failed."
  )
}


cat("\n")
cat(
  "R Bayesian regime model completed successfully.\n"
)

cat(
  "============================================\n"
)