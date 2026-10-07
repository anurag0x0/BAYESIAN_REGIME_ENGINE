# ============================================================
# STEP 29E
# FINAL R CHANGEPOINT + CONFORMAL VALIDATION
# ============================================================

required_packages <- c(
  "data.table",
  "changepoint"
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
library(changepoint)

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

PROJECT_ROOT <- "C:/BAYESIAN_REGIME_ENGINE"

INPUT_FILE <- file.path(
  PROJECT_ROOT,
  "data",
  "processed",
  "nifty50_features.csv"
)

BAYESIAN_FILE <- file.path(
  PROJECT_ROOT,
  "reports",
  "r_validation",
  "r_bayesian_regime.csv"
)

OUTPUT_DIR <- file.path(
  PROJECT_ROOT,
  "reports",
  "r_validation"
)

dir.create(
  OUTPUT_DIR,
  recursive = TRUE,
  showWarnings = FALSE
)

CHANGEPOINT_FILE <- file.path(
  OUTPUT_DIR,
  "r_changepoint_results.csv"
)

CONFORMAL_FILE <- file.path(
  OUTPUT_DIR,
  "r_conformal_results.csv"
)

SUMMARY_FILE <- file.path(
  OUTPUT_DIR,
  "r_final_validation_summary.txt"
)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

cat("\n")
cat("============================================\n")
cat("STEP 29E - FINAL R VALIDATION\n")
cat("============================================\n\n")

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

if (!file.exists(INPUT_FILE)) {
  stop("Feature dataset not found.")
}

if (!file.exists(BAYESIAN_FILE)) {
  stop("R Bayesian prediction file not found.")
}

features <- fread(INPUT_FILE)

bayesian <- fread(BAYESIAN_FILE)

features[, date := as.IDate(date)]
bayesian[, date := as.IDate(date)]

setorder(features, date)
setorder(bayesian, date)

cat(
  "Feature rows:",
  nrow(features),
  "\n"
)

cat(
  "Bayesian prediction rows:",
  nrow(bayesian),
  "\n"
)

# ------------------------------------------------------------
# 1. CHANGEPOINT DETECTION
# ------------------------------------------------------------

cat("\n")
cat("--------------------------------------------\n")
cat("1. CHANGEPOINT DETECTION\n")
cat("--------------------------------------------\n")

returns <- features$daily_return

returns <- returns[
  is.finite(returns)
]

cp_model <- cpt.meanvar(
  returns,
  method = "PELT",
  penalty = "BIC"
)

cp_positions <- cpts(
  cp_model
)

# Ignore terminal position if returned
cp_positions <- cp_positions[
  cp_positions < length(returns)
]

if (length(cp_positions) > 0) {

  cp_dates <- features$date[
    cp_positions
  ]

  cp_returns <- features$daily_return[
    cp_positions
  ]

} else {

  cp_dates <- as.IDate(character(0))

  cp_returns <- numeric(0)
}

changepoint_result <- data.table(
  changepoint_index = cp_positions,
  changepoint_date = cp_dates,
  daily_return = cp_returns
)

fwrite(
  changepoint_result,
  CHANGEPOINT_FILE
)

cat(
  "Detected changepoints:",
  nrow(changepoint_result),
  "\n"
)

cat(
  "Saved:",
  CHANGEPOINT_FILE,
  "\n"
)

# ------------------------------------------------------------
# 2. CONFORMAL VALIDATION
# ------------------------------------------------------------

cat("\n")
cat("--------------------------------------------\n")
cat("2. CONFORMAL VALIDATION\n")
cat("--------------------------------------------\n")

# Use Bayesian posterior probability.
#
# Nonconformity score:
# absolute probability error against actual class.

bayesian[
  ,
  actual_class :=
    as.integer(actual_high_volatility)
]

bayesian[
  ,
  nonconformity_score :=
    abs(
      posterior_probability -
      actual_class
    )
]

n <- nrow(bayesian)

calibration_n <- floor(
  n * 0.40
)

calibration <- bayesian[
  1:calibration_n
]

test <- bayesian[
  (calibration_n + 1):n
]

target_coverage <- 0.90

alpha <- 1 - target_coverage

# Finite-sample conformal quantile index
q_index <- ceiling(
  (calibration_n + 1) *
    target_coverage
)

q_index <- max(
  1,
  min(
    q_index,
    calibration_n
  )
)

calibration_scores <- sort(
  calibration$nonconformity_score
)

conformal_threshold <- calibration_scores[
  q_index
]

# Prediction set:
#
# Class 0 included if:
# |p - 0| <= threshold
#
# Class 1 included if:
# |p - 1| <= threshold

test[
  ,
  include_class_0 :=
    posterior_probability <=
    conformal_threshold
]

test[
  ,
  include_class_1 :=
    (1 - posterior_probability) <=
    conformal_threshold
]

test[
  ,
  prediction_set_size :=
    as.integer(include_class_0) +
    as.integer(include_class_1)
]

test[
  ,
  covered :=
    (
      (actual_class == 0 &
       include_class_0) |
      (actual_class == 1 &
       include_class_1)
    )
]

empirical_coverage <- mean(
  test$covered
)

average_set_size <- mean(
  test$prediction_set_size
)

conformal_result <- data.table(

  target_coverage =
    target_coverage,

  calibration_observations =
    calibration_n,

  test_observations =
    nrow(test),

  conformal_threshold =
    conformal_threshold,

  empirical_coverage =
    empirical_coverage,

  average_prediction_set_size =
    average_set_size
)

fwrite(
  conformal_result,
  CONFORMAL_FILE
)

cat(
  "Target coverage:",
  target_coverage,
  "\n"
)

cat(
  "Empirical coverage:",
  round(
    empirical_coverage,
    4
  ),
  "\n"
)

cat(
  "Average prediction set size:",
  round(
    average_set_size,
    4
  ),
  "\n"
)

cat(
  "Conformal threshold:",
  round(
    conformal_threshold,
    6
  ),
  "\n"
)

cat(
  "Saved:",
  CONFORMAL_FILE,
  "\n"
)

# ------------------------------------------------------------
# 3. FINAL VALIDATION
# ------------------------------------------------------------

cat("\n")
cat("--------------------------------------------\n")
cat("FINAL VALIDATION\n")
cat("--------------------------------------------\n")

input_exists <- file.exists(
  INPUT_FILE
)

bayesian_exists <- file.exists(
  BAYESIAN_FILE
)

changepoint_exists <- file.exists(
  CHANGEPOINT_FILE
)

conformal_exists <- file.exists(
  CONFORMAL_FILE
)

rows_valid <- nrow(
  bayesian
) > 0

probabilities_valid <- all(
  is.finite(
    bayesian$posterior_probability
  ) &
  bayesian$posterior_probability >= 0 &
  bayesian$posterior_probability <= 1
)

coverage_valid <- is.finite(
  empirical_coverage
)

coverage_range_valid <- (
  empirical_coverage >= 0 &
  empirical_coverage <= 1
)

dates_valid <- (
  anyDuplicated(
    bayesian$date
  ) == 0
)

validation <- data.table(

  check = c(
    "input_exists",
    "bayesian_exists",
    "changepoint_exists",
    "conformal_exists",
    "rows_valid",
    "probabilities_valid",
    "coverage_valid",
    "coverage_range_valid",
    "dates_valid"
  ),

  passed = c(
    input_exists,
    bayesian_exists,
    changepoint_exists,
    conformal_exists,
    rows_valid,
    probabilities_valid,
    coverage_valid,
    coverage_range_valid,
    dates_valid
  )
)

print(validation)

all_passed <- all(
  validation$passed
)

# ------------------------------------------------------------
# 4. SUMMARY
# ------------------------------------------------------------

summary_lines <- c(

  "============================================================",

  "STEP 29E - FINAL R VALIDATION SUMMARY",

  "============================================================",

  "",

  paste(
    "Feature observations:",
    nrow(features)
  ),

  paste(
    "Bayesian observations:",
    nrow(bayesian)
  ),

  "",

  "CHANGEPOINT:",

  paste(
    "Detected changepoints:",
    nrow(changepoint_result)
  ),

  "",

  "CONFORMAL:",

  paste(
    "Target coverage:",
    target_coverage
  ),

  paste(
    "Empirical coverage:",
    round(
      empirical_coverage,
      6
    )
  ),

  paste(
    "Average prediction set size:",
    round(
      average_set_size,
      6
    )
  ),

  paste(
    "Conformal threshold:",
    round(
      conformal_threshold,
      6
    )
  ),

  "",

  "VALIDATION:",

  paste(
    "All checks passed:",
    all_passed
  ),

  "",

  "OUTPUTS:",

  CHANGEPOINT_FILE,

  CONFORMAL_FILE,

  "",

  "============================================================"
)

writeLines(
  summary_lines,
  SUMMARY_FILE
)

# ------------------------------------------------------------
# 5. FINAL STATUS
# ------------------------------------------------------------

if (all_passed) {

  cat("\n")
  cat("============================================\n")
  cat("STEP 29E VALIDATION: PASSED\n")
  cat("============================================\n")

} else {

  cat("\n")
  cat("============================================\n")
  cat("STEP 29E VALIDATION: FAILED\n")
  cat("============================================\n")

  failed <- validation[
    passed == FALSE,
    check
  ]

  cat(
    "Failed:",
    paste(
      failed,
      collapse = ", "
    ),
    "\n"
  )

  stop(
    "Step 29E validation failed."
  )
}

cat("\n")
cat(
  "Final R validation completed successfully.\n"
)