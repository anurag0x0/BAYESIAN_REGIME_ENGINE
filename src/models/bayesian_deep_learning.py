# ============================================================
# PROJECT 1A
# BAYESIAN REGIME DETECTION ENGINE
#
# STEP 22
# Bayesian Deep Learning using MC Dropout
# ============================================================
#
# Objective:
#   Predict next-day NIFTY 50 direction using a neural network
#   with Monte Carlo Dropout for predictive uncertainty.
#
# Method:
#   1. Chronological train/test split
#   2. Feature scaling using training data only
#   3. Feed-forward neural network
#   4. Dropout layers
#   5. MC Dropout inference
#   6. Multiple stochastic forward passes
#   7. Mean probability
#   8. Predictive uncertainty
#   9. Predictive entropy
#   10. Confidence
#
# IMPORTANT:
#   Future target columns are never used as input features.
#
# ============================================================

from pathlib import Path
import json
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
)

import joblib

warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "processed"
    / "nifty50_features.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
)

REPORT_DIR = (
    ROOT
    / "reports"
    / "bayesian_deep_learning"
)

MODEL_DIR = (
    ROOT
    / "models"
    / "bayesian_deep_learning"
)

PREDICTION_FILE = (
    OUTPUT_DIR
    / "bayesian_deep_learning_predictions.csv"
)

METRICS_FILE = (
    REPORT_DIR
    / "metrics.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "summary.txt"
)

CONFIG_FILE = (
    MODEL_DIR
    / "model_config.json"
)

SCALER_FILE = (
    MODEL_DIR
    / "feature_scaler.pkl"
)

MODEL_FILE = (
    MODEL_DIR
    / "mc_dropout_model.pt"
)


# ============================================================
# 2. CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TRAIN_RATIO = 0.80

BATCH_SIZE = 64

EPOCHS = 40

LEARNING_RATE = 0.001

DROPOUT_RATE = 0.25

HIDDEN_1 = 64

HIDDEN_2 = 32

# Number of stochastic MC Dropout passes
MC_SAMPLES = 100

# Early stopping
PATIENCE = 8

# Device
DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# ============================================================
# 3. RANDOM SEED
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# 4. CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Input file: {INPUT_FILE}"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Columns: {len(df.columns)}"
    )

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_candidates = [
        "date",
        "Date",
        "DATE",
    ]

    date_column = None

    for col in date_candidates:

        if col in df.columns:

            date_column = col

            break

    if date_column is None:

        raise ValueError(
            "Date column not found."
        )

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    target_candidates = [
        "target_next_day_direction",
        "next_day_direction",
        "target_direction",
    ]

    target_column = None

    for col in target_candidates:

        if col in df.columns:

            target_column = col

            break

    if target_column is None:

        raise ValueError(
            "Target direction column not found."
        )

    # --------------------------------------------------------
    # Return column
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
            "Return column not found."
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df
        .dropna(subset=[date_column])
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    print(
        f"Date column: {date_column}"
    )

    print(
        f"Target column: {target_column}"
    )

    print(
        f"Return column: {return_column}"
    )

    # --------------------------------------------------------
    # Convert target
    # --------------------------------------------------------

    df[target_column] = pd.to_numeric(
        df[target_column],
        errors="coerce"
    )

    df = df.dropna(
        subset=[target_column]
    )

    # --------------------------------------------------------
    # Keep binary target only
    # --------------------------------------------------------

    unique_targets = sorted(
        df[target_column]
        .dropna()
        .unique()
        .tolist()
    )

    print(
        f"Target values: {unique_targets}"
    )

    if not set(unique_targets).issubset(
        {0, 1}
    ):

        raise ValueError(
            "Target must contain binary 0/1 values."
        )

    return (
        df,
        date_column,
        target_column,
        return_column,
    )


# ============================================================
# 6. SELECT FEATURES
# ============================================================

def select_features(
    df,
    date_column,
    target_column,
):
    """
    Select numeric market features while explicitly
    excluding target/future columns.

    This is important for leakage prevention.
    """

    print("\n" + "=" * 70)
    print("SELECTING MODEL FEATURES")
    print("=" * 70)

    numeric_columns = (
        df
        .select_dtypes(
            include=[np.number]
        )
        .columns
        .tolist()
    )

    # --------------------------------------------------------
    # Explicit leakage exclusions
    # --------------------------------------------------------

    forbidden_keywords = [
        "target",
        "next_day",
        "future",
        "actual",
        "prediction",
        "label",
    ]

    excluded = []

    features = []

    for col in numeric_columns:

        col_lower = col.lower()

        if col == target_column:

            excluded.append(col)

            continue

        if any(
            keyword in col_lower
            for keyword in forbidden_keywords
        ):

            excluded.append(col)

            continue

        features.append(col)

    # --------------------------------------------------------
    # Remove date-like numeric columns
    # --------------------------------------------------------

    if date_column in features:

        features.remove(
            date_column
        )

    # --------------------------------------------------------
    # Remove columns with all missing
    # --------------------------------------------------------

    features = [
        col
        for col in features
        if not df[col].isna().all()
    ]

    if len(features) == 0:

        raise ValueError(
            "No valid numeric features found."
        )

    print(
        f"Selected features: {len(features)}"
    )

    print("\nFeatures:")

    for feature in features:

        print(
            f"  - {feature}"
        )

    print("\nExcluded columns:")

    for col in excluded:

        print(
            f"  - {col}"
        )

    return features


# ============================================================
# 7. CLEAN FEATURE MATRIX
# ============================================================

def prepare_dataset(
    df,
    features,
    target_column,
):
    print("\n" + "=" * 70)
    print("PREPARING FEATURE MATRIX")
    print("=" * 70)

    work = df[
        [
            "date"
            if "date" in df.columns
            else df.columns[
                0
            ],
            *features,
            target_column,
        ]
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    for feature in features:

        work[feature] = pd.to_numeric(
            work[feature],
            errors="coerce"
        )

    work[target_column] = pd.to_numeric(
        work[target_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Replace infinities
    # --------------------------------------------------------

    work = work.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Drop missing rows
    # --------------------------------------------------------

    before = len(work)

    work = work.dropna(
        subset=features + [target_column]
    ).reset_index(
        drop=True
    )

    after = len(work)

    print(
        f"Rows before cleaning: {before}"
    )

    print(
        f"Rows after cleaning : {after}"
    )

    if after < 500:

        raise ValueError(
            "Too few rows after cleaning."
        )

    X = work[
        features
    ].values.astype(
        np.float32
    )

    y = work[
        target_column
    ].values.astype(
        np.float32
    )

    dates = pd.to_datetime(
        work["date"]
    )

    return (
        work,
        X,
        y,
        dates,
    )


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

def chronological_split(
    X,
    y,
    dates,
):
    print("\n" + "=" * 70)
    print("CHRONOLOGICAL TRAIN / TEST SPLIT")
    print("=" * 70)

    split_index = int(
        len(X)
        * TRAIN_RATIO
    )

    X_train = X[
        :split_index
    ]

    X_test = X[
        split_index:
    ]

    y_train = y[
        :split_index
    ]

    y_test = y[
        split_index:
    ]

    dates_train = dates[
        :split_index
    ]

    dates_test = dates[
        split_index:
    ]

    print(
        f"Total : {len(X)}"
    )

    print(
        f"Train : {len(X_train)}"
    )

    print(
        f"Test  : {len(X_test)}"
    )

    print(
        f"Train period: "
        f"{dates_train.iloc[0].date()} -> "
        f"{dates_train.iloc[-1].date()}"
    )

    print(
        f"Test period : "
        f"{dates_test.iloc[0].date()} -> "
        f"{dates_test.iloc[-1].date()}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train,
        dates_test,
    )


# ============================================================
# 9. SCALE FEATURES
# ============================================================

def scale_features(
    X_train,
    X_test,
):
    print("\n" + "=" * 70)
    print("SCALING FEATURES")
    print("=" * 70)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    ).astype(
        np.float32
    )

    X_test_scaled = scaler.transform(
        X_test
    ).astype(
        np.float32
    )

    print(
        "Scaler fitted using TRAINING DATA ONLY."
    )

    joblib.dump(
        scaler,
        SCALER_FILE
    )

    print(
        f"Scaler saved:\n{SCALER_FILE}"
    )

    return (
        X_train_scaled,
        X_test_scaled,
        scaler,
    )


# ============================================================
# 10. BAYESIAN NEURAL NETWORK
# ============================================================

class MCDropoutNetwork(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        dropout_rate=DROPOUT_RATE,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                hidden_1
            ),

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                hidden_1,
                hidden_2
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                hidden_2
            ),

            nn.Dropout(
                dropout_rate
            ),

            nn.Linear(
                hidden_2,
                1
            ),
        )

    def forward(
        self,
        x,
    ):

        return self.network(
            x
        )


# ============================================================
# 11. TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train,
    input_dim,
):
    print("\n" + "=" * 70)
    print("TRAINING MC DROPOUT NEURAL NETWORK")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Input features: {input_dim}"
    )

    print(
        f"Hidden layers: "
        f"{HIDDEN_1} -> {HIDDEN_2}"
    )

    print(
        f"Dropout rate: {DROPOUT_RATE}"
    )

    # --------------------------------------------------------
    # Tensor conversion
    # --------------------------------------------------------

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_tensor = torch.tensor(
        y_train.reshape(-1, 1),
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MCDropoutNetwork(
        input_dim=input_dim
    ).to(
        DEVICE
    )

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_loss = np.inf

    best_state = None

    patience_counter = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        samples = 0

        for batch_X, batch_y in loader:

            batch_X = batch_X.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(
                batch_X
            )

            loss = criterion(
                logits,
                batch_y
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                batch_X.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            samples += batch_size

        epoch_loss = (
            running_loss
            / samples
        )

        if (
            epoch == 1
            or epoch % 5 == 0
            or epoch == EPOCHS
        ):

            print(
                f"Epoch "
                f"{epoch:03d}/{EPOCHS} "
                f"| Loss: "
                f"{epoch_loss:.6f}"
            )

        # ----------------------------------------------------
        # Early stopping based on training objective
        # ----------------------------------------------------

        if epoch_loss < (
            best_loss - 1e-5
        ):

            best_loss = epoch_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "input_dim":
                input_dim,

            "hidden_1":
                HIDDEN_1,

            "hidden_2":
                HIDDEN_2,

            "dropout_rate":
                DROPOUT_RATE,

        },
        MODEL_FILE
    )

    print(
        f"\nModel saved:\n{MODEL_FILE}"
    )

    return model


# ============================================================
# 12. MC DROPOUT INFERENCE
# ============================================================

def mc_dropout_predict(
    model,
    X,
    mc_samples=MC_SAMPLES,
):
    """
    Perform stochastic forward passes.

    Dropout remains ACTIVE during inference.

    Each pass produces a different probability.

    We then calculate:

        mean probability
        standard deviation
        confidence
        entropy
        direction
    """

    print("\n" + "=" * 70)
    print("MC DROPOUT PREDICTION")
    print("=" * 70)

    print(
        f"MC samples: {mc_samples}"
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    ).to(
        DEVICE
    )

    # --------------------------------------------------------
    # Keep dropout active
    # --------------------------------------------------------

    model.train()

    predictions = []

    with torch.no_grad():

        for sample in range(
            mc_samples
        ):

            logits = model(
                X_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

            probabilities = (
                probabilities
                .detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            predictions.append(
                probabilities
            )

    predictions = np.asarray(
        predictions,
        dtype=np.float64
    )

    # Shape:
    # (MC samples, observations)

    mean_probability = (
        predictions.mean(
            axis=0
        )
    )

    predictive_std = (
        predictions.std(
            axis=0
        )
    )

    # --------------------------------------------------------
    # Predictive entropy
    # --------------------------------------------------------

    p = np.clip(
        mean_probability,
        1e-8,
        1 - 1e-8
    )

    entropy = -(
        p * np.log(p)
        +
        (1 - p)
        * np.log(1 - p)
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = np.maximum(
        mean_probability,
        1 - mean_probability
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    predicted_direction = (
        mean_probability >= 0.5
    ).astype(int)

    # --------------------------------------------------------
    # Uncertainty state
    # --------------------------------------------------------

    uncertainty_state = np.select(
        [
            predictive_std < 0.02,
            predictive_std < 0.05,
        ],
        [
            "Low",
            "Medium",
        ],
        default="High",
    )

    return {
        "mean_probability":
            mean_probability,

        "predictive_std":
            predictive_std,

        "entropy":
            entropy,

        "confidence":
            confidence,

        "predicted_direction":
            predicted_direction,

        "uncertainty_state":
            uncertainty_state,

        "mc_predictions":
            predictions,
    }


# ============================================================
# 13. EVALUATE MODEL
# ============================================================

def evaluate_predictions(
    y_true,
    predictions,
):
    print("\n" + "=" * 70)
    print("EVALUATING BAYESIAN DEEP LEARNING MODEL")
    print("=" * 70)

    probability = (
        predictions[
            "mean_probability"
        ]
    )

    direction = (
        predictions[
            "predicted_direction"
        ]
    )

    # --------------------------------------------------------
    # Core metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        direction
    )

    precision = precision_score(
        y_true,
        direction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        direction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        direction,
        zero_division=0
    )

    brier = brier_score_loss(
        y_true,
        probability
    )

    logloss = log_loss(
        y_true,
        np.column_stack(
            [
                1 - probability,
                probability,
            ]
        ),
        labels=[0, 1],
    )

    try:

        roc_auc = roc_auc_score(
            y_true,
            probability
        )

    except ValueError:

        roc_auc = np.nan

    cm = confusion_matrix(
        y_true,
        direction
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1 Score : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"Brier    : {brier:.4f}"
    )

    print(
        f"Log Loss : {logloss:.4f}"
    )

    print("\nConfusion Matrix:")

    print(cm)

    metrics = {
        "model":
            "Bayesian Deep Learning MC Dropout",

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "brier_score":
            float(brier),

        "log_loss":
            float(logloss),

        "mc_samples":
            MC_SAMPLES,

        "test_observations":
            len(y_true),

        "mean_predictive_std":
            float(
                predictions[
                    "predictive_std"
                ].mean()
            ),

        "mean_confidence":
            float(
                predictions[
                    "confidence"
                ].mean()
            ),

        "mean_entropy":
            float(
                predictions[
                    "entropy"
                ].mean()
            ),
    }

    return metrics


# ============================================================
# 14. CREATE PREDICTION DATASET
# ============================================================

def create_prediction_dataset(
    dates,
    y_true,
    predictions,
):
    print("\n" + "=" * 70)
    print("CREATING PREDICTION DATASET")
    print("=" * 70)

    result = pd.DataFrame(
        {
            "forecast_for_date":
                pd.to_datetime(
                    dates
                ),

            "actual_next_day_direction":
                y_true.astype(int),

            "probability_positive":
                predictions[
                    "mean_probability"
                ],

            "predictive_std":
                predictions[
                    "predictive_std"
                ],

            "predictive_entropy":
                predictions[
                    "entropy"
                ],

            "confidence":
                predictions[
                    "confidence"
                ],

            "predicted_direction":
                predictions[
                    "predicted_direction"
                ],

            "uncertainty_state":
                predictions[
                    "uncertainty_state"
                ],
        }
    )

    # --------------------------------------------------------
    # Prediction error
    # --------------------------------------------------------

    result[
        "prediction_correct"
    ] = (
        result[
            "predicted_direction"
        ]
        ==
        result[
            "actual_next_day_direction"
        ]
    ).astype(int)

    # --------------------------------------------------------
    # Uncertainty percentile
    # --------------------------------------------------------

    result[
        "uncertainty_percentile"
    ] = (
        result[
            "predictive_std"
        ]
        .rank(
            pct=True
        )
        * 100
    )

    # --------------------------------------------------------
    # Direction label
    # --------------------------------------------------------

    result[
        "predicted_direction_label"
    ] = np.where(
        result[
            "predicted_direction"
        ] == 1,
        "UP",
        "DOWN"
    )

    result[
        "actual_direction_label"
    ] = np.where(
        result[
            "actual_next_day_direction"
        ] == 1,
        "UP",
        "DOWN"
    )

    result.to_csv(
        PREDICTION_FILE,
        index=False
    )

    print(
        f"Prediction file saved:\n"
        f"{PREDICTION_FILE}"
    )

    print(
        f"Rows: {len(result)}"
    )

    return result


# ============================================================
# 15. SAVE METRICS
# ============================================================

def save_metrics(
    metrics,
):

    metrics_df = pd.DataFrame(
        [metrics]
    )

    metrics_df.to_csv(
        METRICS_FILE,
        index=False
    )

    print(
        f"Metrics saved:\n"
        f"{METRICS_FILE}"
    )


# ============================================================
# 16. SAVE SUMMARY
# ============================================================

def save_summary(
    metrics,
    features,
    n_train,
    n_test,
):
    summary = []

    summary.append(
        "BAYESIAN DEEP LEARNING - MC DROPOUT"
    )

    summary.append(
        "=" * 60
    )

    summary.append(
        "Model: Feed-Forward Neural Network + MC Dropout"
    )

    summary.append(
        f"Device: {DEVICE}"
    )

    summary.append(
        f"Training observations: {n_train}"
    )

    summary.append(
        f"Test observations: {n_test}"
    )

    summary.append(
        f"Feature count: {len(features)}"
    )

    summary.append(
        f"MC samples: {MC_SAMPLES}"
    )

    summary.append(
        f"Dropout rate: {DROPOUT_RATE}"
    )

    summary.append("")

    summary.append(
        "Performance:"
    )

    summary.append(
        f"Accuracy: "
        f"{metrics['accuracy']:.6f}"
    )

    summary.append(
        f"Precision: "
        f"{metrics['precision']:.6f}"
    )

    summary.append(
        f"Recall: "
        f"{metrics['recall']:.6f}"
    )

    summary.append(
        f"F1: "
        f"{metrics['f1']:.6f}"
    )

    summary.append(
        f"ROC-AUC: "
        f"{metrics['roc_auc']:.6f}"
    )

    summary.append(
        f"Brier Score: "
        f"{metrics['brier_score']:.6f}"
    )

    summary.append(
        f"Log Loss: "
        f"{metrics['log_loss']:.6f}"
    )

    summary.append("")

    summary.append(
        "Uncertainty:"
    )

    summary.append(
        f"Mean predictive std: "
        f"{metrics['mean_predictive_std']:.6f}"
    )

    summary.append(
        f"Mean confidence: "
        f"{metrics['mean_confidence']:.6f}"
    )

    summary.append(
        f"Mean entropy: "
        f"{metrics['mean_entropy']:.6f}"
    )

    SUMMARY_FILE.write_text(
        "\n".join(summary),
        encoding="utf-8"
    )

    print(
        f"Summary saved:\n"
        f"{SUMMARY_FILE}"
    )


# ============================================================
# 17. SAVE CONFIGURATION
# ============================================================

def save_config(
    features,
    input_dim,
):

    config = {

        "model":
            "MC Dropout Bayesian Neural Network",

        "input_dim":
            input_dim,

        "features":
            features,

        "hidden_1":
            HIDDEN_1,

        "hidden_2":
            HIDDEN_2,

        "dropout_rate":
            DROPOUT_RATE,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "mc_samples":
            MC_SAMPLES,

        "train_ratio":
            TRAIN_RATIO,

        "random_seed":
            RANDOM_SEED,

        "device":
            str(DEVICE),
    }

    CONFIG_FILE.write_text(
        json.dumps(
            config,
            indent=4
        ),
        encoding="utf-8"
    )

    print(
        f"Configuration saved:\n"
        f"{CONFIG_FILE}"
    )


# ============================================================
# 18. FINAL VALIDATION
# ============================================================

def final_validation(
    prediction_df,
):
    print("\n" + "=" * 70)
    print("FINAL VALIDATION")
    print("=" * 70)

    probability = (
        prediction_df[
            "probability_positive"
        ]
    )

    predictive_std = (
        prediction_df[
            "predictive_std"
        ]
    )

    # --------------------------------------------------------
    # Probability range
    # --------------------------------------------------------

    probability_valid = (
        probability
        .between(
            0,
            1
        )
        .all()
    )

    # --------------------------------------------------------
    # Uncertainty finite
    # --------------------------------------------------------

    uncertainty_valid = (
        np.isfinite(
            predictive_std
        ).all()
    )

    # --------------------------------------------------------
    # Dates
    # --------------------------------------------------------

    dates_sorted = (
        prediction_df[
            "forecast_for_date"
        ]
        .is_monotonic_increasing
    )

    # --------------------------------------------------------
    # Duplicate dates
    # --------------------------------------------------------

    duplicate_dates = (
        prediction_df[
            "forecast_for_date"
        ]
        .duplicated()
        .sum()
    )

    no_duplicates = (
        duplicate_dates == 0
    )

    # --------------------------------------------------------
    # Direction validity
    # --------------------------------------------------------

    directions_valid = (
        prediction_df[
            "predicted_direction"
        ]
        .isin([0, 1])
        .all()
    )

    print(
        f"Probability range valid : "
        f"{probability_valid}"
    )

    print(
        f"Uncertainty finite      : "
        f"{uncertainty_valid}"
    )

    print(
        f"Dates sorted            : "
        f"{dates_sorted}"
    )

    print(
        f"No duplicate dates      : "
        f"{no_duplicates}"
    )

    print(
        f"Directions valid        : "
        f"{directions_valid}"
    )

    passed = (
        probability_valid
        and uncertainty_valid
        and dates_sorted
        and no_duplicates
        and directions_valid
    )

    print(
        f"\nSTEP 22 VALIDATION: "
        f"{'PASSED' if passed else 'FAILED'}"
    )

    return passed


# ============================================================
# 19. MAIN
# ============================================================

def main():

    print("\n")

    print("=" * 70)

    print(
        "PROJECT 1A - BAYESIAN REGIME DETECTION ENGINE"
    )

    print(
        "STEP 22 - BAYESIAN DEEP LEARNING / MC DROPOUT"
    )

    print("=" * 70)

    set_seed()

    print(
        f"\nUsing device: {DEVICE}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        df,
        date_column,
        target_column,
        return_column,
    ) = load_data()

    # --------------------------------------------------------
    # Rename date consistently
    # --------------------------------------------------------

    if date_column != "date":

        df = df.rename(
            columns={
                date_column:
                    "date"
            }
        )

        date_column = "date"

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    features = select_features(
        df,
        date_column,
        target_column,
    )

    # --------------------------------------------------------
    # Prepare dataset
    # --------------------------------------------------------

    (
        work,
        X,
        y,
        dates,
    ) = prepare_dataset(
        df,
        features,
        target_column,
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        dates_train,
        dates_test,
    ) = chronological_split(
        X,
        y,
        dates,
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    (
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = scale_features(
        X_train,
        X_test,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train_scaled,
        y_train,
        input_dim=X_train_scaled.shape[1],
    )

    # --------------------------------------------------------
    # MC Dropout
    # --------------------------------------------------------

    predictions = mc_dropout_predict(
        model,
        X_test_scaled,
        mc_samples=MC_SAMPLES,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    metrics = evaluate_predictions(
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Prediction dataset
    # --------------------------------------------------------

    prediction_df = create_prediction_dataset(
        dates_test,
        y_test,
        predictions,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    save_metrics(
        metrics
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    save_summary(
        metrics,
        features,
        len(X_train),
        len(X_test),
    )

    # --------------------------------------------------------
    # Save config
    # --------------------------------------------------------

    save_config(
        features,
        X_train_scaled.shape[1],
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_passed = final_validation(
        prediction_df
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    if validation_passed:

        print(
            "STEP 22 COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "STEP 22 COMPLETED WITH VALIDATION ISSUES"
        )

    print("=" * 70)

    print("\nGenerated files:")

    print(
        f"1. {PREDICTION_FILE}"
    )

    print(
        f"2. {METRICS_FILE}"
    )

    print(
        f"3. {SUMMARY_FILE}"
    )

    print(
        f"4. {CONFIG_FILE}"
    )

    print(
        f"5. {SCALER_FILE}"
    )

    print(
        f"6. {MODEL_FILE}"
    )

    print("\nFinal model metrics:")

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Brier    : "
        f"{metrics['brier_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{metrics['roc_auc']:.4f}"
    )

    print(
        f"Log Loss : "
        f"{metrics['log_loss']:.4f}"
    )

    print(
        f"Mean uncertainty: "
        f"{metrics['mean_predictive_std']:.4f}"
    )

    print(
        "\nBayesian Deep Learning with MC Dropout "
        "is ready for ensemble integration."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()