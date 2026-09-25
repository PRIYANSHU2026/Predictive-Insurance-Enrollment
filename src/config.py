# src/config.py
# ============================================================
# Centralised Configuration for the ML Pipeline
# ============================================================
"""
Single source of truth for file paths, feature definitions, model
hyperparameters, and MLflow settings.  Every other module imports from
here so that changes propagate automatically.
"""

from pathlib import Path

# ── Project Root ────────────────────────────────────────────
# Resolve relative to this file so it works regardless of cwd.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Data Paths ──────────────────────────────────────────────
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RAW_CSV_PATH = RAW_DATA_DIR / "employee_data.csv"

# ── Artifact Paths ──────────────────────────────────────────
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

# ── Feature Definitions ────────────────────────────────────
# These lists must stay in sync with the dataset schema.
TARGET_COLUMN = "enrolled"
ID_COLUMN = "employee_id"

NUMERICAL_FEATURES = [
    "age",
    "salary",
    "tenure_years",
]

CATEGORICAL_FEATURES = [
    "gender",
    "marital_status",
    "employment_type",
    "region",
    "has_dependents",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ── Train / Test Split ─────────────────────────────────────
TEST_SIZE = 0.20
RANDOM_STATE = 42         # Global seed for reproducibility

# ── MLflow Settings ─────────────────────────────────────────
MLFLOW_EXPERIMENT_NAME = "insurance_enrollment_prediction"
MLFLOW_TRACKING_URI = str(PROJECT_ROOT / "mlruns")

# ── Model Hyperparameter Search Spaces ──────────────────────
# Used by RandomizedSearchCV in model_training.py.

LOGISTIC_REGRESSION_PARAMS = {
    "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "classifier__penalty": ["l2"],
    "classifier__solver": ["lbfgs"],
    "classifier__max_iter": [1000],
}

RANDOM_FOREST_PARAMS = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [5, 10, 15, 20, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__class_weight": ["balanced", None],
}

LIGHTGBM_PARAMS = {
    "classifier__n_estimators": [100, 200, 300, 500],
    "classifier__max_depth": [3, 5, 7, 10, -1],
    "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "classifier__num_leaves": [15, 31, 63, 127],
    "classifier__min_child_samples": [5, 10, 20, 50],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "classifier__class_weight": ["balanced", None],
}

# ── Cross-Validation ───────────────────────────────────────
CV_FOLDS = 5
CV_SCORING = "roc_auc"     # Primary metric for model selection
N_ITER_RANDOM_SEARCH = 30  # Number of random search iterations
