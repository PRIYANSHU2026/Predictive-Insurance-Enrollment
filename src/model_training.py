# src/model_training.py
# ============================================================
# Model Training, Hyperparameter Tuning & MLflow Tracking
# ============================================================
"""
Trains multiple classifiers, performs hyperparameter tuning via
RandomizedSearchCV, and logs every experiment to MLflow.

Workflow (called from ``run_pipeline.py``):
1. Receive preprocessed splits + sklearn preprocessor.
2. For each candidate model (Logistic Regression, Random Forest, LightGBM):
   a. Wrap (preprocessor + classifier) in an sklearn Pipeline.
   b. Run RandomizedSearchCV with stratified k-fold.
   c. Log params, metrics, and the fitted pipeline to MLflow.
3. Select the champion model by best cross-validated ROC-AUC.
4. Persist the champion pipeline to disk.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import (
    BEST_MODEL_PATH,
    CV_FOLDS,
    CV_SCORING,
    LIGHTGBM_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    N_ITER_RANDOM_SEARCH,
    RANDOM_FOREST_PARAMS,
    RANDOM_STATE,
)

# Suppress noisy convergence / future warnings during search
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────
def _get_candidate_models() -> List[Tuple[str, Any, Dict]]:
    """
    Return a list of (name, estimator, param_grid) tuples.

    LightGBM is imported lazily so the rest of the codebase works
    even if the package is missing (graceful degradation).
    """
    candidates: List[Tuple[str, Any, Dict]] = [
        (
            "LogisticRegression",
            LogisticRegression(random_state=RANDOM_STATE),
            LOGISTIC_REGRESSION_PARAMS,
        ),
        (
            "RandomForest",
            RandomForestClassifier(random_state=RANDOM_STATE),
            RANDOM_FOREST_PARAMS,
        ),
    ]

    try:
        from lightgbm import LGBMClassifier

        candidates.append(
            (
                "LightGBM",
                LGBMClassifier(
                    random_state=RANDOM_STATE,
                    verbose=-1,  # silence per-iteration logs
                ),
                LIGHTGBM_PARAMS,
            )
        )
    except (ImportError, OSError) as exc:
        logger.warning(
            "LightGBM unavailable — skipping. (%s) "
            "On macOS you may need: brew install libomp",
            exc,
        )

    return candidates


def _evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of classification metrics.

    Returns a dict suitable for ``mlflow.log_metrics()``.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    }
    return metrics


def _log_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray) -> None:
    """Log the confusion matrix as a text artifact in MLflow."""
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    text = f"Confusion Matrix:\n{cm}\n\nClassification Report:\n{report}"
    mlflow.log_text(text, "classification_report.txt")


# ── Main Training Loop ────────────────────────────────────
def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> Pipeline:
    """
    Train all candidate models, tune hyperparameters, log to MLflow,
    and return the best-performing pipeline.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Feature matrices (raw — the preprocessor is inside the pipeline).
    y_train, y_test : pd.Series
        Binary target arrays.
    preprocessor : ColumnTransformer
        Unfitted transformer built by ``data_processing.build_preprocessor()``.

    Returns
    -------
    Pipeline
        The best sklearn Pipeline (preprocessor + classifier), already
        refitted on the full training set with the best hyperparameters.
    """
    # --- MLflow setup ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    candidates = _get_candidate_models()
    best_score = -np.inf
    best_pipeline: Pipeline | None = None
    best_model_name: str = ""

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for name, estimator, param_grid in candidates:
        logger.info("=" * 60)
        logger.info("Training model: %s", name)
        logger.info("=" * 60)

        # Build the full pipeline: preprocessor → classifier
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        # --- Hyperparameter tuning ---
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=min(N_ITER_RANDOM_SEARCH, _grid_size(param_grid)),
            scoring=CV_SCORING,
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,      # refit on full train set with best params
            verbose=0,
            return_train_score=True,
        )
        search.fit(X_train, y_train)

        fitted_pipeline = search.best_estimator_
        cv_best_score = search.best_score_

        # --- Evaluate on held-out test set ---
        test_metrics = _evaluate_model(fitted_pipeline, X_test, y_test)

        # --- Log everything to MLflow ---
        with mlflow.start_run(run_name=name):
            # Log best hyperparameters (strip the "classifier__" prefix for clarity)
            clean_params = {
                k.replace("classifier__", ""): v
                for k, v in search.best_params_.items()
            }
            mlflow.log_params(clean_params)
            mlflow.log_metric("cv_best_roc_auc", cv_best_score)
            mlflow.log_metrics(test_metrics)

            # Log confusion matrix & classification report
            y_pred = fitted_pipeline.predict(X_test)
            _log_confusion_matrix(y_test, y_pred)

            # Log the full pipeline as an MLflow model artifact
            mlflow.sklearn.log_model(fitted_pipeline, artifact_path="model")

            mlflow.set_tag("model_type", name)

        # --- Track champion ---
        logger.info(
            "%s → CV ROC-AUC=%.4f | Test ROC-AUC=%.4f | Test F1=%.4f",
            name, cv_best_score, test_metrics["test_roc_auc"], test_metrics["test_f1"],
        )

        if cv_best_score > best_score:
            best_score = cv_best_score
            best_pipeline = fitted_pipeline
            best_model_name = name

    # --- Persist the champion model ---
    assert best_pipeline is not None, "No model was trained."
    joblib.dump(best_pipeline, BEST_MODEL_PATH)

    logger.info("=" * 60)
    logger.info(
        "🏆 Champion model: %s  (CV ROC-AUC=%.4f)", best_model_name, best_score
    )
    logger.info("   Saved to: %s", BEST_MODEL_PATH)
    logger.info("=" * 60)

    return best_pipeline


def _grid_size(param_grid: Dict) -> int:
    """Compute the total number of combinations in a param grid."""
    size = 1
    for values in param_grid.values():
        size *= len(values)
    return size
