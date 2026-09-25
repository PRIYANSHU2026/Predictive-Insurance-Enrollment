# src/data_processing.py
# ============================================================
# Data Loading, Cleaning, and Preprocessing Pipeline
# ============================================================
"""
Handles every step between raw CSV and model-ready numpy arrays:

1. **load_raw_data**      – Read the CSV and perform basic validation.
2. **clean_data**         – Fix types, handle missing values, drop IDs.
3. **build_preprocessor** – Construct a reusable sklearn ColumnTransformer.
4. **split_data**         – Stratified train/test split.
5. **prepare_data**       – Orchestrator that chains all of the above.
"""

from __future__ import annotations

import logging
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_CSV_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


# ── 1. Loading ─────────────────────────────────────────────
def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw employee CSV and run sanity checks.

    Parameters
    ----------
    path : str | None
        Override for the default path in config.

    Returns
    -------
    pd.DataFrame
        Raw dataframe exactly as read from disk.
    """
    csv_path = path or RAW_CSV_PATH
    logger.info("Loading raw data from %s", csv_path)
    df = pd.read_csv(csv_path)

    # --- Sanity checks ---
    expected_cols = set(ALL_FEATURES + [TARGET_COLUMN, ID_COLUMN])
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise ValueError(
            f"Missing expected columns in CSV: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    logger.info(
        "Loaded %d rows × %d columns. Target distribution:\n%s",
        len(df),
        len(df.columns),
        df[TARGET_COLUMN].value_counts().to_string(),
    )
    return df


# ── 2. Cleaning ────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning transformations:

    * Drop the ID column (non-predictive).
    * Convert `has_dependents` from Yes/No to a string category so
      OneHotEncoder can handle it uniformly.
    * Drop rows with null target.
    * Fill remaining numerical nulls with median, categorical with mode.

    Returns a *copy* — the input frame is never mutated.
    """
    logger.info("Cleaning data …")
    df = df.copy()

    # Drop identifier — it carries no predictive signal
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    # Drop rows where the target is missing (if any)
    n_null_target = df[TARGET_COLUMN].isna().sum()
    if n_null_target > 0:
        logger.warning("Dropping %d rows with null target.", n_null_target)
        df = df.dropna(subset=[TARGET_COLUMN])

    # --- Handle missing values ---
    # Numerical: impute with median
    for col in NUMERICAL_FEATURES:
        if df[col].isna().any():
            median_val = df[col].median()
            logger.info("Imputing %s nulls with median=%.2f", col, median_val)
            df[col] = df[col].fillna(median_val)

    # Categorical: impute with mode
    for col in CATEGORICAL_FEATURES:
        if df[col].isna().any():
            mode_val = df[col].mode()[0]
            logger.info("Imputing %s nulls with mode=%s", col, mode_val)
            df[col] = df[col].fillna(mode_val)

    # Ensure correct dtypes
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    logger.info("Cleaning complete. Shape: %s", df.shape)
    return df


# ── 3. Preprocessor Construction ──────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:

    * **Numerical features** → StandardScaler (zero mean, unit variance).
    * **Categorical features** → OneHotEncoder with ``handle_unknown='ignore'``
      so unseen categories at inference time do not crash the pipeline.

    The transformer is designed to be embedded inside a full sklearn
    Pipeline alongside the classifier.
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",  # graceful at inference time
                    sparse_output=False,       # dense array for downstream compat
                    drop="if_binary",          # avoid multicollinearity for binary cols
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # silently drop any extra columns
    )

    logger.info("Built preprocessor with %d num + %d cat features.",
                len(NUMERICAL_FEATURES), len(CATEGORICAL_FEATURES))
    return preprocessor


# ── 4. Splitting ───────────────────────────────────────────
def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform a stratified train/test split on the cleaned dataframe.

    Stratification on the target ensures that train and test sets
    maintain the same class distribution — critical when the dataset
    may be imbalanced.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[ALL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split data → train=%d, test=%d  (test_size=%.0f%%)",
        len(X_train), len(X_test), TEST_SIZE * 100,
    )
    return X_train, X_test, y_train, y_test


# ── 5. Orchestrator ───────────────────────────────────────
def prepare_data(
    raw_path: str | None = None,
    persist: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    End-to-end data preparation:

    1. Load raw CSV.
    2. Clean the data.
    3. Split into train / test.
    4. Build (but do NOT fit) the preprocessor.
    5. Optionally persist processed splits to ``data/processed/``.

    The preprocessor is returned *unfitted* so it can be fitted inside
    the sklearn Pipeline during model training — this avoids data leakage.

    Parameters
    ----------
    raw_path : str | None
        Override path to the raw CSV file.
    persist : bool
        If True, save train/test CSVs to ``data/processed/``.

    Returns
    -------
    X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_raw_data(raw_path)
    df = clean_data(df)
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()

    if persist:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        train_df = X_train.copy()
        train_df[TARGET_COLUMN] = y_train
        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)

        test_df = X_test.copy()
        test_df[TARGET_COLUMN] = y_test
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)

        logger.info("Persisted train/test splits to %s", PROCESSED_DATA_DIR)

    return X_train, X_test, y_train, y_test, preprocessor


# ── CLI convenience ────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print(f"\n✅ Data prepared. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
