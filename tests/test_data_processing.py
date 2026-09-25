# tests/test_data_processing.py
# ============================================================
# Unit Tests for the Data Processing Module
# ============================================================
"""
Validates that every stage of the data processing pipeline behaves
correctly under normal and edge-case inputs.
"""

import pandas as pd
import pytest

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLUMN,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
)
from src.data_processing import (
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_data,
)


# ── Fixtures ──────────────────────────────────────────────
@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Create a minimal synthetic dataframe matching the schema."""
    return pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "age": [25, 30, 45, 50, 35, 28, 42, 55, 33, 60],
            "gender": [
                "Male", "Female", "Male", "Female", "Male",
                "Female", "Male", "Female", "Male", "Female",
            ],
            "marital_status": [
                "Single", "Married", "Divorced", "Married", "Single",
                "Single", "Married", "Divorced", "Married", "Single",
            ],
            "salary": [
                50000, 60000, 75000, 80000, 55000,
                45000, 70000, 90000, 62000, 85000,
            ],
            "employment_type": [
                "Full-time", "Part-time", "Contract", "Full-time", "Part-time",
                "Full-time", "Contract", "Full-time", "Part-time", "Contract",
            ],
            "region": [
                "West", "South", "Northeast", "Midwest", "West",
                "South", "Northeast", "Midwest", "West", "South",
            ],
            "has_dependents": [
                "Yes", "No", "Yes", "No", "Yes",
                "No", "Yes", "No", "Yes", "No",
            ],
            "tenure_years": [1.0, 5.0, 10.0, 15.0, 3.0, 2.0, 8.0, 20.0, 4.0, 12.0],
            "enrolled": [0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
        }
    )


@pytest.fixture
def cleaned_df(sample_raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned version of the sample dataframe."""
    return clean_data(sample_raw_df)


# ── Tests: clean_data ─────────────────────────────────────
class TestCleanData:
    """Tests for the clean_data function."""

    def test_id_column_dropped(self, cleaned_df: pd.DataFrame):
        """Employee ID should be removed — it has no predictive value."""
        assert ID_COLUMN not in cleaned_df.columns

    def test_no_missing_values(self, cleaned_df: pd.DataFrame):
        """After cleaning, there should be zero NaN values."""
        assert cleaned_df.isna().sum().sum() == 0

    def test_numerical_dtypes(self, cleaned_df: pd.DataFrame):
        """Numerical features must be numeric dtype."""
        for col in NUMERICAL_FEATURES:
            assert pd.api.types.is_numeric_dtype(cleaned_df[col]), (
                f"{col} should be numeric, got {cleaned_df[col].dtype}"
            )

    def test_categorical_dtypes(self, cleaned_df: pd.DataFrame):
        """Categorical features must be string dtype after cleaning."""
        for col in CATEGORICAL_FEATURES:
            assert cleaned_df[col].dtype == object or pd.api.types.is_string_dtype(
                cleaned_df[col]
            ), f"{col} should be string, got {cleaned_df[col].dtype}"

    def test_target_preserved(self, cleaned_df: pd.DataFrame):
        """The target column must survive cleaning."""
        assert TARGET_COLUMN in cleaned_df.columns

    def test_handles_missing_values(self, sample_raw_df: pd.DataFrame):
        """Inject NaNs and verify imputation fills them."""
        df_with_na = sample_raw_df.copy()
        df_with_na.loc[0, "age"] = None
        df_with_na.loc[1, "gender"] = None

        result = clean_data(df_with_na)
        assert result["age"].isna().sum() == 0
        assert result["gender"].isna().sum() == 0

    def test_immutability(self, sample_raw_df: pd.DataFrame):
        """clean_data must not mutate the input dataframe."""
        original = sample_raw_df.copy()
        _ = clean_data(sample_raw_df)
        pd.testing.assert_frame_equal(sample_raw_df, original)


# ── Tests: split_data ─────────────────────────────────────
class TestSplitData:
    """Tests for the split_data function."""

    def test_split_sizes(self, cleaned_df: pd.DataFrame):
        """Train + test rows should equal the total rows."""
        X_train, X_test, y_train, y_test = split_data(cleaned_df)
        assert len(X_train) + len(X_test) == len(cleaned_df)

    def test_no_target_leakage(self, cleaned_df: pd.DataFrame):
        """X splits must not contain the target column."""
        X_train, X_test, _, _ = split_data(cleaned_df)
        assert TARGET_COLUMN not in X_train.columns
        assert TARGET_COLUMN not in X_test.columns

    def test_stratification(self, cleaned_df: pd.DataFrame):
        """Class proportions in train ≈ class proportions in test."""
        _, _, y_train, y_test = split_data(cleaned_df)
        train_ratio = y_train.mean()
        test_ratio = y_test.mean()
        # Allow up to 30% relative difference for a tiny 10-row dataset
        assert abs(train_ratio - test_ratio) < 0.30


# ── Tests: build_preprocessor ────────────────────────────
class TestBuildPreprocessor:
    """Tests for the preprocessor construction."""

    def test_preprocessor_has_transformers(self):
        """The ColumnTransformer should contain both num and cat pipelines."""
        preprocessor = build_preprocessor()
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "num" in transformer_names
        assert "cat" in transformer_names

    def test_preprocessor_fits_and_transforms(self, cleaned_df: pd.DataFrame):
        """Preprocessor should fit on X and produce a 2-D array."""
        preprocessor = build_preprocessor()
        X = cleaned_df[ALL_FEATURES]
        transformed = preprocessor.fit_transform(X)
        assert transformed.ndim == 2
        assert transformed.shape[0] == len(X)
