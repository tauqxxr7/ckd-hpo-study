"""Preprocessing utilities for CKD HPO experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42


def load_uci_ckd(path: str | Path) -> pd.DataFrame:
    """Load a local copy of the UCI CKD dataset.

    The raw dataset is not committed to this repository. Download it from the
    UCI Machine Learning Repository and pass the local file path here.
    """
    return pd.read_csv(path)


def load_eicu_ckd(path: str | Path) -> pd.DataFrame:
    """Load a locally prepared eICU-derived CKD cohort.

    eICU data require credentialed PhysioNet access. This function expects a
    de-identified, locally prepared analysis table and does not perform cohort
    extraction from raw eICU tables.
    """
    return pd.read_csv(path)


def split_features_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target label."""
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' not found.")
    return dataframe.drop(columns=[target_column]), dataframe[target_column]


def infer_feature_types(
    dataframe: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature names from a dataframe."""
    numeric_features = dataframe.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [
        column for column in dataframe.columns if column not in numeric_features
    ]
    return numeric_features, categorical_features


def build_preprocessor(
    numeric_features: Iterable[str],
    categorical_features: Iterable[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Create a leakage-safe preprocessing transformer."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )


def make_train_test_split(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified train-test split."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
