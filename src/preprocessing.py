"""Preprocessing utilities for CKD HPO experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_SEED = 42
UCI_NUMERIC_COLUMNS = [
    "age",
    "bp",
    "sg",
    "al",
    "su",
    "bgr",
    "bu",
    "sc",
    "sod",
    "pot",
    "hemo",
    "pcv",
    "wbcc",
    "rbcc",
]


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


def clean_uci_ckd_dataframe(
    dataframe: pd.DataFrame,
    target_column: str = "class",
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw UCI CKD rows and return feature matrix plus binary target."""
    clean_data = dataframe.copy()
    clean_data.columns = [column.strip().lower() for column in clean_data.columns]

    if target_column not in clean_data.columns:
        raise ValueError(f"Target column '{target_column}' not found in UCI CKD data.")

    for column in clean_data.select_dtypes(include=["object"]).columns:
        clean_data[column] = (
            clean_data[column]
            .astype("string")
            .str.strip()
            .str.replace("\t", "", regex=False)
            .replace({"?": pd.NA, "": pd.NA, "<NA>": pd.NA})
        )

    for column in UCI_NUMERIC_COLUMNS:
        if column in clean_data.columns:
            clean_data[column] = pd.to_numeric(clean_data[column], errors="coerce")

    target = clean_data[target_column].astype("string").str.lower().str.strip()
    target = target.str.replace("\t", "", regex=False)
    target = target.map({"ckd": 1, "notckd": 0})
    valid_rows = target.notna()

    features = clean_data.loc[valid_rows].drop(columns=[target_column])
    features = features.astype(object).where(pd.notna(features), np.nan)
    labels = target.loc[valid_rows].astype(int)
    return features, labels


def prepare_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = "class",
) -> tuple[pd.DataFrame, pd.Series]:
    """Accept raw CKD data and return cleaned X and y."""
    return clean_uci_ckd_dataframe(dataframe, target_column=target_column)


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


def build_model_pipeline(
    estimator,
    features: pd.DataFrame,
    scale_numeric: bool = True,
) -> Pipeline:
    """Build a preprocessing-plus-model pipeline for raw CKD features."""
    numeric_features, categorical_features = infer_feature_types(features)
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


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
