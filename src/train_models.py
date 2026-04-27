"""Model factory functions for CKD prediction experiments."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


RANDOM_SEED = 42


def build_random_forest(random_state: int = RANDOM_SEED) -> RandomForestClassifier:
    """Create a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost(random_state: int = RANDOM_SEED):
    """Create an XGBoost classifier.

    XGBoost is listed in requirements.txt. If unavailable in a restricted
    environment, install xgboost or skip this model in local experiments.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for build_xgboost(). Install it with "
            "`pip install xgboost`."
        ) from exc

    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def build_svm(random_state: int = RANDOM_SEED) -> SVC:
    """Create an SVM classifier with probability estimates enabled."""
    return SVC(
        C=1.0,
        kernel="rbf",
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=random_state,
    )


def build_model(model_name: str, random_state: int = RANDOM_SEED):
    """Build a model by name."""
    normalized_name = model_name.strip().lower().replace("_", "-")
    if normalized_name in {"random-forest", "rf"}:
        return build_random_forest(random_state=random_state)
    if normalized_name in {"xgboost", "xgb"}:
        return build_xgboost(random_state=random_state)
    if normalized_name in {"svm", "support-vector-machine"}:
        return build_svm(random_state=random_state)
    raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    for name in ("random-forest", "svm"):
        print(name, build_model(name))
