"""Evaluation metrics for CKD prediction experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Container for binary classification metrics."""

    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    confusion_matrix: np.ndarray


def evaluate_predictions(y_true, y_pred, y_score=None) -> EvaluationResult:
    """Evaluate binary predictions."""
    roc_auc = None
    if y_score is not None:
        roc_auc = roc_auc_score(y_true, y_score)

    return EvaluationResult(
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1_score=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc,
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )


def evaluate_binary_classifier(model, x_test, y_test) -> EvaluationResult:
    """Evaluate a fitted binary classifier."""
    y_pred = model.predict(x_test)
    y_score = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)
        if probabilities.shape[1] > 1:
            y_score = probabilities[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(x_test)

    return evaluate_predictions(y_test, y_pred, y_score=y_score)


if __name__ == "__main__":
    example = evaluate_predictions([0, 1, 1, 0], [0, 1, 0, 0], [0.1, 0.9, 0.4, 0.2])
    print(example)
