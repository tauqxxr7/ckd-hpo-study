"""Run the end-to-end CKD HPO experiment pipeline."""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd

from data_loader import load_uci_ckd
from evaluate import evaluate_binary_classifier
from optimize_hpo import run_grid_search, run_random_search
from plot_results import generate_all_plots
from preprocessing import (
    build_model_pipeline,
    make_train_test_split,
    prepare_features_and_target,
)
from train_models import build_model


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
RANDOM_SEED = 42


def _model_configs() -> dict[str, dict[str, object]]:
    """Return CPU-friendly model and search-space definitions."""
    configs: dict[str, dict[str, object]] = {
        "Random Forest": {
            "builder_name": "random-forest",
            "grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 6],
                "model__min_samples_split": [2, 5],
            },
            "random": {
                "model__n_estimators": [100, 150, 200, 300],
                "model__max_depth": [None, 4, 6, 8],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
            "scale_numeric": False,
        },
        "SVM": {
            "builder_name": "svm",
            "grid": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale"],
            },
            "random": {
                "model__C": [0.1, 0.5, 1, 2, 5, 10],
                "model__kernel": ["linear", "rbf"],
                "model__gamma": ["scale", "auto"],
            },
            "scale_numeric": True,
        },
    }

    try:
        build_model("xgboost")
    except ImportError:
        print("Skipping XGBoost because the optional xgboost package is not installed.")
    else:
        configs["XGBoost"] = {
            "builder_name": "xgboost",
            "grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
            },
            "random": {
                "model__n_estimators": [100, 150, 200, 300],
                "model__max_depth": [2, 3, 4],
                "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
                "model__subsample": [0.8, 0.9, 1.0],
            },
            "scale_numeric": False,
        }
    return configs


def _evaluate_and_record(
    dataset: str,
    model_name: str,
    hpo_method: str,
    search_result,
    x_test,
    y_test,
    runtime_minutes: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate a fitted search object and build result rows."""
    evaluation = evaluate_binary_classifier(search_result.best_estimator_, x_test, y_test)
    cv_std = float(search_result.cv_results_["std_test_score"][search_result.best_index_])
    performance_row = {
        "dataset": dataset,
        "model": model_name,
        "hpo_method": hpo_method,
        "f1_score": round(evaluation.f1_score * 100, 2),
        "std_dev": round(cv_std * 100, 2),
        "precision": round(evaluation.precision * 100, 2),
        "recall": round(evaluation.recall * 100, 2),
    }
    runtime_row = {
        "dataset": dataset,
        "hpo_method": hpo_method,
        "runtime_minutes": round(runtime_minutes, 4),
    }
    return performance_row, runtime_row


def run_pipeline() -> None:
    """Run download, preprocessing, HPO, evaluation, persistence, and plotting."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading UCI CKD dataset...")
    raw_data = load_uci_ckd(download=True)
    x, y = prepare_features_and_target(raw_data)
    x_train, x_test, y_train, y_test = make_train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
    )

    performance_rows: list[dict[str, object]] = []
    runtime_rows: list[dict[str, object]] = []
    models = _model_configs()

    for model_name, config in models.items():
        print(f"Running HPO for {model_name}...")
        for hpo_method, search_fn, search_space_key in (
            ("Grid Search", run_grid_search, "grid"),
            ("Random Search", run_random_search, "random"),
        ):
            estimator = build_model(str(config["builder_name"]), random_state=RANDOM_SEED)
            pipeline = build_model_pipeline(
                estimator=estimator,
                features=x_train,
                scale_numeric=bool(config["scale_numeric"]),
            )
            start_time = time.perf_counter()
            if hpo_method == "Random Search":
                search_result = search_fn(
                    pipeline,
                    config[search_space_key],
                    x_train,
                    y_train,
                    n_iter=8,
                    scoring="f1",
                    n_splits=5,
                    random_state=RANDOM_SEED,
                )
            else:
                search_result = search_fn(
                    pipeline,
                    config[search_space_key],
                    x_train,
                    y_train,
                    scoring="f1",
                    n_splits=5,
                )
            runtime_minutes = (time.perf_counter() - start_time) / 60

            performance_row, runtime_row = _evaluate_and_record(
                dataset="UCI",
                model_name=model_name,
                hpo_method=hpo_method,
                search_result=search_result,
                x_test=x_test,
                y_test=y_test,
                runtime_minutes=runtime_minutes,
            )
            performance_rows.append(performance_row)
            runtime_rows.append(runtime_row)

            model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_{hpo_method.lower().replace(' ', '_')}.pkl"
            with model_path.open("wb") as model_file:
                pickle.dump(search_result.best_estimator_, model_file)
            print(
                f"  {hpo_method}: F1={performance_row['f1_score']}%, "
                f"precision={performance_row['precision']}%, recall={performance_row['recall']}%"
            )

    performance = pd.DataFrame(performance_rows)
    runtime = pd.DataFrame(runtime_rows)
    runtime_summary = (
        runtime.groupby(["dataset", "hpo_method"], as_index=False)["runtime_minutes"]
        .mean()
        .sort_values(["dataset", "runtime_minutes"])
    )
    performance_by_method = (
        performance.groupby(["dataset", "hpo_method"], as_index=False)["f1_score"]
        .max()
        .sort_values(["dataset", "f1_score"], ascending=[True, False])
    )
    runtime_summary["efficiency_rank"] = runtime_summary.groupby("dataset")[
        "runtime_minutes"
    ].rank(method="dense").astype(int)
    performance_by_method["performance_rank"] = performance_by_method.groupby("dataset")[
        "f1_score"
    ].rank(method="dense", ascending=False).astype(int)
    runtime_output = runtime_summary.merge(
        performance_by_method[["dataset", "hpo_method", "performance_rank"]],
        on=["dataset", "hpo_method"],
        how="left",
    )
    runtime_output = runtime_output[
        ["dataset", "hpo_method", "runtime_minutes", "performance_rank", "efficiency_rank"]
    ]

    performance_path = RESULTS_DIR / "performance_table.csv"
    runtime_path = RESULTS_DIR / "runtime_table.csv"
    performance.to_csv(performance_path, index=False)
    runtime_output.to_csv(runtime_path, index=False)
    print(f"Saved metrics to {performance_path}")
    print(f"Saved runtime summary to {runtime_path}")

    print("Generating figures...")
    for figure_path in generate_all_plots():
        print(f"Saved {figure_path}")


if __name__ == "__main__":
    run_pipeline()
