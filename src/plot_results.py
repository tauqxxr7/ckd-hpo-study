"""Plot result summaries for the CKD HPO study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "paper" / "figures"


def _prepare_output_dir(output_dir: Path = FIGURES_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def plot_f1_comparison(
    performance_csv: Path = RESULTS_DIR / "performance_table.csv",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generate a bar chart comparing F1-score by dataset, model, and HPO method."""
    output_dir = _prepare_output_dir(output_dir)
    data = pd.read_csv(performance_csv)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=data, x="hpo_method", y="f1_score", hue="dataset")
    plt.xlabel("HPO method")
    plt.ylabel("F1-score (%)")
    plt.title("F1-score comparison across HPO methods")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / "f1_score_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_runtime_comparison(
    runtime_csv: Path = RESULTS_DIR / "runtime_table.csv",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generate a runtime comparison chart."""
    output_dir = _prepare_output_dir(output_dir)
    data = pd.read_csv(runtime_csv)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x="hpo_method", y="runtime_minutes", hue="dataset")
    plt.xlabel("HPO method")
    plt.ylabel("Runtime (minutes)")
    plt.title("Runtime comparison across HPO methods")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / "runtime_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_f1_vs_runtime(
    performance_csv: Path = RESULTS_DIR / "performance_table.csv",
    runtime_csv: Path = RESULTS_DIR / "runtime_table.csv",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generate a scatter plot comparing F1-score and runtime."""
    output_dir = _prepare_output_dir(output_dir)
    performance = pd.read_csv(performance_csv)
    runtime = pd.read_csv(runtime_csv)

    best_by_dataset_method = (
        performance.groupby(["dataset", "hpo_method"], as_index=False)["f1_score"].max()
    )
    merged = best_by_dataset_method.merge(runtime, on=["dataset", "hpo_method"], how="inner")

    plt.figure(figsize=(9, 6))
    sns.scatterplot(
        data=merged,
        x="runtime_minutes",
        y="f1_score",
        hue="hpo_method",
        style="dataset",
        s=120,
    )
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("Best F1-score (%)")
    plt.title("F1-score versus runtime")
    plt.tight_layout()

    output_path = output_dir / "f1_vs_runtime.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def generate_all_plots() -> list[Path]:
    """Generate all summary figures."""
    return [
        plot_f1_comparison(),
        plot_runtime_comparison(),
        plot_f1_vs_runtime(),
    ]


if __name__ == "__main__":
    for figure_path in generate_all_plots():
        print(f"Saved {figure_path}")
