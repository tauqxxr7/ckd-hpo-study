"""Plot result summaries for the CKD HPO study."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".matplotlib-cache"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import pandas as pd
import seaborn as sns


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


def plot_hpo_method_comparison(
    performance_csv: Path = RESULTS_DIR / "performance_table.csv",
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generate a chart summarizing HPO method performance across datasets."""
    output_dir = _prepare_output_dir(output_dir)
    data = pd.read_csv(performance_csv)
    summary = (
        data.groupby(["dataset", "hpo_method"], as_index=False)
        .agg(mean_f1_score=("f1_score", "mean"), mean_std_dev=("std_dev", "mean"))
        .sort_values(["dataset", "mean_f1_score"], ascending=[True, False])
    )

    plt.figure(figsize=(12, 6))
    use_dodge = summary["dataset"].nunique() > 1
    sns.pointplot(
        data=summary,
        x="hpo_method",
        y="mean_f1_score",
        hue="dataset",
        dodge=0.35 if use_dodge else False,
        markers="o",
        linestyles="-",
    )
    plt.xlabel("HPO method")
    plt.ylabel("Mean F1-score across models (%)")
    plt.title("Grid Search and Random Search comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = output_dir / "hpo_method_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_workflow_diagram(output_dir: Path = FIGURES_DIR) -> Path:
    """Generate the experiment workflow diagram used in the manuscript."""
    output_dir = _prepare_output_dir(output_dir)
    steps = [
        "Dataset\nLoading",
        "Preprocessing",
        "Train/Test\nSplit",
        "HPO",
        "Model\nTraining",
        "Evaluation",
        "Results/\nFigures",
    ]
    x_positions = range(len(steps))

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.set_xlim(-0.6, len(steps) - 0.4)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for index, (x_position, label) in enumerate(zip(x_positions, steps)):
        ax.text(
            x_position,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.35,rounding_size=0.08",
                "facecolor": "#f4f7fb",
                "edgecolor": "#315a89",
                "linewidth": 1.4,
            },
        )
        if index < len(steps) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x_position + 0.38, 0.5),
                    (x_position + 0.72, 0.5),
                    arrowstyle="-|>",
                    mutation_scale=16,
                    linewidth=1.6,
                    color="#315a89",
                )
            )

    output_path = output_dir / "workflow_diagram.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def generate_all_plots() -> list[Path]:
    """Generate all summary figures."""
    return [
        plot_workflow_diagram(),
        plot_f1_comparison(),
        plot_runtime_comparison(),
        plot_f1_vs_runtime(),
        plot_hpo_method_comparison(),
    ]


if __name__ == "__main__":
    for figure_path in generate_all_plots():
        print(f"Saved {figure_path}")
