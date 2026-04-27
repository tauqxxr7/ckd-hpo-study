# CKD HPO Study

Repository for the research paper:

**Comparative Evaluation of Hyperparameter Optimization Techniques for Chronic Kidney Disease Prediction: A Multi-Dataset Study**

Author: **Tauqeer Sameer Bharde**  
Artificial Intelligence and Data Science Engineering  
SIES Graduate School of Technology, Mumbai, Maharashtra, India

## Verification & Reproducibility Status

Latest verification: 2026-04-27, final repository audit.

Reproduce the public pipeline with:

```bash
pip install -r requirements.txt
python src/main.py
```

Generated outputs:

- `results/performance_table.csv`
- `results/runtime_table.csv`
- `paper/figures/f1_score_comparison.png`
- `paper/figures/runtime_comparison.png`
- `paper/figures/f1_vs_runtime.png`
- `paper/figures/hpo_method_comparison.png`

Verification proof files are stored in [`docs/verification/`](docs/verification/).

The public runnable pipeline reproduces UCI CKD experiments. eICU requires credentialed PhysioNet access and is not distributed or reproduced by the public pipeline. Trained model artifacts are generated locally in `results/models/` and ignored by Git.

## Repository Quality Checklist

- [x] End-to-end pipeline runs
- [x] Figures generated
- [x] Results CSV generated
- [x] Paper manuscript included
- [x] References included
- [x] Ethics statement included
- [x] Reproducibility statement included
- [x] Raw clinical data excluded
- [x] Reviewer checklist included

## Project Overview

This repository provides a journal-ready and runnable research package for comparing hyperparameter optimization (HPO) strategies for chronic kidney disease (CKD) prediction. The included end-to-end pipeline downloads the public UCI CKD dataset, preprocesses it, trains CPU-compatible machine learning models, runs Grid Search and Random Search, evaluates metrics, saves model artifacts, and regenerates result figures.

The manuscript also discusses an eICU-based extension, but raw eICU data are not distributed because access requires PhysioNet credentialing.

## Quick Start

```bash
git clone https://github.com/tauqxxr7/ckd-hpo-study.git
cd ckd-hpo-study
pip install -r requirements.txt
python src/main.py
```

Running `python src/main.py` will:

- download the UCI CKD dataset into `data/raw/`
- train Random Forest and SVM models, with XGBoost included only when installed
- run Grid Search and Random Search
- save metrics into `results/performance_table.csv`
- save runtime summaries into `results/runtime_table.csv`
- save trained model pickle files into `results/models/`
- generate figures in `paper/figures/`

The pipeline runs on CPU and does not require a GPU.

## Research Objective

The objective is to compare Grid Search, Random Search, Bayesian Optimization/TPE, CMA-ES, and Hyperband for tuning machine learning models used in CKD prediction. The runnable pipeline focuses on dependency-light Grid Search and Random Search so that new users can reproduce baseline results with one command.

## Datasets

- **UCI Chronic Kidney Disease dataset**: A public tabular CKD benchmark dataset available from the UCI Machine Learning Repository. The runnable pipeline downloads this dataset automatically.
- **eICU Collaborative Research Database**: A larger critical-care dataset requiring credentialed access through PhysioNet. It is discussed in the manuscript but not downloaded by the pipeline.

Raw clinical datasets are not committed to this repository. See [data/README.md](data/README.md) for access and ethics notes.

## Models

- Random Forest
- Support Vector Machine (SVM)
- XGBoost, optional if the `xgboost` package is available

## Hyperparameter Optimization Methods

The runnable pipeline executes:

- Grid Search
- Random Search

The repository also includes optional helper functions for:

- Bayesian Optimization / Tree-structured Parzen Estimator (TPE)
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- Hyperband

Optional HPO methods are wrapped so missing optional libraries do not break the main pipeline.

## Evaluation Metrics

- F1-score
- Precision
- Recall
- Runtime

Results are reported as percentages in the generated CSV files.

## Reproducibility

The intended reproducibility protocol is:

- Random seed: `42`
- Validation: 5-fold stratified cross-validation
- Preprocessing: fitted only on training folds through scikit-learn pipelines
- Reporting: test-set precision, recall, F1-score, and cross-validation standard deviation

## Repository Structure

```text
ckd-hpo-study/
|-- README.md
|-- docs/
|   `-- verification/
|-- paper/
|   |-- main.tex
|   |-- references.bib
|   |-- REVIEWER_CHECKLIST.md
|   `-- figures/
|-- data/
|   `-- README.md
|-- notebooks/
|   `-- ckd_hpo_experiments.ipynb
|-- src/
|   |-- data_loader.py
|   |-- main.py
|   |-- preprocessing.py
|   |-- train_models.py
|   |-- optimize_hpo.py
|   |-- evaluate.py
|   `-- plot_results.py
|-- results/
|   |-- performance_table.csv
|   `-- runtime_table.csv
|-- requirements.txt
|-- LICENSE
`-- .gitignore
```

## Citation

If you use this repository, please cite:

```text
Bharde, T. S. Comparative Evaluation of Hyperparameter Optimization Techniques
for Chronic Kidney Disease Prediction: A Multi-Dataset Study.
SIES Graduate School of Technology, Mumbai, Maharashtra, India.
```

## Author

Tauqeer Sameer Bharde  
Artificial Intelligence and Data Science Engineering  
SIES Graduate School of Technology, Mumbai, Maharashtra, India
