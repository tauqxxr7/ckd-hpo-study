# CKD HPO Study

Repository for the research paper:

**Comparative Evaluation of Hyperparameter Optimization Techniques for Chronic Kidney Disease Prediction: A Multi-Dataset Study**

Author: **Tauqeer Sameer Bharde**  
Artificial Intelligence & Data Science Engineering  
SIES Graduate School of Technology, Mumbai, India

## Project Overview

This repository provides a journal-ready research package for comparing hyperparameter optimization (HPO) strategies for chronic kidney disease (CKD) prediction across two clinical datasets. The study evaluates whether commonly used HPO methods improve predictive performance consistently across dataset scale and clinical context, while also considering runtime cost.

The package includes a Springer LNCS-style manuscript draft, reproducible experiment code, result summary tables, data access guidance, and reviewer-readiness documentation.

## Research Objective

The objective is to compare Grid Search, Random Search, Bayesian Optimization/TPE, CMA-ES, and Hyperband for tuning machine learning models used in CKD prediction. The evaluation focuses on predictive performance and computational efficiency rather than deployment readiness.

## Datasets

- **UCI Chronic Kidney Disease dataset**: A public tabular CKD benchmark dataset available from the UCI Machine Learning Repository.
- **eICU Collaborative Research Database**: A larger critical-care dataset requiring credentialed access through PhysioNet.

Raw clinical datasets are not committed to this repository. See [data/README.md](data/README.md) for access and ethics notes.

## Models

- Random Forest
- XGBoost
- Support Vector Machine (SVM)

## Hyperparameter Optimization Methods

- Grid Search
- Random Search
- Bayesian Optimization / Tree-structured Parzen Estimator (TPE)
- Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
- Hyperband

## Evaluation Metrics

- F1-score
- Precision
- Recall
- Runtime

Results are reported as mean ± standard deviation where repeated validation estimates are available.

## Key Results

- On the UCI CKD dataset, the best observed result was **SVM + Random Search = 99.33% ± 0.36% F1-score**.
- On the eICU dataset, the strongest summarized HPO result was **Random Search = 94.54% F1-score**.
- Random Search provided the best practical time-performance tradeoff across the summarized experiments.
- Grid Search was computationally expensive and showed degraded performance on the larger eICU setting, likely because exhaustive search over a fixed grid became inefficient under a constrained budget.

These results should be interpreted as experimental findings from the reported setup, not as evidence of clinical deployment readiness.

## Reproducibility

The intended reproducibility protocol is:

- Random seed: `42`
- Validation: 5-fold stratified cross-validation
- Search budget: same budget across HPO methods where applicable
- Preprocessing: fitted only on training folds to reduce data leakage risk
- Reporting: mean ± standard deviation across validation folds or repeated runs

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare datasets locally according to [data/README.md](data/README.md). Do not commit private clinical data.

Run the experiment modules as needed:

```bash
python src/train_models.py
python src/optimize_hpo.py
python src/evaluate.py
python src/plot_results.py
```

The current source files provide reusable experiment scaffolding and plotting utilities. Dataset-specific paths and feature definitions should be configured locally before full execution.

## Repository Structure

```text
ckd-hpo-study/
├── README.md
├── paper/
│   ├── main.tex
│   ├── references.bib
│   ├── REVIEWER_CHECKLIST.md
│   └── figures/
├── data/
│   └── README.md
├── notebooks/
│   └── ckd_hpo_experiments.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── optimize_hpo.py
│   ├── evaluate.py
│   └── plot_results.py
├── results/
│   ├── performance_table.csv
│   └── runtime_table.csv
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Citation

If you use this repository, please cite:

```text
Bharde, T. S. Comparative Evaluation of Hyperparameter Optimization Techniques
for Chronic Kidney Disease Prediction: A Multi-Dataset Study.
SIES Graduate School of Technology, Mumbai, India.
```

## Author

Tauqeer Sameer Bharde  
Artificial Intelligence & Data Science Engineering  
SIES Graduate School of Technology, Mumbai, India
