# CKD Hyperparameter Optimization Study

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-CKD%20Prediction-green)
![Reproducible Research](https://img.shields.io/badge/Reproducible-Research-purple)
![MIT License](https://img.shields.io/badge/License-MIT-lightgrey)

Reproducible clinical machine learning pipeline comparing hyperparameter optimization strategies for CKD prediction with Springer-style reporting, generated figures, and verification artifacts.

Paper title: **Comparative Evaluation of Hyperparameter Optimization Techniques for Chronic Kidney Disease Prediction: A Multi-Dataset Study**

Author: **Tauqeer Sameer Bharde**  
Artificial Intelligence and Data Science Engineering  
SIES Graduate School of Technology, Mumbai, Maharashtra, India

## Results Snapshot

Public pipeline: **UCI Chronic Kidney Disease dataset**

- **SVM + Grid Search:** 100.0% held-out F1
- **SVM + Random Search:** 100.0% held-out F1
- **Random Forest + Random Search:** 98.99% held-out F1

These results reflect benchmark reproducibility-run performance, not clinical deployment readiness.

## Quick Start

```bash
git clone https://github.com/tauqxxr7/ckd-hpo-study.git
cd ckd-hpo-study
pip install -r requirements.txt
python src/main.py
```

Generated outputs:

- `results/performance_table.csv`
- `results/runtime_table.csv`
- `paper/figures/*.png`
- local trained models in `results/models/` ignored by Git

## Visual Outputs

![Workflow](paper/figures/workflow_diagram.png)

![F1-score comparison](paper/figures/f1_score_comparison.png)

![Runtime comparison](paper/figures/runtime_comparison.png)

![F1 vs runtime](paper/figures/f1_vs_runtime.png)

![HPO method comparison](paper/figures/hpo_method_comparison.png)

## Research Paper

- [Overleaf upload package](paper/overleaf_upload.zip)
- Manuscript source: [`paper/main.tex`](paper/main.tex)
- References: [`paper/references.bib`](paper/references.bib)

If `paper/ckd_hpo_manuscript.pdf` is not present, upload the `/paper` folder to Overleaf and compile using the Springer LNCS template.

## Verification Proof

Verification artifacts are available in [`docs/verification/`](docs/verification/):

- [`pipeline_run.txt`](docs/verification/pipeline_run.txt)
- [`syntax_check.txt`](docs/verification/syntax_check.txt)
- [`results_preview.md`](docs/verification/results_preview.md)
- [`figures_preview.md`](docs/verification/figures_preview.md)

## Key Contributions

- End-to-end reproducible CKD prediction pipeline
- Grid Search vs Random Search comparison
- Runtime/performance trade-off analysis
- Springer LNCS manuscript included
- Verification artifacts included
- Public UCI pipeline separated from credentialed eICU discussion

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

## Project Structure

```text
ckd-hpo-study/
|-- README.md
|-- data/
|   `-- README.md
|-- docs/
|   `-- verification/
|-- notebooks/
|   `-- ckd_hpo_experiments.ipynb
|-- paper/
|   |-- main.tex
|   |-- references.bib
|   |-- REVIEWER_CHECKLIST.md
|   |-- overleaf_upload.zip
|   `-- figures/
|-- results/
|   |-- performance_table.csv
|   `-- runtime_table.csv
|-- src/
|   |-- data_loader.py
|   |-- main.py
|   |-- preprocessing.py
|   |-- train_models.py
|   |-- optimize_hpo.py
|   |-- evaluate.py
|   `-- plot_results.py
|-- requirements.txt
|-- LICENSE
`-- .gitignore
```

## Technical Notes

- Modular ML pipeline with separate loading, preprocessing, training, HPO, evaluation, and plotting modules
- Leakage-safe preprocessing through scikit-learn pipelines fitted inside training folds
- Reproducible experiment execution using seed `42`
- Generated figures and CSV outputs committed for reviewer inspection
- Verification artifacts captured under `docs/verification/`
- CPU-only execution; no GPU required

## Datasets

- **UCI CKD:** public benchmark used by the runnable pipeline
- **eICU:** credentialed PhysioNet dataset discussed as a summarized extension; raw data are not distributed

Raw/private clinical data are excluded by `.gitignore`.

## Limitations

- UCI CKD is a small benchmark dataset
- eICU requires credentialed PhysioNet access
- No external clinical validation yet
- Not a diagnostic tool

## Citation

```text
Bharde, T. S. Comparative Evaluation of Hyperparameter Optimization Techniques
for Chronic Kidney Disease Prediction: A Multi-Dataset Study.
SIES Graduate School of Technology, Mumbai, Maharashtra, India.
```

## License

This repository is released under the [MIT License](LICENSE).
