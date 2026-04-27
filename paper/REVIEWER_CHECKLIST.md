# Reviewer-Readiness Checklist

## TRIPOD-AI Alignment

- Prediction task, target outcome, candidate predictors, and model families are described.
- Validation design is stated as 5-fold stratified cross-validation.
- Hyperparameter optimization methods and evaluation metrics are reported.
- Results are summarized with mean ± standard deviation where applicable.

## PROBAST Risk-of-Bias Awareness

- Small-sample limitations of the UCI CKD dataset are acknowledged.
- eICU cohort construction and generalizability limitations are acknowledged.
- Model evaluation is not presented as clinical validation.
- Further calibration, temporal validation, and external validation are identified as required future work.

## Data Leakage Prevention

- Preprocessing is specified as fitted only on training folds.
- Train-validation splits are stratified.
- Raw clinical data are excluded from the repository.
- Dataset-specific preprocessing decisions should be documented before submission.

## External Validation Limitation

- The manuscript does not claim external validation.
- Independent-site validation is listed as future work.
- Deployment claims are avoided.

## Clinical Deployment Limitation

- The repository is for research reproducibility and method comparison only.
- The manuscript does not present the model as a diagnostic tool.
- Clinical use would require governance review, validation, calibration, monitoring, and clinician oversight.

## Ethical Use Statement

- UCI data should be obtained from the official UCI source.
- eICU data require PhysioNet credentialing and compliance with data use terms.
- Private, patient-level, and credentialed data must not be committed.
- Fairness, privacy, and explainability assessments should be performed before any translational study.
