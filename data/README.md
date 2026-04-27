# Data Access and Ethics

Raw clinical datasets are not committed to this repository.

## UCI CKD Dataset

The UCI Chronic Kidney Disease dataset can be downloaded from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease

Users should review the dataset documentation, citation guidance, and permitted use terms before running experiments.

## eICU Dataset

The eICU Collaborative Research Database requires credentialed access through PhysioNet:

https://physionet.org/content/eicu-crd/

Users must complete the required training, credentialing, data use agreement, and institutional requirements before accessing or processing eICU data.

## Local Data Layout

Suggested local-only directories:

```text
data/raw/
data/private/
```

These paths are ignored by Git. Do not commit raw, private, credentialed, or patient-level clinical data.

## Ethical Use

This repository is intended for research reproducibility and method comparison only. It is not a clinical diagnostic product. Any clinical use would require proper governance, privacy review, bias assessment, calibration analysis, external validation, and clinician oversight.
