# Verification Report

Verification date: 2026-04-27  
Verifier environment: Windows PowerShell, Python 3.12

## Commands Run

```powershell
git pull --ff-only
pip install -r requirements.txt
python src/main.py
python -m py_compile (Get-ChildItem src -Filter *.py).FullName
python -m py_compile src/*.py
```

## Outputs Generated

- `results/performance_table.csv`
- `results/runtime_table.csv`
- `paper/figures/f1_score_comparison.png`
- `paper/figures/runtime_comparison.png`
- `paper/figures/f1_vs_runtime.png`
- `paper/figures/hpo_method_comparison.png`
- local trained model artifacts in `results/models/` ignored by Git

## Files Checked

- `README.md`
- `paper/main.tex`
- `paper/references.bib`
- `paper/figures/*.png`
- `src/*.py`
- `results/*.csv`
- `.gitignore`

## Proof Artifacts

- `environment_setup.txt`: dependency setup output.
- `pipeline_run.txt`: complete `python src/main.py` run output.
- `syntax_check.txt`: Windows-safe syntax-check result.
- `syntax_check_posix_attempt.txt`: POSIX-style wildcard attempt on Windows.
- `repo_tree.txt`: repository file inventory excluding ignored raw/model/cache files.
- `results_preview.md`: CSV result previews.
- `figures_preview.md`: generated figure list and file sizes.

Screenshots not available in this environment. Text proof artifacts are provided instead.

## Known Limitations

- The public pipeline reproduces UCI CKD experiments only.
- eICU data require credentialed PhysioNet access and are not distributed.
- Local LaTeX compilation was not available because no TeX engine was found in this environment.
- Springer `llncs.cls` is not included locally; use Overleaf or the official Springer LNCS template package.
- `python -m py_compile src/*.py` is a POSIX-style shell command; the local WSL/bash bridge was not usable in this environment, so the Windows-safe equivalent was used and passed.
- Optional packages such as XGBoost and advanced HPO libraries are not required for the core pipeline.

## Submission Readiness Verdict

The repository is publication/reviewer/recruiter ready as a reproducible code package for the public UCI CKD pipeline. For manuscript submission, compile `paper/main.tex` in Overleaf or a local Springer LNCS environment and review final journal formatting requirements.
